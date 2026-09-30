import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


@torch.no_grad()
def collect_outputs(model, loader, device, threshold=0.5, classifier_criterion=None):
    """Runs the model over ``loader`` and gathers labels, predictions, losses and router weights.

    Prototype similarity is the inference rule; the auxiliary classifier is only evaluated when a
    ``classifier_criterion`` is given (for reference).
    """
    model.eval()
    labels, proto_preds, proto_scores, cls_logits, router_weights = [], [], [], [], []
    proto_loss = cls_loss = 0.0

    for text, audio, vision, y in loader:
        text, audio, vision, y = (t.to(device) for t in (text, audio, vision, y))
        latents = model.encode(text, audio, vision)
        fused, weights = model.fuse(latents)

        preds, scores = model.prototype.predict(fused, threshold=threshold)
        proto_loss += model.prototype(fused, y).item()
        labels.append(y.cpu())
        proto_preds.append(preds.cpu())
        proto_scores.append(scores.cpu())
        if weights is not None:
            router_weights.append(weights.cpu())
        if classifier_criterion is not None:
            logits = model.classifier(latents)
            cls_loss += classifier_criterion(logits, y).item()
            cls_logits.append(logits.cpu())

    n_batches = len(loader)
    return {
        "labels": torch.cat(labels).numpy(),
        "proto_pred": torch.cat(proto_preds).numpy(),
        "proto_scores": torch.cat(proto_scores).numpy(),
        "proto_loss": proto_loss / n_batches,
        "cls_logits": torch.cat(cls_logits).numpy() if cls_logits else None,
        "cls_loss": cls_loss / n_batches if cls_logits else None,
        "router_weights": torch.cat(router_weights).numpy() if router_weights else None,
    }


def single_label_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def jaccard_accuracy(y_true, y_pred):
    """Mean per-sample |y ∩ ŷ| / |y ∪ ŷ| (a sample with empty y and ŷ scores 0)."""
    y_true, y_pred = y_true > 0, y_pred > 0
    intersection = (y_true & y_pred).sum(axis=1)
    union = np.maximum((y_true | y_pred).sum(axis=1), 1)
    return float((intersection / union).mean())


def multi_label_metrics(y_true, y_pred):
    metrics = {
        "exact_match": float((y_true == y_pred).all(axis=1).mean()),
        "jaccard_accuracy": jaccard_accuracy(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
    }
    for name, fn in (("precision", precision_score), ("recall", recall_score), ("f1", f1_score)):
        for average in ("micro", "macro", "samples"):
            metrics[f"{average}_{name}"] = fn(y_true, y_pred, average=average, zero_division=0)
    return metrics


def per_class_scores(y_true, y_pred):
    kwargs = dict(average=None, zero_division=0)
    return {
        "precision": precision_score(y_true, y_pred, **kwargs),
        "recall": recall_score(y_true, y_pred, **kwargs),
        "f1": f1_score(y_true, y_pred, **kwargs),
    }


def format_router_weights(weights):
    if weights is None:
        return "Router disabled (mean fusion)."
    mean, std = weights.mean(axis=0), weights.std(axis=0)
    return "\n".join(
        f"  {name:<7} mean {m:.3f}  std {s:.3f}" for name, m, s in zip(("Text", "Audio", "Vision"), mean, std)
    )
