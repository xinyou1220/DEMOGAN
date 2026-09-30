import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

MODALITY_NAMES = ("Text", "Audio", "Vision")


def _annotate_matrix(ax, cm, fmt):
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, normalize=False, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    if normalize:
        cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(title=title, xlabel="Predicted label", ylabel="True label",
           xticks=range(len(class_names)), yticks=range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    _annotate_matrix(ax, cm, ".2f" if normalize else "d")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history, save_path, warmup_epochs=None, freeze_epoch=None):
    """One panel per recorded series; ``train_*`` are per-epoch, ``val_*`` follow ``val_epoch``."""
    series = {k: v for k, v in history.items() if k not in ("val_epoch", "val_score") and len(v) > 0}
    n_cols = 3
    n_rows = math.ceil(len(series) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for ax, (name, values) in zip(axes.flat, series.items()):
        x = history["val_epoch"] if name.startswith("val_") else range(1, len(values) + 1)
        ax.plot(list(x), values, linewidth=1.5)
        ax.set_title(name.replace("_", " "), fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        if warmup_epochs:
            ax.axvline(warmup_epochs, color="tab:blue", linestyle="--", alpha=0.6)
        if freeze_epoch:
            ax.axvline(freeze_epoch, color="tab:red", linestyle="--", alpha=0.6)
    for ax in list(axes.flat)[len(series):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multilabel_summary(y_true, y_pred, metrics, per_class_f1, router_weights, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    ax.bar(range(len(per_class_f1)), per_class_f1, color="steelblue")
    ax.set(title="Per-class F1", xlabel="Class", ylabel="F1", ylim=(0, 1), xticks=range(len(per_class_f1)))
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    names = ["exact_match", "micro_f1", "macro_f1", "samples_f1"]
    values = [metrics[n] for n in names]
    ax.bar(names, values, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"])
    ax.set(title="Overall metrics", ylabel="Score", ylim=(0, 1))
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    if router_weights is not None:
        mean, std = router_weights.mean(axis=0), router_weights.std(axis=0)
        ax.bar(range(3), mean, yerr=std, capsize=5, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
        ax.set(title="Router modality weights (mean ± std)", ylabel="Weight",
               xticks=range(3), ylim=(0, float((mean + std).max()) + 0.1))
        ax.set_xticklabels(MODALITY_NAMES)
    else:
        ax.text(0.5, 0.5, "Router disabled", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    ax = axes[1, 1]
    x = np.arange(y_true.shape[1])
    ax.bar(x - 0.175, y_true.sum(axis=0), 0.35, label="True", color="steelblue")
    ax.bar(x + 0.175, y_pred.sum(axis=0), 0.35, label="Predicted", color="coral")
    ax.set(title="Label counts: true vs predicted", xlabel="Class", ylabel="Count", xticks=x)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_confusion(y_true, y_pred, save_path):
    num_classes = y_true.shape[1]
    n_cols = 4
    n_rows = math.ceil(num_classes / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    for c, ax in enumerate(axes.flat):
        if c >= num_classes:
            ax.axis("off")
            continue
        cm = confusion_matrix(y_true[:, c], y_pred[:, c], labels=[0, 1])
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(title=f"Class {c}", xlabel="Predicted", ylabel="True", xticks=[0, 1], yticks=[0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])
        _annotate_matrix(ax, cm, "d")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
