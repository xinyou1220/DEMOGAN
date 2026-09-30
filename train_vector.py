"""Single-label DEMOGAN on utterance-level feature vectors (JSON / JSONL).

Example:
    python train_vector.py --train data/train.jsonl --val data/val.jsonl --test data/test.jsonl
    python train_vector.py ... --modality-masking   # randomly drop modalities during training
"""
import argparse
import os

import numpy as np
import torch

from demogan.data import (
    VectorFeatureDataset,
    inverse_frequency_weights,
    make_loaders,
    print_class_distribution,
)
from demogan.evaluation import collect_outputs, format_router_weights, single_label_metrics
from demogan.losses import FocalLoss
from demogan.models import DEMOGAN
from demogan.plotting import plot_confusion_matrix, plot_training_curves
from demogan.trainer import ModalityMasking, TrainConfig, train_joint
from demogan.utils import create_run_dir, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    data = p.add_argument_group("data")
    data.add_argument("--train", required=True, help="training split (.json or .jsonl)")
    data.add_argument("--val", required=True, help="validation split")
    data.add_argument("--test", required=True, help="test split")
    data.add_argument("--num-classes", type=int, default=7)
    data.add_argument("--batch-size", type=int, default=128)
    data.add_argument("--num-workers", type=int, default=4)

    model = p.add_argument_group("model")
    model.add_argument("--latent-dim", type=int, default=512)
    model.add_argument("--no-router", action="store_true", help="fuse modalities by averaging")
    model.add_argument("--router-temperature", type=float, default=0.05)
    model.add_argument("--prototype-temperature", type=float, default=0.1)
    model.add_argument("--classifier-d-model", type=int, default=64)
    model.add_argument("--classifier-heads", type=int, default=2)
    model.add_argument("--classifier-dropout", type=float, default=0.6)

    train = p.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--warmup-epochs", type=int, default=20)
    train.add_argument("--freeze-gan-epoch", type=int, default=70, help="use 0 to never freeze")
    train.add_argument("--no-classifier-after-freeze", action="store_true")
    train.add_argument("--lr-gan", type=float, default=2e-4)
    train.add_argument("--lr-classifier", type=float, default=1e-3)
    train.add_argument("--lr-prototype", type=float, default=1e-3)
    train.add_argument("--lr-router", type=float, default=1e-3)
    train.add_argument("--lambda-intra", type=float, default=0.5)
    train.add_argument("--lambda-cross", type=float, default=0.3)
    train.add_argument("--lambda-prototype", type=float, default=1.0)
    train.add_argument("--lambda-classifier", type=float, default=1.0)
    train.add_argument("--lambda-router-entropy", type=float, default=0.5)
    train.add_argument("--contrastive-temperature", type=float, default=0.07)
    train.add_argument("--seed", type=int, default=16)

    masking = p.add_argument_group("modality masking")
    masking.add_argument("--modality-masking", action="store_true")
    masking.add_argument("--mask-ratio", type=float, default=0.3)
    masking.add_argument("--mask-start-epoch", type=int, default=1)
    masking.add_argument("--mask-allow-all", action="store_true",
                         help="drop each modality independently instead of always keeping one")

    p.add_argument("--output-dir", default="runs")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = create_run_dir(args.output_dir, "vector")
    save_json(vars(args), os.path.join(run_dir, "config.json"))
    print(f"Device: {device} | outputs: {run_dir}")

    splits = {name: VectorFeatureDataset.from_file(path)
              for name, path in (("train", args.train), ("val", args.val), ("test", args.test))}
    for name, ds in splits.items():
        print_class_distribution(ds.labels, name)

    set_seed(args.seed)
    train_loader, val_loader, test_loader = make_loaders(
        splits["train"], splits["val"], splits["test"], args.batch_size, args.num_workers
    )

    model = DEMOGAN(
        splits["train"].input_dims,
        args.num_classes,
        latent_dim=args.latent_dim,
        use_router=not args.no_router,
        router_temperature=args.router_temperature,
        prototype_temperature=args.prototype_temperature,
        classifier_d_model=args.classifier_d_model,
        classifier_heads=args.classifier_heads,
        classifier_dropout=args.classifier_dropout,
    ).to(device)

    class_weights = inverse_frequency_weights(splits["train"].labels, args.num_classes).to(device)
    classifier_criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    def evaluate(model, loader, evaluate_classifier=True):
        out = collect_outputs(model, loader, device,
                              classifier_criterion=classifier_criterion if evaluate_classifier else None)
        proto = single_label_metrics(out["labels"], out["proto_pred"])
        metrics = {"score": proto["accuracy"], "prototype_loss": out["proto_loss"],
                   "prototype_acc": proto["accuracy"], "prototype_macro_f1": proto["macro_f1"]}
        if out["cls_logits"] is not None:
            cls = single_label_metrics(out["labels"], out["cls_logits"].argmax(axis=1))
            metrics.update(classifier_loss=out["cls_loss"], classifier_acc=cls["accuracy"],
                           classifier_macro_f1=cls["macro_f1"])
        if out["router_weights"] is not None:
            for name, w in zip(("text", "audio", "vision"), out["router_weights"].mean(axis=0)):
                metrics[f"router_{name}"] = float(w)
        return metrics

    cfg = TrainConfig(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        freeze_gan_epoch=args.freeze_gan_epoch or None,
        train_classifier_after_freeze=not args.no_classifier_after_freeze,
        lr_gan=args.lr_gan,
        lr_classifier=args.lr_classifier,
        lr_prototype=args.lr_prototype,
        lr_router=args.lr_router,
        lambda_intra=args.lambda_intra,
        lambda_cross=args.lambda_cross,
        lambda_prototype=args.lambda_prototype,
        lambda_classifier=args.lambda_classifier,
        lambda_router_entropy=args.lambda_router_entropy,
        contrastive_temperature=args.contrastive_temperature,
    )
    masking = None
    if args.modality_masking:
        masking = ModalityMasking(args.mask_ratio, args.mask_start_epoch, not args.mask_allow_all)

    history = train_joint(model, train_loader, val_loader, device, cfg, classifier_criterion,
                          evaluate, run_dir, masking)

    checkpoint = torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']}")

    out = collect_outputs(model, test_loader, device, classifier_criterion=classifier_criterion)
    proto = single_label_metrics(out["labels"], out["proto_pred"])
    cls = single_label_metrics(out["labels"], out["cls_logits"].argmax(axis=1))

    lines = ["Test results (inference = prototype similarity; classifier shown for reference)", ""]
    for title, m in (("Prototype", proto), ("Classifier", cls)):
        lines += [f"{title}:"] + [f"  {k:<12} {v:.4f}" for k, v in m.items()] + [""]
    lines += ["Router modality weights:", format_router_weights(out["router_weights"])]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(run_dir, "test_results.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    save_json({"prototype": proto, "classifier": cls, "best_epoch": checkpoint["epoch"]},
              os.path.join(run_dir, "test_metrics.json"))

    class_names = [str(c) for c in range(args.num_classes)]
    for title, y_pred in (("prototype", out["proto_pred"]), ("classifier", out["cls_logits"].argmax(axis=1))):
        for normalize in (False, True):
            suffix = "_normalized" if normalize else ""
            plot_confusion_matrix(out["labels"], y_pred, class_names,
                                  os.path.join(run_dir, f"confusion_matrix_{title}{suffix}.png"),
                                  normalize=normalize, title=f"{title.capitalize()} confusion matrix")
    plot_training_curves(history, os.path.join(run_dir, "training_curves.png"),
                         args.warmup_epochs, cfg.freeze_gan_epoch)
    np.save(os.path.join(run_dir, "learned_prototypes.npy"), model.prototype.normalized_prototypes().cpu().numpy())
    print(f"\nAll outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
