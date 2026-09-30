"""Multi-label DEMOGAN on sequence features (M3ED ``.pt`` file in CARAT format).

Example:
    python train_sequence.py --data data/m3ed_data_60.pt
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from demogan.data import M3EDSequenceDataset, make_loaders
from demogan.evaluation import collect_outputs, format_router_weights, multi_label_metrics, per_class_scores
from demogan.models import DEMOGAN
from demogan.plotting import plot_multilabel_summary, plot_per_class_confusion, plot_training_curves
from demogan.trainer import TrainConfig, train_joint
from demogan.utils import create_run_dir, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    data = p.add_argument_group("data")
    data.add_argument("--data", required=True, help=".pt file with train/valid/test splits")
    data.add_argument("--num-classes", type=int, default=7)
    data.add_argument("--label-offset", type=int, default=4, help="token id of the first emotion in 'tgt'")
    data.add_argument("--batch-size", type=int, default=32)
    data.add_argument("--num-workers", type=int, default=4)

    model = p.add_argument_group("model")
    model.add_argument("--latent-dim", type=int, default=512)
    model.add_argument("--pooling", choices=("mean", "max", "first", "last"), default="max")
    model.add_argument("--discriminator-dropout", type=float, default=0.2)
    model.add_argument("--no-router", action="store_true", help="fuse modalities by averaging")
    model.add_argument("--router-temperature", type=float, default=0.1)
    model.add_argument("--prototype-temperature", type=float, default=0.1)
    model.add_argument("--threshold", type=float, default=0.5, help="multi-label decision threshold")
    model.add_argument("--classifier-d-model", type=int, default=256)
    model.add_argument("--classifier-heads", type=int, default=8)
    model.add_argument("--classifier-dropout", type=float, default=0.3)

    train = p.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--warmup-epochs", type=int, default=15)
    train.add_argument("--freeze-gan-epoch", type=int, default=50, help="use 0 to never freeze")
    train.add_argument("--no-classifier-after-freeze", action="store_true")
    train.add_argument("--lr-gan", type=float, default=1e-4)
    train.add_argument("--lr-classifier", type=float, default=1e-4)
    train.add_argument("--lr-prototype", type=float, default=1e-4)
    train.add_argument("--lr-router", type=float, default=1e-4)
    train.add_argument("--lambda-intra", type=float, default=1.0)
    train.add_argument("--lambda-cross", type=float, default=1.0)
    train.add_argument("--lambda-prototype", type=float, default=1.0)
    train.add_argument("--lambda-classifier", type=float, default=1.0)
    train.add_argument("--lambda-router-entropy", type=float, default=0.01)
    train.add_argument("--contrastive-temperature", type=float, default=0.07)
    train.add_argument("--seed", type=int, default=222)

    p.add_argument("--output-dir", default="runs")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = create_run_dir(args.output_dir, "sequence")
    save_json(vars(args), os.path.join(run_dir, "config.json"))
    print(f"Device: {device} | outputs: {run_dir}")

    splits = M3EDSequenceDataset.load_splits(args.data, num_classes=args.num_classes,
                                             label_offset=args.label_offset)
    print("Split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    train_loader, val_loader, test_loader = make_loaders(
        splits["train"], splits["valid"], splits["test"], args.batch_size, args.num_workers
    )

    model = DEMOGAN(
        splits["train"].input_dims,
        args.num_classes,
        latent_dim=args.latent_dim,
        multi_label=True,
        pooling_type=args.pooling,
        discriminator_dropout=args.discriminator_dropout,
        use_router=not args.no_router,
        router_temperature=args.router_temperature,
        prototype_temperature=args.prototype_temperature,
        classifier_d_model=args.classifier_d_model,
        classifier_heads=args.classifier_heads,
        classifier_dropout=args.classifier_dropout,
    ).to(device)

    def evaluate(model, loader, evaluate_classifier=False):
        out = collect_outputs(model, loader, device, threshold=args.threshold)
        exact_match = float((out["proto_pred"] == out["labels"]).all(axis=1).mean())
        return {"score": exact_match, "prototype_loss": out["proto_loss"], "exact_match": exact_match}

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
    history = train_joint(model, train_loader, val_loader, device, cfg, nn.BCEWithLogitsLoss(),
                          evaluate, run_dir)

    checkpoint = torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']}")

    out = collect_outputs(model, test_loader, device, threshold=args.threshold)
    y_true, y_pred = out["labels"], out["proto_pred"]
    metrics = multi_label_metrics(y_true, y_pred)
    per_class = per_class_scores(y_true, y_pred)

    lines = ["Test results (prototype inference, multi-label)", ""]
    lines += [f"  {k:<18} {v:.4f}" for k, v in metrics.items()]
    lines += ["", f"{'Class':<8}{'Precision':<12}{'Recall':<12}{'F1':<12}"]
    lines += [f"{c:<8}{per_class['precision'][c]:<12.4f}{per_class['recall'][c]:<12.4f}{per_class['f1'][c]:<12.4f}"
              for c in range(args.num_classes)]
    lines += ["", "Router modality weights:", format_router_weights(out["router_weights"])]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(run_dir, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    np.savez(os.path.join(run_dir, "test_results.npz"), y_true=y_true, y_pred=y_pred,
             y_logits=out["proto_scores"], **metrics, **{f"per_class_{k}": v for k, v in per_class.items()})
    plot_multilabel_summary(y_true, y_pred, metrics, per_class["f1"], out["router_weights"],
                            os.path.join(run_dir, "test_summary.png"))
    plot_per_class_confusion(y_true, y_pred, os.path.join(run_dir, "confusion_matrices.png"))
    plot_training_curves(history, os.path.join(run_dir, "training_curves.png"),
                         args.warmup_epochs, cfg.freeze_gan_epoch)
    print(f"\nAll outputs saved to {run_dir}")


if __name__ == "__main__":
    main()
