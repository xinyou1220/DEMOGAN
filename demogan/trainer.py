"""Three-stage joint training shared by the vector and sequence pipelines.

Stages (per epoch):
  1. warmup  (epoch <= warmup_epochs): latent GAN + contrastive losses only.
  2. joint   : GAN + contrastive + router + prototype + auxiliary classifier, end to end.
  3. frozen  (epoch >= freeze_gan_epoch): the GAN is frozen; prototype + router (and optionally the
     classifier) keep training on fixed latents.
"""
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from .losses import SupervisedContrastiveLoss, cross_modal_contrastive_loss, router_entropy_loss
from .models import MODALITIES


@dataclass
class TrainConfig:
    epochs: int = 100
    warmup_epochs: int = 20
    freeze_gan_epoch: Optional[int] = None
    train_classifier_after_freeze: bool = True
    lr_gan: float = 2e-4
    lr_classifier: float = 1e-3
    lr_prototype: float = 1e-3
    lr_router: float = 1e-3
    betas: tuple = (0.5, 0.999)
    lambda_intra: float = 0.5
    lambda_cross: float = 0.3
    lambda_prototype: float = 1.0
    lambda_classifier: float = 1.0
    lambda_router_entropy: float = 0.1
    contrastive_temperature: float = 0.07
    log_every: int = 100


@dataclass
class ModalityMasking:
    """Randomly zeroes whole modality latents during training (robustness to missing modalities).

    With ``keep_one_modality`` a sample masks 0 / 1 / 2 modalities with probabilities
    ``1 - ratio`` / ``0.6 * ratio`` / ``0.4 * ratio``; otherwise every modality is dropped
    independently with probability ``ratio``.
    """

    ratio: float = 0.3
    start_epoch: int = 1
    keep_one_modality: bool = True

    def __call__(self, latents):
        batch_size = next(iter(latents.values())).size(0)
        device = next(iter(latents.values())).device
        n_mod = len(MODALITIES)
        if self.keep_one_modality:
            probs = torch.tensor([1 - self.ratio, 0.6 * self.ratio, 0.4 * self.ratio], device=device)
            num_masked = torch.multinomial(probs, batch_size, replacement=True)
            # Rank modalities by a random score; the first `num_masked` of each row are dropped.
            ranks = torch.rand(batch_size, n_mod, device=device).argsort(dim=1).argsort(dim=1)
            drop = ranks < num_masked.unsqueeze(1)
        else:
            drop = torch.rand(batch_size, n_mod, device=device) < self.ratio
        keep = (~drop).float()
        return {m: latents[m] * keep[:, i : i + 1] for i, m in enumerate(MODALITIES)}


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def train_joint(
    model,
    train_loader,
    val_loader,
    device,
    cfg: TrainConfig,
    classifier_criterion: nn.Module,
    evaluate_fn: Callable,
    save_dir: str,
    masking: Optional[ModalityMasking] = None,
):
    """Trains ``model`` and keeps the checkpoint with the highest ``evaluate_fn(...)['score']``.

    ``evaluate_fn(model, loader, evaluate_classifier)`` must return a dict of floats containing
    ``score``; every entry is appended to the returned history under ``val_<key>`` and the matching
    epoch numbers under ``val_epoch`` (validation is skipped during warmup).
    """
    gan = model.gan
    opt_encoder = torch.optim.Adam(gan.encoder_parameters(), lr=cfg.lr_gan, betas=cfg.betas)
    opt_disc = torch.optim.Adam(gan.discriminator.parameters(), lr=cfg.lr_gan, betas=cfg.betas)
    opt_classifier = torch.optim.Adam(model.classifier.parameters(), lr=cfg.lr_classifier)
    opt_prototype = torch.optim.Adam(model.prototype.parameters(), lr=cfg.lr_prototype)
    opt_router = torch.optim.Adam(model.router.parameters(), lr=cfg.lr_router) if model.use_router else None

    bce = nn.BCEWithLogitsLoss()
    supcon = SupervisedContrastiveLoss(cfg.contrastive_temperature)
    history = defaultdict(list)
    best_score = float("-inf")
    gan_frozen = False
    zero = torch.zeros((), device=device)

    for epoch in range(1, cfg.epochs + 1):
        is_warmup = epoch <= cfg.warmup_epochs
        if cfg.freeze_gan_epoch is not None and epoch >= cfg.freeze_gan_epoch and not gan_frozen:
            set_requires_grad(gan, False)
            gan_frozen = True
            print(f"\nEpoch {epoch}: latent GAN frozen; training prototype/router on fixed latents.\n")
        train_classifier = not is_warmup and (not gan_frozen or cfg.train_classifier_after_freeze)
        stage = "warmup" if is_warmup else ("frozen" if gan_frozen else "joint")

        gan.train(not gan_frozen)
        model.classifier.train(not gan_frozen or cfg.train_classifier_after_freeze)
        model.prototype.train(not is_warmup)
        if model.use_router:
            model.router.train(not is_warmup)

        sums = defaultdict(float)
        for step, (text, audio, vision, labels) in enumerate(train_loader):
            text, audio, vision, labels = (t.to(device) for t in (text, audio, vision, labels))
            batch_size = text.size(0)

            # 1) Discriminator: prior samples are real, encoder latents are fake.
            loss_disc = zero
            if not gan_frozen:
                opt_disc.zero_grad()
                with torch.no_grad():
                    fake = gan.encode(text, audio, vision)
                d_real = gan.discriminator(torch.randn(batch_size, gan.latent_dim, device=device))
                loss_fake = sum(
                    bce(d, torch.zeros_like(d)) for d in (gan.discriminator(fake[m]) for m in MODALITIES)
                ) / len(MODALITIES)
                loss_disc = (bce(d_real, torch.ones_like(d_real)) + loss_fake) / 2
                loss_disc.backward()
                opt_disc.step()

            # 2) Encoders, router, prototypes and classifier.
            for opt in (opt_encoder, opt_prototype, opt_router, opt_classifier):
                if opt is not None:
                    opt.zero_grad()

            with torch.set_grad_enabled(not gan_frozen):
                latents = gan.encode(text, audio, vision)
            if masking is not None and epoch >= masking.start_epoch:
                latents = masking(latents)

            loss_adv = loss_contrastive = zero
            if not gan_frozen:
                loss_adv = sum(
                    bce(d, torch.ones_like(d)) for d in (gan.discriminator(latents[m]) for m in MODALITIES)
                )
                loss_intra = sum(supcon(latents[m], labels) for m in MODALITIES) / len(MODALITIES)
                loss_cross = cross_modal_contrastive_loss(latents, cfg.contrastive_temperature)
                loss_contrastive = cfg.lambda_intra * loss_intra + cfg.lambda_cross * loss_cross

            loss_prototype = loss_classifier = loss_entropy = zero
            weights = None
            if not is_warmup:
                fused, weights = model.fuse(latents)
                if weights is not None:
                    loss_entropy = router_entropy_loss(weights)
                loss_prototype = model.prototype(fused, labels)
                if train_classifier:
                    loss_classifier = classifier_criterion(model.classifier(latents), labels)

            head_loss = (
                cfg.lambda_prototype * loss_prototype
                + cfg.lambda_classifier * loss_classifier
                + cfg.lambda_router_entropy * loss_entropy
            )
            if is_warmup:
                if not gan_frozen:
                    (loss_adv + loss_contrastive).backward()
                    opt_encoder.step()
            else:
                total = head_loss if gan_frozen else loss_adv + loss_contrastive + head_loss
                total.backward()
                if not gan_frozen:
                    opt_encoder.step()
                opt_prototype.step()
                if opt_router is not None:
                    opt_router.step()
                if train_classifier:
                    opt_classifier.step()

            for key, value in (
                ("discriminator", loss_disc),
                ("adversarial", loss_adv),
                ("contrastive", loss_contrastive),
                ("prototype", loss_prototype),
                ("classifier", loss_classifier),
                ("router_entropy", loss_entropy),
            ):
                sums[key] += value.item()

            if step % cfg.log_every == 0:
                router_info = ""
                if weights is not None:
                    w = weights.mean(dim=0)
                    router_info = f" | router T={w[0]:.2f} A={w[1]:.2f} V={w[2]:.2f}"
                print(
                    f"[{stage}] epoch {epoch:03d}/{cfg.epochs} step {step:04d}/{len(train_loader)} "
                    f"D {loss_disc.item():.4f} | adv {loss_adv.item():.4f} | "
                    f"contr {loss_contrastive.item():.4f} | proto {loss_prototype.item():.4f} | "
                    f"cls {loss_classifier.item():.4f}{router_info}"
                )

        for key, value in sums.items():
            history[f"train_{key}"].append(value / len(train_loader))

        if is_warmup:
            continue

        val_metrics = evaluate_fn(model, val_loader, evaluate_classifier=train_classifier)
        history["val_epoch"].append(epoch)
        for key, value in val_metrics.items():
            history[f"val_{key}"].append(value)
        summary = " | ".join(f"{k} {v:.4f}" for k, v in val_metrics.items())
        print(f"\n[{stage}] epoch {epoch}/{cfg.epochs} validation: {summary}\n")

        if val_metrics["score"] > best_score:
            best_score = val_metrics["score"]
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "val_metrics": val_metrics},
                os.path.join(save_dir, "best_model.pth"),
            )
            print(f"Saved best model (val score {best_score:.4f})\n")

    return dict(history)
