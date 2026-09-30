import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020).

    Two samples are positives when they share a class. ``labels`` may be class indices
    ``[batch]`` or multi-hot vectors ``[batch, num_classes]`` (positives share at least one label).
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.shape[0]
        features = F.normalize(features, dim=1)
        logits = features @ features.T / self.temperature

        if labels.dim() == 2:
            positives = (labels @ labels.T > 0).float()
        else:
            labels = labels.view(-1, 1)
            positives = torch.eq(labels, labels.T).float()

        not_self = 1.0 - torch.eye(batch_size, device=features.device)
        positives = positives * not_self

        exp_logits = torch.exp(logits) * not_self
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (positives * log_prob).sum(1) / (positives.sum(1) + 1e-8)
        return -mean_log_prob_pos.mean()


def cross_modal_contrastive_loss(latents, temperature=0.07):
    """InfoNCE that aligns the modalities of the same sample, averaged over every modality pair."""
    names = list(latents)
    normalized = {m: F.normalize(latents[m], dim=1) for m in names}
    batch_size = normalized[names[0]].shape[0]
    positive_mask = torch.eye(batch_size, device=normalized[names[0]].device)

    losses = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            exp_sim = torch.exp(normalized[a] @ normalized[b].T / temperature)
            positive = (exp_sim * positive_mask).sum(1)
            losses.append(-torch.log(positive / (exp_sim.sum(1) + 1e-8)).mean())
    return sum(losses) / len(losses)


def router_entropy_loss(weights, eps=1e-8):
    """Negative routing entropy scaled by batch size; minimizing it keeps the router from collapsing."""
    entropy = -(weights * torch.log(weights + eps)).sum(dim=1)
    return -entropy.mean() * weights.size(0)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
