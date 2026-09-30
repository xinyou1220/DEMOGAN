"""Network components for DEMOGAN.

All modules operate on the three modalities in a fixed order: text, audio, vision.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ("text", "audio", "vision")


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return self.act(x + residual)


class ModalityEncoder(nn.Module):
    """Residual MLP that maps a feature vector (or every step of a sequence) into the latent space.

    Accepts ``[batch, in_dim]`` or ``[batch, seq_len, in_dim]`` and keeps the leading shape.
    """

    def __init__(self, in_dim, latent_dim=512, hidden_dim=1024, num_blocks=3):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        leading_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        x = torch.relu(self.fc_in(x))
        x = self.blocks(x)
        x = self.fc_out(x)
        return x.reshape(*leading_shape, -1)


class SequencePooling(nn.Module):
    """Collapses ``[batch, seq_len, dim]`` to ``[batch, dim]``; 2-D inputs pass through unchanged."""

    POOLING_TYPES = ("mean", "max", "first", "last")

    def __init__(self, pooling_type="mean"):
        super().__init__()
        if pooling_type not in self.POOLING_TYPES:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        self.pooling_type = pooling_type

    def forward(self, x):
        if x.dim() == 2:
            return x
        if self.pooling_type == "mean":
            return x.mean(dim=1)
        if self.pooling_type == "max":
            return x.max(dim=1).values
        if self.pooling_type == "first":
            return x[:, 0]
        return x[:, -1]


class LatentDiscriminator(nn.Module):
    """Distinguishes samples of the N(0, I) prior from encoder outputs."""

    def __init__(self, latent_dim=512, hidden_dim=512, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2, inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class TriModalLatentGAN(nn.Module):
    """Three modality encoders that share one latent discriminator (adversarial autoencoder style)."""

    def __init__(self, input_dims, latent_dim=512, pooling_type="mean", discriminator_dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoders = nn.ModuleDict(
            {m: ModalityEncoder(input_dims[m], latent_dim) for m in MODALITIES}
        )
        self.pooling = SequencePooling(pooling_type)
        self.discriminator = LatentDiscriminator(latent_dim, dropout=discriminator_dropout)

    def encode(self, text, audio, vision):
        """Returns one pooled latent vector ``[batch, latent_dim]`` per modality."""
        inputs = dict(zip(MODALITIES, (text, audio, vision)))
        return {m: self.pooling(self.encoders[m](inputs[m])) for m in MODALITIES}

    def encoder_parameters(self):
        return self.encoders.parameters()


class RouterNetwork(nn.Module):
    """Predicts per-sample modality weights ``W = softmax(G([z_t; z_a; z_v]) / temperature)``."""

    def __init__(self, latent_dim, num_modalities=3, temperature=0.1):
        super().__init__()
        input_dim = latent_dim * num_modalities
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.BatchNorm1d(input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 4, num_modalities),
        )

    def forward(self, latents):
        logits = self.net(torch.cat(latents, dim=1))
        return F.softmax(logits / self.temperature, dim=1)


class SelfAttentionClassifier(nn.Module):
    """Auxiliary classifier: self-attention over the three modality tokens, then an MLP head."""

    def __init__(self, latent_dim=512, d_model=64, n_heads=2, dropout=0.6, num_classes=7):
        super().__init__()
        self.proj = nn.ModuleDict({m: nn.Linear(latent_dim, d_model) for m in MODALITIES})
        self.proj_norm = nn.ModuleDict({m: nn.LayerNorm(d_model) for m in MODALITIES})
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, num_classes)
        )

    def forward(self, latents):
        tokens = torch.stack(
            [self.proj_norm[m](F.relu(self.proj[m](latents[m]))) for m in MODALITIES], dim=1
        )
        attn_out, _ = self.self_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        pooled = tokens.mean(dim=1)
        fused = self.norm2(pooled + self.ffn(pooled))
        return self.head(fused)


class PrototypeLearning(nn.Module):
    """One learnable prototype per class; logits are cosine similarities divided by a temperature.

    Single-label mode trains with cross-entropy and predicts the most similar prototype.
    Multi-label mode trains with binary cross-entropy and predicts every class whose
    ``sigmoid(logit)`` exceeds ``threshold``.
    """

    def __init__(self, num_classes=7, latent_dim=512, temperature=0.1, multi_label=False):
        super().__init__()
        self.temperature = temperature
        self.multi_label = multi_label
        self.prototypes = nn.Parameter(torch.empty(num_classes, latent_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def similarities(self, features):
        return F.normalize(features, dim=1) @ F.normalize(self.prototypes, dim=1).T

    def forward(self, features, labels):
        logits = self.similarities(features) / self.temperature
        if self.multi_label:
            return F.binary_cross_entropy_with_logits(logits, labels)
        return F.cross_entropy(logits, labels)

    def predict(self, features, threshold=0.5):
        similarities = self.similarities(features)
        if self.multi_label:
            logits = similarities / self.temperature
            return (torch.sigmoid(logits) > threshold).float(), logits
        return similarities.argmax(dim=1), similarities

    def normalized_prototypes(self):
        return F.normalize(self.prototypes, dim=1).detach()


class DEMOGAN(nn.Module):
    """Bundles the latent GAN, router, prototype head and auxiliary classifier."""

    def __init__(
        self,
        input_dims,
        num_classes,
        latent_dim=512,
        multi_label=False,
        pooling_type="mean",
        discriminator_dropout=0.0,
        use_router=True,
        router_temperature=0.1,
        prototype_temperature=0.1,
        classifier_d_model=64,
        classifier_heads=2,
        classifier_dropout=0.6,
    ):
        super().__init__()
        self.gan = TriModalLatentGAN(input_dims, latent_dim, pooling_type, discriminator_dropout)
        self.router = RouterNetwork(latent_dim, len(MODALITIES), router_temperature) if use_router else None
        self.prototype = PrototypeLearning(num_classes, latent_dim, prototype_temperature, multi_label)
        self.classifier = SelfAttentionClassifier(
            latent_dim, classifier_d_model, classifier_heads, classifier_dropout, num_classes
        )

    @property
    def use_router(self):
        return self.router is not None

    def encode(self, text, audio, vision):
        return self.gan.encode(text, audio, vision)

    def fuse(self, latents, use_router=True):
        """Fuses modality latents; returns ``(fused, weights)`` where ``weights`` is ``None`` for mean fusion."""
        stacked = [latents[m] for m in MODALITIES]
        if use_router and self.router is not None:
            weights = self.router(stacked)
            fused = sum(z * weights[:, i : i + 1] for i, z in enumerate(stacked))
            return fused, weights
        return sum(stacked) / len(stacked), None
