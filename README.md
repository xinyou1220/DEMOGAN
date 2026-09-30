# DEMOGAN

**Dynamic modality routing + prototype learning on a GAN-regularized latent space for tri-modal emotion recognition.**

DEMOGAN encodes text, audio and vision features into a shared latent space, regularizes that space with a latent discriminator and contrastive objectives, fuses the modalities with a per-sample router, and classifies by similarity to learned class prototypes.

The repository has two pipelines that share one model package:

| Script | Input | Task | Prototype loss |
|---|---|---|---|
| `train_vector.py` | one feature vector per modality (JSON / JSONL) | single-label, 7 classes | cross-entropy |
| `train_sequence.py` | one feature sequence per modality (M3ED `.pt`, CARAT format) | multi-label, 7 classes | binary cross-entropy |

## Method

```
text ─► Encoder_T ─┐                         ┌─► Router ─► W ∈ R³ ─┐
audio ─► Encoder_A ─┼─► z_T, z_A, z_V (pooled) ┤                      ├─► z = Σ W_m z_m ─► Prototypes ─► prediction
vision ─► Encoder_V ─┘         │               └──────────────────────┘
                               ├─► Latent discriminator (prior N(0, I) vs. encoder output)
                               ├─► Supervised contrastive loss (within each modality)
                               ├─► Cross-modal InfoNCE (same sample, different modality)
                               └─► Self-attention classifier (auxiliary loss only)
```

1. **Modality encoders.** Each modality has a residual MLP encoder that maps features to a `latent_dim` space (default 512). For sequences, the encoder runs on every time step and the result is pooled (`mean`, `max`, `first` or `last`).
2. **Latent GAN.** A shared discriminator learns to separate samples of `N(0, I)` from encoder outputs. The encoders are trained to fool it, which pulls all three modalities toward the same prior (adversarial-autoencoder style).
3. **Contrastive alignment.**
   - *Intra-modal supervised contrastive loss:* samples of the same class are pulled together within each modality. In the multi-label setting, samples that share any label count as positives.
   - *Cross-modal InfoNCE:* the three modalities of one sample are aligned, averaged over the T–A, T–V and A–V pairs.
4. **Router.** An MLP reads `[z_T; z_A; z_V]` and outputs `W = softmax(G(·) / τ)`. The fused vector is `z = W_T z_T + W_A z_A + W_V z_V`. A router-entropy term keeps the router from collapsing onto one modality. With `--no-router`, the modalities are averaged instead.
5. **Prototype learning.** Each class has a learnable prototype `p_c`. Logits are `cos(z, p_c) / τ_p`. **Predictions at inference time come from the prototypes**: arg-max in the single-label case, `sigmoid(logit) > threshold` in the multi-label case.
6. **Auxiliary classifier.** A small self-attention classifier over the three modality tokens adds a training signal only. Its test metrics are reported for reference.

### Training schedule

| Stage | Epochs | What is trained |
|---|---|---|
| Warm-up | `1 … warmup_epochs` | Encoders + discriminator with adversarial and contrastive losses |
| Joint | until `freeze_gan_epoch` | Everything end to end |
| Frozen | `freeze_gan_epoch …` | GAN frozen; prototypes, router and (by default) the classifier keep training |

The checkpoint with the best validation score is kept. The score is prototype accuracy for the vector pipeline and exact-match accuracy for the sequence pipeline.

**Modality masking** (`train_vector.py --modality-masking`) randomly zeroes whole modality latents during training to improve robustness to missing modalities. By default at least one modality is always kept.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ and PyTorch 2.x are required. A CUDA GPU is recommended.

## Data

### Vector features (`train_vector.py`)

One file per split, either a JSON list or JSONL (one object per line):

```json
{"text_feat": [0.12, ...], "audio_feat": [0.03, ...], "vision_feat": [0.40, ...], "label_id": 3}
```

Feature dimensions are read from the first record. `label_id` must be in `[0, num_classes)`.

### Sequence features (`train_sequence.py`)

This pipeline reads a single `.pt` file in the format used by the CARAT M3ED release (`m3ed_data_*.pt`):

```python
{
  "train" | "valid" | "test": {
    "src-text":   array [N, seq_len, d_text],
    "src-audio":  array [N, seq_len, d_audio],   # -inf values are replaced with 0
    "src-visual": array [N, seq_len, d_vision],
    "tgt":        list of token lists [<bos>, e_1, ..., e_k, <eos>],
  }
}
```

Emotion token `e` maps to class `e - label_offset` (default offset 4), which yields a multi-hot target.

Datasets are not redistributed here. Obtain them from their original sources and follow their licenses.

## Usage

```bash
# single-label, vector features
python train_vector.py --train data/train.jsonl --val data/val.jsonl --test data/test.jsonl

# same, with modality masking
python train_vector.py --train ... --val ... --test ... --modality-masking --mask-ratio 0.3

# multi-label, sequence features
python train_sequence.py --data data/m3ed_data_60.pt --pooling max
```

The defaults reproduce the hyper-parameters used in our experiments. Run a script with `--help` to see every option, for example `--no-router`, `--freeze-gan-epoch 0` (never freeze), `--lambda-*` and the learning rates.

### Outputs

Each run writes to `runs/<vector|sequence>_<timestamp>/`:

| File | Content |
|---|---|
| `config.json` | All command-line arguments |
| `best_model.pth` | `{"epoch", "model", "val_metrics"}` for the best validation epoch |
| `training_curves.png` | Training losses and validation metrics per epoch |
| `test_results.txt` / `test_metrics.json` | Test metrics for the prototype head and the classifier (vector) |
| `confusion_matrix_*.png`, `learned_prototypes.npy` | Confusion matrices and normalized prototypes (vector) |
| `test_report.txt`, `test_results.npz` | Exact match, Jaccard accuracy, Hamming loss, micro/macro/samples P/R/F1, per-class scores (sequence) |
| `test_summary.png`, `confusion_matrices.png` | Summary plots and per-class binary confusion matrices (sequence) |

## Repository layout

```
demogan/
  models.py      encoders, pooling, latent discriminator, router, prototypes, classifier, DEMOGAN wrapper
  losses.py      supervised contrastive, cross-modal InfoNCE, router entropy, focal loss
  data.py        JSON/JSONL vector dataset and M3ED sequence dataset
  trainer.py     three-stage joint training loop and modality masking
  evaluation.py  prediction collection and single-/multi-label metrics
  plotting.py    training curves, confusion matrices and summaries
  utils.py       seeding and run-directory helpers
train_vector.py    single-label entry point
train_sequence.py  multi-label entry point
```

## Status

This project is archived and no longer maintained. Issues and pull requests may not receive a response.

## License

The code is released under the [MIT License](LICENSE). Datasets and pretrained models used by the scripts are subject to their own licenses.
