import json
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def load_json_records(path):
    """Loads a ``.json`` list or a ``.jsonl`` file (one JSON object per line)."""
    with open(path, "r", encoding="utf-8") as f:
        if str(path).endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


class VectorFeatureDataset(Dataset):
    """Utterance-level features stored as JSON records.

    Each record needs ``text_feat``, ``audio_feat``, ``vision_feat`` (lists of floats) and
    ``label_id`` (int).
    """

    def __init__(self, records):
        self.records = records

    @classmethod
    def from_file(cls, path):
        return cls(load_json_records(path))

    @property
    def labels(self):
        return [r["label_id"] for r in self.records]

    @property
    def input_dims(self):
        first = self.records[0]
        return {
            "text": len(first["text_feat"]),
            "audio": len(first["audio_feat"]),
            "vision": len(first["vision_feat"]),
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return (
            torch.tensor(r["text_feat"], dtype=torch.float32),
            torch.tensor(r["audio_feat"], dtype=torch.float32),
            torch.tensor(r["vision_feat"], dtype=torch.float32),
            torch.tensor(r["label_id"], dtype=torch.long),
        )


class M3EDSequenceDataset(Dataset):
    """Sequence features for multi-label emotion recognition (CARAT-style M3ED ``.pt`` file).

    The file holds ``{'train' | 'valid' | 'test': {'src-text', 'src-audio', 'src-visual', 'tgt'}}``.
    ``src-*`` are ``[N, seq_len, dim]`` arrays. ``tgt[i]`` is a token list
    ``[<bos>, e_1, ..., e_k, <eos>]`` whose emotion ids start at ``label_offset``.
    """

    def __init__(self, split_data, num_classes=7, label_offset=4):
        self.text = split_data["src-text"]
        self.audio = split_data["src-audio"]
        self.visual = split_data["src-visual"]
        self.targets = split_data["tgt"]
        self.num_classes = num_classes
        self.label_offset = label_offset

    @classmethod
    def load_splits(cls, path, **kwargs):
        data = torch.load(path, weights_only=False)
        return {split: cls(data[split], **kwargs) for split in ("train", "valid", "test")}

    @property
    def input_dims(self):
        return {
            "text": self.text[0].shape[-1],
            "audio": self.audio[0].shape[-1],
            "vision": self.visual[0].shape[-1],
        }

    def __len__(self):
        return len(self.targets)

    def _multi_hot(self, target):
        label = np.zeros(self.num_classes, dtype=np.float32)
        for emotion_token in target[1:-1]:
            label[emotion_token - self.label_offset] = 1.0
        return label

    def __getitem__(self, idx):
        audio = np.where(np.isneginf(self.audio[idx]), 0.0, self.audio[idx])
        return (
            torch.as_tensor(self.text[idx], dtype=torch.float32),
            torch.as_tensor(audio, dtype=torch.float32),
            torch.as_tensor(self.visual[idx], dtype=torch.float32),
            torch.from_numpy(self._multi_hot(self.targets[idx])),
        )


def make_loaders(train_set, val_set, test_set, batch_size, num_workers=4):
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def inverse_frequency_weights(labels, num_classes):
    """``weight_c = N / (num_present_classes * count_c)``; absent classes get weight 0."""
    counts = Counter(labels)
    total = len(labels)
    return torch.tensor(
        [total / (len(counts) * counts[c]) if counts[c] else 0.0 for c in range(num_classes)],
        dtype=torch.float32,
    )


def print_class_distribution(labels, split_name):
    counts = Counter(labels)
    total = len(labels)
    print(f"\n{split_name} class distribution:")
    print(f"{'Class':<10}{'Count':<10}{'Share':<10}")
    for class_id in sorted(counts):
        print(f"{class_id:<10}{counts[class_id]:<10}{counts[class_id] / total:.2%}")
    print(f"{'Total':<10}{total:<10}")
