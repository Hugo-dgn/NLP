import torch
from torch.utils.data import Dataset

import utils

ASPECTS, LABELS, LABEL2ID, ID2LABEL, NUM_CLASSES = utils.get_constant()

# ---------------------------------------------------------------------------
# Custom collator
# ---------------------------------------------------------------------------

class AspectCollator:
    """
    Pads input_ids and attention_mask to the longest sequence in the batch.
    Stacks labels (shape (3,) per sample) into a (N, 3) tensor.
    DataCollatorWithPadding cannot be used here because it tries to pad
    the multi-label 'labels' field as if it were a sequence.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, samples: list[dict]) -> dict:
        max_len = max(len(s["input_ids"]) for s in samples)
        input_ids, attention_masks, labels = [], [], []
        for s in samples:
            pad = max_len - len(s["input_ids"])
            input_ids.append(s["input_ids"] + [self.pad_token_id] * pad)
            attention_masks.append(s["attention_mask"] + [0] * pad)
            labels.append(s["labels"])
        return {
            "input_ids":      torch.tensor(input_ids,       dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels":         torch.stack(labels),   # (N, 3)
        }


# ---------------------------------------------------------------------------
# Precomputed-embedding dataset (phase 1 only)
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """
    Stores mean-pooled encoder embeddings (precomputed, frozen) + labels.
    Used in phase 1 so the encoder is never called during training.
    """

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings  # (N, emb_dim)
        self.labels = labels          # (N, 3)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {"embeddings": self.embeddings[idx], "labels": self.labels[idx]}


def embedding_collator(samples: list[dict]) -> dict:
    return {
        "embeddings": torch.stack([s["embeddings"] for s in samples]),
        "labels":     torch.stack([s["labels"]     for s in samples]),
    }
    
# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------

class ReviewDataset(Dataset):
    """Wraps a list of dicts from the TSV data into a torch Dataset."""

    def __init__(self, data: list[dict], tokenizer, max_length: int = 512):
        self.texts = [item["Review"].replace('"', '') for item in data]
        # Build label tensors: shape (N, 3) — one label index per aspect
        self.labels = torch.tensor(
            [
                [LABEL2ID[utils.normalize_label(item[aspect])] for aspect in ASPECTS]
                for item in data
            ],
            dtype=torch.long,
        )
        encoded = tokenizer(
            self.texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors=None,
        )
        self.input_ids = encoded["input_ids"]
        self.attention_mask = encoded["attention_mask"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }