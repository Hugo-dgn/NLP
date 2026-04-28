import torch
from transformers import (
    TrainerCallback,
)

import re
import unicodedata


ASPECTS = ["Price", "Food", "Service"]
LABELS = ["Positive", "Negative", "Mixed", "No Opinion"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)  # 4

def get_constant():
    return ASPECTS, LABELS, LABEL2ID, ID2LABEL, NUM_CLASSES

def clean_review(text: str) -> str:
    # 1) Normalize unicode (important!)
    text = unicodedata.normalize("NFKC", text)

    # 2) Remove emojis (broad coverage)
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"  # most emojis
        "\U00002700-\U000027BF"
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)

    # 3) Normalize quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # 4) Remove backslashes and double quotes ONLY
    text = re.sub(r'[\\"]', "", text)

    # 5) Remove control characters (but keep newlines)
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)

    # 6) Remove weird symbols but KEEP:
    # letters (all languages), numbers, punctuation, apostrophes
    text = re.sub(r"[^\w\s.,!?;:'()\-\nÀ-ÿ]", " ", text)

    # 7) Normalize spacing
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()

def normalize_label(raw: str) -> str:
    """
    Clean noisy labels from the TSV.
    E.g. 'Positive#NE' -> 'Positive'
    Falls back to 'No Opinion' for any unrecognised value.
    """
    cleaned = raw.strip().split("#")[0].strip()
    return cleaned if cleaned in LABEL2ID else "No Opinion"

def compute_metrics(eval_pred):
    logits, labels = eval_pred  # logits: (N, 3, 4), labels: (N, 3)
    preds = logits.argmax(axis=-1)  # (N, 3)
    
    # macro accuracy: average per-aspect accuracy
    correct = (preds == labels).mean(axis=0)  # accuracy per aspect
    macro_acc = correct.mean()
    
    return {
        "macro_acc":      float(macro_acc),
        "acc_price":      float(correct[0]),
        "acc_food":       float(correct[1]),
        "acc_service":    float(correct[2]),
    }