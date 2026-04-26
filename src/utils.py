import torch
from transformers import (
    TrainerCallback,
)

import re
import unicodedata

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


ASPECTS = ["Price", "Food", "Service"]
LABELS = ["Positive", "Negative", "Mixed", "No Opinion"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)  # 4

def get_constant():
    return ASPECTS, LABELS, LABEL2ID, ID2LABEL, NUM_CLASSES

"""def clean_review(text):
    return text.replace('"', '')"""

def normalize_label(raw: str) -> str:
    """
    Clean noisy labels from the TSV.
    E.g. 'Positive#NE' -> 'Positive', '  Mixed ' -> 'Mixed'.
    Falls back to 'No Opinion' for any unrecognised value.
    """
    cleaned = raw.strip().split("#")[0].strip()
    return cleaned if cleaned in LABEL2ID else "No Opinion"

def eval(preds: list[dict], eval_data: list[dict]) -> dict[str,float]:
    n = len(eval_data)
    correct_counts = {aspect: 0.0 for aspect in ASPECTS}
    for pred, ref in zip(preds, eval_data):
        if pred is None:
            continue
        for aspect in ASPECTS:
            if aspect in pred and pred[aspect] == ref[aspect]:
                correct_counts[aspect] += 1
    for aspect in correct_counts:
        correct_counts[aspect] = 100*correct_counts[aspect]/n
    macro_acc = sum(acc for acc in correct_counts.values())/len(ASPECTS)
    correct_counts['macro_acc'] = macro_acc
    return correct_counts

def evaluate(extractor, eval_data, eval_batch_size):
    preds = []
    for start_idx in range(0, len(eval_data), eval_batch_size):
        batch = [element['Review'] for element in eval_data[start_idx:start_idx+eval_batch_size]]
        batch_preds = extractor.predict(batch)
        preds.extend(batch_preds)
    accuracies = eval(preds, eval_data)
    return accuracies

class EvalCallback(TrainerCallback):
    """Calls utils.evaluate on train and val sets after every epoch and tracks the best model."""

    def __init__(self, extractor, train_data, val_data):
        self.extractor  = extractor
        self.train_data = train_data
        self.val_data   = val_data
        self.best_acc   = -1.0
        self.best_state = None
        # Set to precomputed embedding tensors during phase 1; None during phase 2
        self.train_emb: torch.Tensor | None = None
        self.val_emb:   torch.Tensor | None = None

    def _eval_from_embeddings(self, embeddings: torch.Tensor, ref_data: list[dict],
                               batch_size: int = 16) -> float:
        """Run heads-only prediction on precomputed embeddings, return macro_acc."""
        device = self.extractor.device
        self.extractor.model.eval()
        preds = []
        for start in range(0, len(embeddings), batch_size):
            batch_emb = embeddings[start:start + batch_size].to(device)
            with torch.no_grad():
                logits_list = [head(batch_emb) for head in self.extractor.model.heads]
            batch_preds = torch.stack(logits_list, dim=1).argmax(dim=-1).cpu().tolist()
            for row in batch_preds:
                preds.append({aspect: ID2LABEL[row[j]] for j, aspect in enumerate(ASPECTS)})
        return eval(preds, ref_data)['macro_acc']

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        model.eval()
        if self.train_emb is not None:
            # Phase 1: use precomputed embeddings — no encoder call
            acc_val   = self._eval_from_embeddings(self.val_emb,   self.val_data)
            acc_train = self._eval_from_embeddings(self.train_emb, self.train_data)
        else:
            # Phase 2: normal evaluation through the full model
            acc_val   = evaluate(self.extractor, self.val_data,   16)['macro_acc']
            acc_train = evaluate(self.extractor, self.train_data, 16)['macro_acc']
        model.train()

        if acc_val > self.best_acc:
            self.best_acc = acc_val
            # Always snapshot the FULL model (ThreeHeadTransformerClassifier),
            # not the HeadsOnlyModel wrapper that Trainer may be holding in phase 1.
            full_model = self.extractor.model
            self.best_state = {k: v.cpu().clone() for k, v in full_model.state_dict().items()}
            print(f"  [eval] New best val acc={acc_val:.4f} (train acc={acc_train:.4f})")
        else:
            print(f"  [eval] val acc={acc_val:.4f} (train acc={acc_train:.4f})")