from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
)
from transformers.modeling_outputs import ModelOutput

from accelerate import Accelerator

import utils
from data import EmbeddingDataset

ASPECTS, LABELS, LABEL2ID, ID2LABEL, NUM_CLASSES = utils.get_constant()

# ---------------------------------------------------------------------------
# Heads-only model (phase 1 only)
# ---------------------------------------------------------------------------

@dataclass
class ThreeHeadOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None  # shape (N, 3, 4)


class HeadsOnlyModel(nn.Module):
    """
    Thin wrapper around the classification heads for phase 1.
    Receives precomputed pooled embeddings directly — no encoder involved.
    Shares the same heads and loss_fns objects as ThreeHeadTransformerClassifier
    so weight updates carry over without any copying.
    """

    # Trainer requires a `config` attribute to detect the model type
    config = None

    def __init__(self, heads: nn.ModuleList, loss_fns: nn.ModuleList):
        super().__init__()
        self.heads    = heads     # shared reference — same tensors as the full model
        self.loss_fns = loss_fns  # shared reference

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> ThreeHeadOutput:
        logits_list = [head(embeddings) for head in self.heads]
        logits = torch.stack(logits_list, dim=1)  # (N, 3, 4)

        loss = None
        if labels is not None:
            loss = sum(
                self.loss_fns[i](logits_list[i], labels[:, i])
                for i in range(len(ASPECTS))
            ) / len(ASPECTS)

        return ThreeHeadOutput(loss=loss, logits=logits)

def get_clean_indices(
    emb_train: EmbeddingDataset,
    heads_model: HeadsOnlyModel,
    tau: float = 0.9,
) -> torch.Tensor:
    accelerator = Accelerator()
    heads_model.eval()

    loader = DataLoader(emb_train, batch_size=1, shuffle=False)

    # accelerate handles device placement for model and batches
    heads_model, loader = accelerator.prepare(heads_model, loader)

    all_losses = []
    with torch.no_grad():
        for batch in loader:
            # tensors already on correct device — no .to() needed
            output = heads_model(embeddings=batch["embeddings"], labels=batch["labels"])
            all_losses.append(output.loss.cpu())

    # unwrap so the caller gets a clean model back
    heads_model = accelerator.unwrap_model(heads_model)

    all_losses    = torch.stack(all_losses)  # (N,)
    threshold     = torch.quantile(all_losses, tau)
    clean_indices = torch.where(all_losses < threshold)[0]

    print(
        f"Separation: {len(clean_indices)} clean / "
        f"{len(all_losses) - len(clean_indices)} noisy "
        f"(tau={tau}, total={len(all_losses)})"
    )

    return clean_indices

# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------



class MixupEmbedding(nn.Module):
    def __init__(self, alpha: float = 0.3, mix_prob: float = 0.4):
        super().__init__()
        self.alpha = alpha
        self.mix_prob = mix_prob  # percentage of batch to mix
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int):

        if not self.training or labels is None:
            return embeddings, labels

        B = embeddings.size(0)

        # One-hot labels
        soft_labels = F.one_hot(labels, num_classes).float()

        # Decide how many samples to mix
        num_mix = int(self.mix_prob * B)

        # Random subset of indices to mix
        mix_indices = torch.randperm(B, device=embeddings.device)[:num_mix]

        # Shuffle for pairing
        shuffle_indices = mix_indices[torch.randperm(num_mix)]

        lam = self.beta.sample().to(embeddings.device)

        # Clone to avoid in-place issues
        mixed_embeddings = embeddings.clone()
        mixed_labels = soft_labels.clone()

        # Apply mixup only on subset
        mixed_embeddings[mix_indices] = (
            lam * embeddings[mix_indices] +
            (1 - lam) * embeddings[shuffle_indices]
        )

        mixed_labels[mix_indices] = (
            lam * soft_labels[mix_indices] +
            (1 - lam) * soft_labels[shuffle_indices]
        )

        return mixed_embeddings, mixed_labels


# ---------------------------------------------------------------------------
# 3-head classifier model
# ---------------------------------------------------------------------------

class ThreeHeadTransformerClassifier(nn.Module):
    """
    Encoder-only transformer with 3 independent classification heads,
    one per aspect (Price, Food, Service), each predicting 4 classes.
    """

    def __init__(self, plm_name: str, num_classes: int = NUM_CLASSES, mix_alpha: float = 0.3, mix_prob: float = 0.4, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(plm_name)
        self.encoder = AutoModel.from_pretrained(
            plm_name,
            output_attentions=False,
            device_map=None,
        )
        emb_dim = self.config.hidden_size
        
        self.mixup = MixupEmbedding(mix_alpha, mix_prob)

        # One classification head per aspect
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(emb_dim, emb_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(emb_dim, num_classes),
            )
            for _ in ASPECTS
        ])

        # One cross-entropy loss per head (weights are set later from train data)
        self.loss_fns = nn.ModuleList([nn.CrossEntropyLoss(label_smoothing=0) for _ in ASPECTS])

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling that excludes padding tokens."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask  # (N, emb_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,  # Trainer may forward extra keys (e.g. token_type_ids)
    ) -> ThreeHeadOutput:
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(encoder_out.last_hidden_state, attention_mask)  # (N, emb_dim)
        
        pooled, labels = self.mixup(pooled, labels, num_classes=4)

        logits_list = [head(pooled) for head in self.heads]  # 3 x (N, 4)
        logits = torch.stack(logits_list, dim=1)              # (N, 3, 4)

        loss = None
        if labels is not None:
            # labels shape: (N, 3)
            loss = sum(
                self.loss_fns[i](logits_list[i], labels[:, i]) for i in range(len(ASPECTS))
            ) / len(ASPECTS)

        return ThreeHeadOutput(loss=loss, logits=logits)