from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
)
from transformers.modeling_outputs import ModelOutput

import utils

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
            )

        return ThreeHeadOutput(loss=loss, logits=logits)





# ---------------------------------------------------------------------------
# 3-head classifier model
# ---------------------------------------------------------------------------

class ThreeHeadTransformerClassifier(nn.Module):
    """
    Encoder-only transformer with 3 independent classification heads,
    one per aspect (Price, Food, Service), each predicting 4 classes.
    """

    def __init__(self, plm_name: str, num_classes: int = NUM_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(plm_name)
        self.encoder = AutoModel.from_pretrained(
            plm_name,
            output_attentions=False,
            device_map=None,
        )
        emb_dim = self.config.hidden_size

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
        self.loss_fns = nn.ModuleList([nn.CrossEntropyLoss() for _ in ASPECTS])

        # Freeze the encoder at init: only heads are trained during phase 1
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze the encoder for full fine-tuning (called before phase 2)."""
        for param in self.encoder.parameters():
            param.requires_grad = True

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

        logits_list = [head(pooled) for head in self.heads]  # 3 x (N, 4)
        logits = torch.stack(logits_list, dim=1)              # (N, 3, 4)

        loss = None
        if labels is not None:
            # labels shape: (N, 3)
            loss = sum(
                self.loss_fns[i](logits_list[i], labels[:, i]) for i in range(len(ASPECTS))
            )

        return ThreeHeadOutput(loss=loss, logits=logits)