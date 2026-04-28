from typing import Literal

import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_scheduler
)

from accelerate import Accelerator

from tqdm.auto import tqdm

import model
import utils
import data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Authorized encoder-only model chosen: multilingual RoBERTa (French reviews)
PLM_NAME = "FacebookAI/xlm-roberta-base"

ASPECTS, LABELS, LABEL2ID, ID2LABEL, NUM_CLASSES = utils.get_constant()

# ---------------------------------------------------------------------------
# OpinionExtractor
# ---------------------------------------------------------------------------

class OpinionExtractor:

    # SET TO "FT" because we fine-tune an encoder-only model
    method: Literal["NOFT", "FT"] = "FT"

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.plm_name = PLM_NAME

        # -----------------------------------------------------------------
        # Hyperparameters
        # -----------------------------------------------------------------
        mix_alpha = getattr(self.cfg, "mix_alpha", 0.3)
        mix_prob  = getattr(self.cfg, "mix_prob",  0.4)

        self.tokenizer = AutoTokenizer.from_pretrained(self.plm_name)
        self.model     = model.ThreeHeadTransformerClassifier(
            self.plm_name, mix_alpha=mix_alpha, mix_prob=mix_prob
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_class_weights(self, train_data: list[dict]) -> list[torch.Tensor]:
        """
        Compute inverse-frequency class weights for each aspect head from the
        training data.
        Formula: w_c = N / (K * n_c), normalised so mean weight == 1.
        Returns a list of 3 float tensors of shape (NUM_CLASSES,), one per aspect.
        """
        N = len(train_data)
        weights = []
        for aspect in ASPECTS:
            counts = torch.zeros(NUM_CLASSES)
            for item in train_data:
                counts[LABEL2ID[utils.normalize_label(item[aspect])]] += 1
            counts = torch.clamp(counts, min=1)
            w = N / (NUM_CLASSES * counts)
            w = w / w.mean()
            weights.append(w)
            print(f"  Class weights [{aspect}]: " +
                  " | ".join(f"{LABELS[i]}: {w[i]:.3f}" for i in range(NUM_CLASSES)))
        return weights

    def _make_training_args(
        self,
        output_dir: str,
        num_epochs: int,
        lr: float,
        batch_size: int,
        weight_decay: float,
        gradient_accumulation_steps: int,
        lr_scheduler_type: str = "constant",
        warmup_ratio: float = None,
        report_to="none"
    ) -> TrainingArguments:
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            logging_strategy="epoch",
            report_to=report_to,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_strategy="epoch",      # was evaluation_strategy
            save_strategy="best",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_acc",
            greater_is_better=True,
        )

    def _precompute_embeddings(self, dataset: data.ReviewDataset) -> torch.Tensor:
        accelerator = Accelerator()
        
        collator = data.AspectCollator(pad_token_id=self.tokenizer.pad_token_id)
        loader   = torch.utils.data.DataLoader(
            dataset, batch_size=64, collate_fn=collator
        )

        # prepare() moves the encoder and dataloader to the correct device
        # for this process (cuda:0, cuda:1, cpu, etc.)
        encoder, loader = accelerator.prepare(self.model.encoder, loader)
        encoder.eval()

        all_embeddings = []
        for batch in loader:
            # batch tensors are already on the correct device — no .to() needed
            with torch.no_grad():
                out    = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                pooled = self.model._pool(out.last_hidden_state, batch["attention_mask"])
                all_embeddings.append(pooled.cpu())

        # unwrap so the Trainer can re-prepare it cleanly in Phase 2
        self.model.encoder = accelerator.unwrap_model(encoder)

        return torch.cat(all_embeddings, dim=0)  # (N, emb_dim)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def train(self, train_data: list[dict], val_data: list[dict]) -> None:
        """
        Fine-tunes the 3-head classifier on train_data using HuggingFace Trainer.

        Phase 1 (num_epochs_head epochs): encoder frozen, heads only, Adam at
            head_learning_rate, trained on precomputed embeddings.
        Phase 2 (num_epochs epochs): encoder unfrozen, full fine-tuning with
            a cosine LR schedule and separate LRs for encoder vs. heads.

        Best model (by val macro_acc) is restored at the end.
        """

        # -----------------------------------------------------------------
        # Hyperparameters
        # -----------------------------------------------------------------
        num_epochs      = getattr(self.cfg, "num_epochs",          5)
        num_epochs_head = getattr(self.cfg, "num_epochs_head",     1)
        batch_size      = getattr(self.cfg, "train_batch_size",   16)
        weight_decay    = getattr(self.cfg, "weight_decay",      0.01)
        head_lr         = getattr(self.cfg, "head_learning_rate", 1e-3)
        lr              = getattr(self.cfg, "learning_rate",      2e-5)
        tau             = getattr(self.cfg, "tau",               0.95)
        gradient_accumulation_steps = getattr(self.cfg, "grad_acc", 16)

        # -----------------------------------------------------------------
        # Datasets & collator
        # -----------------------------------------------------------------
        train_dataset = data.ReviewDataset(train_data, self.tokenizer)
        val_dataset   = data.ReviewDataset(val_data,   self.tokenizer)
        collator      = data.AspectCollator(pad_token_id=self.tokenizer.pad_token_id)

        # -----------------------------------------------------------------
        # Class weights
        # -----------------------------------------------------------------
        print("Computing per-aspect class weights from training data...")
        class_weights = self._compute_class_weights(train_data)
        for head_idx, w in enumerate(class_weights):
            continue
            self.model.loss_fns[head_idx] = nn.CrossEntropyLoss(
                weight=w, label_smoothing=0.1
            )

        # -----------------------------------------------------------------
        # Phase 1: heads only on precomputed embeddings (encoder never called)
        # -----------------------------------------------------------------
        if num_epochs_head > 0:
            print(f"\n--- Phase 1: precomputing embeddings... ---")
            train_emb = self._precompute_embeddings(train_dataset)
            val_emb   = self._precompute_embeddings(val_dataset)

            emb_train = data.EmbeddingDataset(train_emb, train_dataset.labels)
            emb_val   = data.EmbeddingDataset(val_emb,   val_dataset.labels)

            # Thin wrapper that shares heads/loss_fns with the full model
            heads_model = model.HeadsOnlyModel(self.model.heads, self.model.loss_fns)
            
            print(f"--- Phase 1: heads only ({num_epochs_head} epoch(s), lr={head_lr:.2e}) ---")
            phase1_trainer = Trainer(
                model=heads_model,
                args=self._make_training_args(
                    output_dir="./phase1",
                    num_epochs=num_epochs_head,
                    lr=head_lr,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                ),
                train_dataset=emb_train,
                eval_dataset=emb_val,
                data_collator=data.embedding_collator,
                compute_metrics=utils.compute_metrics,
            )
            phase1_trainer.train()
            
        clean_indices      = model.get_clean_indices(emb_train, heads_model, tau)
        clean_train_dataset = torch.utils.data.Subset(
            train_dataset, clean_indices.tolist()
        )

        # -----------------------------------------------------------------
        # Phase 2: full fine-tuning (encoder unfrozen)
        # -----------------------------------------------------------------
        print(f"\n--- Phase 2: full fine-tuning ({num_epochs} epoch(s), lr={lr:.2e}) ---")

        phase2_trainer = Trainer(
            model=self.model,
            args=self._make_training_args(
                output_dir="./phase2",
                num_epochs=num_epochs,
                lr=lr,
                batch_size=batch_size,
                weight_decay=weight_decay,
                gradient_accumulation_steps=gradient_accumulation_steps,
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
                report_to="wandb"
            ),
            train_dataset=clean_train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            compute_metrics=utils.compute_metrics,
        )
        phase2_trainer.train()

        self.model.eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def predict(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        accelerator = Accelerator()

        self.model.eval()
        texts = [utils.clean_review(text) for text in texts]

        # Tokenize all texts upfront
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Wrap in a DataLoader so accelerate can handle device placement
        dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
        loader  = DataLoader(dataset, batch_size=batch_size)

        # prepare() moves model and batches to the correct device automatically
        model, loader = accelerator.prepare(self.model, loader)
        model.eval()

        all_preds = []
        for input_ids, attention_mask in loader:
            # tensors already on correct device — no .to() needed
            with torch.no_grad():
                output = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = output.logits.argmax(dim=-1).cpu().tolist()  # (batch, 3)
            all_preds.extend(preds)

        # unwrap so the model stays clean after predict()
        self.model = accelerator.unwrap_model(model)

        return [
            {aspect: ID2LABEL[all_preds[i][j]] for j, aspect in enumerate(ASPECTS)}
            for i in range(len(texts))
        ]