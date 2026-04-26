from typing import Literal

import torch
from torch.optim import AdamW
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_scheduler
)

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
        mix_prob = getattr(self.cfg, "mix_prob", 0.4)
        

        self.tokenizer = AutoTokenizer.from_pretrained(self.plm_name)
        self.model = model.ThreeHeadTransformerClassifier(self.plm_name, mix_alpha=mix_alpha, mix_prob=mix_prob)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _compute_class_weights(self, train_data: list[dict]) -> list[torch.Tensor]:
        """
        Compute inverse-frequency class weights for each aspect head from the training data.
        Formula: w_c = N / (K * n_c), normalised so the mean weight == 1 (keeps loss scale stable).
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

    def _make_training_args(self, output_dir: str, num_epochs: int, lr: float,
                            batch_size: int, weight_decay: float,
                            gradient_accumulation_steps : int) -> TrainingArguments:
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            lr_scheduler_type="constant",
            logging_strategy="epoch",
            save_strategy="no",             # best-model saving handled by callback
            report_to="wandb",
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    @torch.no_grad()
    def _precompute_embeddings(self, dataset: data.ReviewDataset) -> torch.Tensor:
        """
        Run the frozen encoder over the full dataset in mini-batches and
        return mean-pooled embeddings of shape (N, emb_dim) on CPU.
        """
        self.model.encoder.eval()
        self.model.encoder.to(self.device)
        collator = data.AspectCollator(pad_token_id=self.tokenizer.pad_token_id)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, collate_fn=collator)
        all_embeddings = []
        for batch in loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            out = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self.model._pool(out.last_hidden_state, attention_mask)
            all_embeddings.append(pooled.cpu())
        return torch.cat(all_embeddings, dim=0)  # (N, emb_dim)

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def train(self, train_data: list[dict], val_data: list[dict]) -> None:
        """
        Fine-tunes the 3-head classifier on train_data using HuggingFace Trainer.
        Phase 1 (num_epochs_head epochs): encoder frozen, heads only, Adam at head_learning_rate.
        Phase 2 (num_epochs epochs): encoder unfrozen, AdamW at learning_rate.
        Best model (by val macro_acc) is restored at the end.
        """

        # -----------------------------------------------------------------
        # Hyperparameters
        # -----------------------------------------------------------------
        num_epochs      = getattr(self.cfg, "num_epochs",         5)
        num_epochs_head = getattr(self.cfg, "num_epochs_head",    1)
        batch_size      = getattr(self.cfg, "train_batch_size",  16)
        weight_decay    = getattr(self.cfg, "weight_decay",     0.01)
        head_lr         = getattr(self.cfg, "head_learning_rate", 1e-3)
        lr              = getattr(self.cfg, "learning_rate",      2e-5)
        gradient_accumulation_steps = getattr(self.cfg, "grad_acc", 16)

        # -----------------------------------------------------------------
        # Datasets & collator
        # -----------------------------------------------------------------
        train_dataset = data.ReviewDataset(train_data, self.tokenizer)
        val_dataset   = data.ReviewDataset(val_data,   self.tokenizer)
        collator      = data.AspectCollator(pad_token_id=self.tokenizer.pad_token_id)

        # Shared callback — tracks best model across both phases
        eval_callback = utils.EvalCallback(self, train_data, val_data)
        
        
        # -----------------------------------------------------------------
        # Class weights
        # -----------------------------------------------------------------
        print("Computing per-aspect class weights from training data...")
        class_weights = self._compute_class_weights(train_data)
        for head_idx, w in enumerate(class_weights):
            continue
            self.model.loss_fns[head_idx] = nn.CrossEntropyLoss(weight=w.to(self.device), label_smoothing=0.1)

        # -----------------------------------------------------------------
        # Phase 1: heads only on precomputed embeddings (encoder never called)
        # -----------------------------------------------------------------
        if num_epochs_head > 0:
            print(f"\n--- Phase 1: precomputing embeddings... ---")
            train_emb = self._precompute_embeddings(train_dataset)
            val_emb   = self._precompute_embeddings(val_dataset)

            emb_train = data.EmbeddingDataset(train_emb, train_dataset.labels)
            emb_val   = data.EmbeddingDataset(val_emb,   val_dataset.labels)

            # Give the callback the precomputed embeddings so it skips the encoder
            eval_callback.train_emb = train_emb
            eval_callback.val_emb   = val_emb

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
                    gradient_accumulation_steps=gradient_accumulation_steps
                ),
                train_dataset=emb_train,
                eval_dataset=emb_val,
                data_collator=data.embedding_collator,
                callbacks=[eval_callback],
            )
            phase1_trainer.train()

            # Clear embeddings: phase 2 uses the full encoder via utils.evaluate
            eval_callback.train_emb = None
            eval_callback.val_emb   = None

        # -----------------------------------------------------------------
        # Phase 2: full fine-tuning (encoder unfrozen)
        # -----------------------------------------------------------------
        print(f"\n--- Phase 2: full fine-tuning ({num_epochs} epoch(s), lr={lr:.2e}) ---")
        self.model.unfreeze()
        
        # Separate parameters
        encoder_params = list(self.model.encoder.parameters())
        head_params = list(self.model.heads.parameters())

        optimizer = AdamW(
            [
                {"params": encoder_params, "lr": lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )
        
        num_training_steps = (
            len(train_dataset) // (batch_size * gradient_accumulation_steps)
        ) * num_epochs
        
        num_warmup_steps = int(0.1 * num_training_steps)
        
        lr_scheduler = get_scheduler(
            name="cosine",  # same as lr_scheduler_type
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
        )

        phase2_trainer = Trainer(
            model=self.model,
            args=self._make_training_args(
                output_dir="./phase2",
                num_epochs=num_epochs,
                lr=lr,
                batch_size=batch_size,
                weight_decay=weight_decay,
                gradient_accumulation_steps=gradient_accumulation_steps,
            ),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            callbacks=[eval_callback],
            optimizers=(optimizer, lr_scheduler)
        )
        phase2_trainer.train()

        # -----------------------------------------------------------------
        # Restore best weights found across both phases
        # -----------------------------------------------------------------
        if eval_callback.best_state is not None:
            print(f"\nRestoring best model (val acc={eval_callback.best_acc:.4f})")
            self.model.load_state_dict(eval_callback.best_state)
        
        self.model.eval()

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def predict(self, texts: list[str]) -> list[dict]:
        """
        :param texts: list of reviews from which to extract the opinion values
        :return: a list of dicts, one per input review, with keys "Price", "Food", "Service"
                 and values from {"Positive", "Negative", "Mixed", "No Opinion"}
        """
        self.model.eval()
        device = self.device
        
        texts = [utils.clean_review(text) for text in texts]

        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids      = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # output.logits shape: (N, 3, 4) — argmax over class dimension
        preds = output.logits.argmax(dim=-1).cpu().tolist()  # (N, 3)

        return [
            {aspect: ID2LABEL[preds[i][j]] for j, aspect in enumerate(ASPECTS)}
            for i in range(len(texts))
        ]