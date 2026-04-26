"""
Restaurant Review Dataset Curation — Gemma 4 via HuggingFace Transformers
==========================================================================
No sudo. No paid API. Pure pip + a free HuggingFace account.

One-time setup:
    1. Create a free account at https://huggingface.co/join
    2. Accept model terms at https://huggingface.co/google/gemma-4-12b-it
    3. Create a token at https://huggingface.co/settings/tokens
    4. pip install --user "transformers>=5.5.0" torch accelerate tqdm pandas
    5. export HF_TOKEN=hf_xxxxxxxxxxxx

Then just run:
    python curate_dataset_gemma4.py

Outputs:
    curated_data.csv   — clean rows (labels auto-corrected where needed)
    excluded_data.csv  — rejected rows with a rejection reason

Hardware guide:
    Model ID                  │ RAM / VRAM  │ Notes
    ──────────────────────────┼─────────────┼──────────────────────────────
    google/gemma-4-E2B-it     │  ~4 GB      │ Laptop / CPU-friendly (2B MoE)
    google/gemma-4-E4B-it     │  ~6 GB      │ Good laptop choice    (4B MoE)
    google/gemma-4-12b-it     │  ~8 GB      │ Best quality / speed ← default
    google/gemma-4-27b-it     │ ~18 GB      │ Strongest, needs a GPU

    Add LOAD_IN_4BIT = True to halve memory at minimal quality cost.
    Requires: pip install --user bitsandbytes

Batch size VRAM guide (gemma-4-12b, bfloat16):
    BATCH_SIZE = 1  →  ~8  GB
    BATCH_SIZE = 4  →  ~12 GB
    BATCH_SIZE = 8  →  ~16 GB
    Lower BATCH_SIZE or enable LOAD_IN_4BIT if you get OOM errors.
"""

import json
import logging
import re
import time
import warnings
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_ID     = "google/gemma-4-31B-it"   # swap to any model in the table above
LOAD_IN_4BIT = False                      # set True + pip install bitsandbytes to halve RAM
BATCH_SIZE   = 8                          # tune to your VRAM — lower if OOM

MAX_NEW_TOKENS       = 512
CONFIDENCE_THRESHOLD = 0.8
MAX_RETRIES          = 2
ROW_DELAY_SECONDS    = 0.0               # inter-batch pause in seconds (0 = no pause)

ASPECTS = ["Price", "Food", "Service"]
DATA_DIR   = Path("../data/")
OUTPUT_DIR = Path("../data/")

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
Vous êtes un expert en curation de données pour l'analyse de sentiment par aspect (ABSA) appliquée aux avis de restaurants. Votre tâche consiste à valider des annotations humaines bruitées.

Pour chaque avis que vous recevez, vous verrez :

* Le texte de l'avis.
* L'étiquette annotée par un humain pour chaque aspect : Prix, Nourriture, Service.

Votre mission :

1. Lire attentivement l'avis.
2. Pour chaque aspect, choisir exactement une étiquette :
   Positif | Négatif | Mixte | Aucune opinion
   (« Aucune opinion » = l'avis ne dit rien sur cet aspect.)
3. Comparer vos étiquettes à celles de l'humain.
4. Attribuer un score de confiance (0.0–1.0) à l'annotation humaine globale
   (1.0 = toutes les étiquettes correspondent parfaitement aux vôtres).
5. Pour toute étiquette avec laquelle vous n'êtes pas d'accord, fournir une courte correction_reason.

Produire UNIQUEMENT un objet JSON valide — pas de balises markdown, pas de texte supplémentaire.
Schéma :
{
"model_labels": {
"Price":   "<Positif|Négatif|Mixte|Aucune opinion>",
"Food":    "<Positif|Négatif|Mixte|Aucune opinion>",
"Service": "<Positif|Négatif|Mixte|Aucune opinion>"
},
"confidence": <float 0.0–1.0>,
"disagreements": {
"<aspect>": "<brève raison pour laquelle l'étiquette humaine est incorrecte>"
},
"exclude": <true|false>,
"exclude_reason": "<chaîne vide si exclude=false, sinon expliquer>"
}

Mettre exclude=true si l'avis est incompréhensible/hors sujet ou si les annotations sont
tellement incohérentes que l'exemple nuirait à l'entraînement du modèle.
"""


def build_messages(review: str, human_labels: dict) -> list[dict]:
    labels_str = "\n".join(
        f"  {a}: {human_labels.get(a, 'No Opinion')}" for a in ASPECTS
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Review:\n\"\"\"\n{review}\n\"\"\"\n\n"
            f"Human annotations:\n{labels_str}\n\n"
            "Output only the JSON object."
        )},
    ]


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model():
    log.info(f"Loading {MODEL_ID}  (4-bit={LOAD_IN_4BIT})")
    log.info("First run downloads weights — may take a few minutes …")

    kwargs = dict(dtype=torch.bfloat16, device_map="cuda")

    if LOAD_IN_4BIT:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except ImportError:
            log.warning("bitsandbytes not found — using bfloat16 full precision.")

    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kwargs)
    # padding_side="left" is required for Gemma 4 batched/generative inference
    processor = AutoProcessor.from_pretrained(MODEL_ID, padding_side="left")
    model.eval()
    log.info("Model ready.")
    return model, processor


# ── Inference ──────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Optional[dict]:
    """Strip markdown fences and extract the first JSON object found."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def run_inference_batch(
    model,
    processor,
    reviews: list[str],
    human_labels_list: list[dict],
) -> list[Optional[dict]]:
    """
    Run batched inference for a list of reviews.

    Returns a list of parsed JSON dicts (or None on failure) in the same
    order as the input reviews.

    Key design notes
    ----------------
    * padding_side="left" (set at load time) is critical: left-padding ensures
      all sequences end at the same position so generation starts correctly for
      every item in the batch.
    * input_len is the padded sequence length; slicing out[i][input_len:] drops
      the shared prompt prefix and keeps only the newly generated tokens.
    """

    # Build one prompt string per review
    texts = []
    for review, human_labels in zip(reviews, human_labels_list):
        messages = build_messages(review, human_labels)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking=False,   # disables internal reasoning channel — cleaner JSON output
        )
        texts.append(text)

    # Tokenize the whole batch at once; processor pads to the longest sequence
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    # After left-padding every sequence has the same length
    input_len = inputs["input_ids"].shape[-1]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,   # greedy = deterministic
                )

            # Decode each item individually, slicing off the prompt prefix
            results = []
            for seq in out:
                generated = processor.decode(seq[input_len:], skip_special_tokens=True)
                parsed = _parse_json(generated)
                if parsed is None:
                    log.warning(f"JSON parse failed. Raw output: {generated[:200]}")
                results.append(parsed)
            return results

        except Exception as exc:
            log.warning(f"Batch attempt {attempt}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    # All retries exhausted — return None for every item in the batch
    return [None] * len(reviews)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "ftdataset_val.tsv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, sep=r' *\t *', encoding='utf-8', engine='python')
        log.info(f"Loaded {len(df):,} rows from {csv_path}")
    else:
        log.warning(f"{csv_path} not found — using synthetic demo data.")
        df = pd.DataFrame({
            "review_text": [
                "The pizza was amazing but the waiter was rude and prices are too high.",
                "Great ambiance and affordable menu!",
                "i rlly luved the food soooo good 10/10 would recommend",
                "???",
                "Service was slow. Food was okay. Prices seem fair.",
                "Terrible experience. Everything was overpriced and cold.",
            ],
            "Price":   ["Negative", "Positive",  "Positive",  "Positive", "Positive", "Negative"],
            "Food":    ["Positive", "No Opinion", "Positive",  "Positive", "Mixed",    "Negative"],
            "Service": ["Negative", "No Opinion", "No Opinion","Positive", "Negative", "Negative"],
        })

    for col in ASPECTS:
        if col in df.columns:
            df[col] = df[col].fillna("No Opinion")
        else:
            df[col] = "No Opinion"
    return df.reset_index(drop=True)


# ── Curation loop ──────────────────────────────────────────────────────────────

def _process_single_result(row: dict, result: Optional[dict]) -> tuple[str, dict]:
    """
    Apply one model result to one row dict.
    Returns ("keep" | "exclude", updated_row_dict).
    """
    new_row = row.copy()

    if result is None:
        new_row.update({
            "exclude_reason":      "Model inference failed after retries",
            "model_confidence":    0.0,
            "model_disagreements": json.dumps({}),
            "model_labels":        json.dumps({}),
        })
        return "exclude", new_row

    confidence     = float(result.get("confidence", 0.0))
    should_exclude = bool(result.get("exclude", False))
    exclude_reason = result.get("exclude_reason", "")
    model_labels   = result.get("model_labels", {})
    disagreements  = result.get("disagreements", {})

    if not should_exclude and confidence < CONFIDENCE_THRESHOLD:
        should_exclude = True
        exclude_reason = (
            f"Low confidence ({confidence:.2f} < {CONFIDENCE_THRESHOLD}). "
            f"Disagreements: {disagreements}"
        )

    new_row.update({
        "model_confidence":    confidence,
        "model_disagreements": json.dumps(disagreements),
        "model_labels":        json.dumps(model_labels),
    })

    if should_exclude:
        new_row["exclude_reason"] = exclude_reason
        return "exclude", new_row

    # Auto-correct labels where the model disagrees
    for asp in ASPECTS:
        if asp in disagreements and asp in model_labels:
            log.info(f"  ✎ {asp}: '{row.get(asp)}' → '{model_labels[asp]}'")
            new_row[asp] = model_labels[asp]
    return "keep", new_row


def curate(df: pd.DataFrame, model, processor):
    curated_rows: list[dict] = []
    excluded_rows: list[dict] = []
    total = len(df)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    log.info(f"Curating {total:,} rows in {n_batches} batches (batch_size={BATCH_SIZE}) …")

    for batch_start in tqdm(range(0, total, BATCH_SIZE), total=n_batches, desc="Batches"):
        batch_df = df.iloc[batch_start : batch_start + BATCH_SIZE]

        reviews     = [str(r.get("review_text", r.get("Review", ""))) for _, r in batch_df.iterrows()]
        labels_list = [{a: r.get(a, "No Opinion") for a in ASPECTS} for _, r in batch_df.iterrows()]

        log.info(
            f"\n[Batch {batch_start // BATCH_SIZE + 1}/{n_batches}] "
            f"rows {batch_start+1}–{batch_start+len(reviews)}"
        )

        results = run_inference_batch(model, processor, reviews, labels_list)

        for (_, row), result in zip(batch_df.iterrows(), results):
            decision, new_row = _process_single_result(row.to_dict(), result)
            conf = new_row.get("model_confidence", 0.0)
            if decision == "keep":
                curated_rows.append(new_row)
                log.info(f"  → KEPT     conf={conf:.2f}  {row.get('review_text', '')[:60]}")
            else:
                excluded_rows.append(new_row)
                log.info(
                    f"  → EXCLUDED conf={conf:.2f}  {new_row.get('exclude_reason', '')[:80]}"
                )

        if ROW_DELAY_SECONDS > 0:
            time.sleep(ROW_DELAY_SECONDS)

    cols = df.columns.tolist()
    curated  = pd.DataFrame(curated_rows)  if curated_rows  else pd.DataFrame(columns=cols)
    excluded = pd.DataFrame(excluded_rows) if excluded_rows else pd.DataFrame(columns=cols)
    return curated, excluded


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(original: pd.DataFrame, curated: pd.DataFrame, excluded: pd.DataFrame):
    total, n_kept, n_excl = len(original), len(curated), len(excluded)
    print("\n" + "═" * 62)
    print("  CURATION SUMMARY")
    print("═" * 62)
    print(f"  Model           : {MODEL_ID}")
    print(f"  4-bit quant     : {LOAD_IN_4BIT}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"  Total processed : {total:>6,}")
    print(f"  Curated (kept)  : {n_kept:>6,}  ({n_kept/total*100:.1f}%)")
    print(f"  Excluded        : {n_excl:>6,}  ({n_excl/total*100:.1f}%)")
    if n_excl and "exclude_reason" in excluded.columns:
        print("\n  Exclusion reasons (sample):")
        for r in excluded["exclude_reason"].dropna().unique()[:5]:
            print(f"    • {str(r)[:80]}")
    if n_kept and "model_confidence" in curated.columns:
        print(f"\n  Avg confidence  : {curated['model_confidence'].mean():.3f}")
    if n_kept:
        print("\n  Label distribution after curation:")
        for asp in ASPECTS:
            if asp in curated.columns:
                dist = curated[asp].value_counts().to_dict()
                print(f"    {asp:<10}" + "  ".join(f"{k}: {v}" for k, v in dist.items()))
    print("═" * 62 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    model, processor = load_model()
    curated, excluded = curate(df, model, processor)

    curated_path  = OUTPUT_DIR / "curated_data.csv"
    excluded_path = OUTPUT_DIR / "excluded_data.csv"
    curated.to_csv(curated_path,  index=False)
    excluded.to_csv(excluded_path, index=False)

    log.info(f"Curated  → {curated_path}  ({len(curated):,} rows)")
    log.info(f"Excluded → {excluded_path} ({len(excluded):,} rows)")
    print_summary(df, curated, excluded)


if __name__ == "__main__":
    main()