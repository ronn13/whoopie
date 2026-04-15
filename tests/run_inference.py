# =============================================================================
# SafeAeroBERT Inference on asn_scraped_accidents
# =============================================================================
# Pulls narratives from your PostgreSQL database, runs the trained
# SafeAeroBERT NER model, and outputs structured event tuples.
#
# Usage:
#   Step 1: Fill in DB credentials in tests/db_connect.py
#   Step 2: python tests/run_inference.py
#   Step 3: Results saved to tests/output/extracted_events_asn.json
#
# What it extracts per report:
#   ACTOR   — who was involved (crew roles, ATC)
#   SYSTEM  — aircraft systems mentioned (TCAS, GPS, autopilot...)
#   TRIGGER — what initiated the event chain
#   PHASE   — from the existing `phase` column (no NER needed)
#   OUTCOME — from the existing `category` column (ADREP class)
# =============================================================================

import os
import sys
import json
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from db_connect import fetch_narratives

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "NASA-AIML/MIKA_SafeAeroBERT"
WEIGHTS_PATH = os.path.join(ROOT, "outputs", "models", "best_aerobert_event_extractor.pt")
MAX_LEN      = 256
SAMPLE_SIZE  = 20       # rows to pull from DB
OUTPUT_DIR   = os.path.join(ROOT, "tests", "output")
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "extracted_events_asn.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Label schema (must match Layer 1 training) ────────────────────────────────
ENTITY_TYPES = ["ACTOR", "SYSTEM", "TRIGGER"]
LABELS       = ["O"] + [f"{prefix}-{t}" for t in ENTITY_TYPES
                        for prefix in ["B", "I"]]
LABEL2ID     = {l: i for i, l in enumerate(LABELS)}
ID2LABEL     = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS   = len(LABELS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Load model
# =============================================================================

def load_model():
    print(f"🧠 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"🧠 Loading model weights from: {WEIGHTS_PATH}")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location=DEVICE)
        )
        print("✅ Trained weights loaded")
    else:
        print(f"⚠️  Weights not found at {WEIGHTS_PATH} — using pretrained only")

    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# =============================================================================
# Inference: narrative → entity spans
# =============================================================================

def extract_entities(text: str, model, tokenizer) -> dict:
    """Run NER on a single narrative string. Returns {ACTOR: [...], ...}"""
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    offset_mapping = encoding["offset_mapping"][0].tolist()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds   = torch.argmax(outputs.logits, dim=-1)[0].tolist()

    # Decode BIO tags → entity spans
    entities     = defaultdict(list)
    current_type = None
    current_start = None
    prev_end      = None

    for pred_id, (tok_start, tok_end) in zip(preds, offset_mapping):
        if tok_start == 0 and tok_end == 0:          # special token
            if current_type and current_start is not None:
                span = text[current_start:prev_end].strip()
                if span:
                    entities[current_type].append(span)
            current_type = None
            continue

        label = ID2LABEL.get(pred_id, "O")

        if label.startswith("B-"):
            if current_type and current_start is not None:
                span = text[current_start:prev_end].strip()
                if span:
                    entities[current_type].append(span)
            current_type  = label[2:]
            current_start = tok_start
            prev_end      = tok_end

        elif label.startswith("I-") and current_type == label[2:]:
            prev_end = tok_end

        else:
            if current_type and current_start is not None:
                span = text[current_start:prev_end].strip()
                if span:
                    entities[current_type].append(span)
            current_type  = None
            current_start = None

    # Deduplicate
    return {k: list(dict.fromkeys(v)) for k, v in entities.items()}


# =============================================================================
# Main pipeline
# =============================================================================

def run():
    print("=" * 60)
    print("  SafeAeroBERT Inference — asn_scraped_accidents")
    print("=" * 60)

    # ── Load model ─────────────────────────────────────────────────────────
    tokenizer, model = load_model()
    print(f"\n🖥️  Device: {DEVICE}\n")

    # ── Pull data from DB ───────────────────────────────────────────────────
    print(f"📂 Fetching all records from database... (this may take a minute)")
    df = fetch_narratives(None)
    print(f"✅ {len(df)} rows retrieved for inference\n")

    # ── Run inference ───────────────────────────────────────────────────────
    results = []
    print("─" * 60)
    
    from tqdm import tqdm
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Entities"):
        uid       = str(row.get("uid", i))
        narrative = str(row.get("narrative", ""))
        phase     = str(row.get("phase", ""))
        category  = str(row.get("category", ""))    # ADREP category if present

        if not narrative.strip():
            continue

        entities = extract_entities(narrative, model, tokenizer)

        result = {
            "uid":               uid,
            "narrative_preview": narrative[:120] + ("..." if len(narrative) > 120 else ""),
            "ACTOR":   entities.get("ACTOR",   []) or ["Unknown"],
            "SYSTEM":  entities.get("SYSTEM",  []) or ["Not identified"],
            "TRIGGER": entities.get("TRIGGER", []) or ["Not explicitly stated"],
            "PHASE":   phase or "Unknown",
            "OUTCOME": category or "Unknown",   # direct ADREP label from DB
        }
        results.append(result)

        # Print preview
        print(f"[{uid}]")
        print(f"  Narrative : {narrative[:100]}...")
        print(f"  ACTOR     : {result['ACTOR']}")
        print(f"  SYSTEM    : {result['SYSTEM']}")
        print(f"  TRIGGER   : {result['TRIGGER']}")
        print(f"  PHASE     : {result['PHASE']}")
        print(f"  OUTCOME   : {result['OUTCOME']}")
        print()

    # ── Save output ─────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("─" * 60)
    print(f"✅ Done — {len(results)} records processed")
    print(f"💾 Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
