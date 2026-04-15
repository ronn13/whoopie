"""
inference.py — Shared model loading & inference logic for the demo UI.

Loads both models once on startup and exposes functions:
  - extract_entities(text)  → Layer 1 NER (ACTOR/SYSTEM/TRIGGER)
  - classify_damage(text)   → Layer 3 severity classification
"""

import os, re, json
from collections import defaultdict
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
# APP_DIR is the directory containing this file (works both locally and in Docker)
APP_DIR    = os.path.dirname(os.path.abspath(__file__))
# Locally: APP_DIR = .../whoopie/app  → models live at ../outputs/models
# In Docker: APP_DIR = /app           → models live at /app/outputs/models
_local_models = os.path.join(APP_DIR, "..", "outputs", "models")
_docker_models = os.path.join(APP_DIR, "outputs", "models")
MODELS_DIR = _docker_models if os.path.isdir(_docker_models) else _local_models

L1_WEIGHTS  = os.path.join(MODELS_DIR, "best_aerobert_event_extractor.pt")
L1_CONFIG   = os.path.join(MODELS_DIR, "event_extractor_config.json")
L3_WEIGHTS  = os.path.join(MODELS_DIR, "safeaerobert_classifier.pt")

# ── Layer-1 config ─────────────────────────────────────────────────────────────
with open(L1_CONFIG) as f:
    _l1_cfg = json.load(f)

L1_MODEL_NAME = _l1_cfg["model_name"]          # bert-base-uncased
L1_MAX_LEN    = _l1_cfg["max_len"]             # 256
L1_LABEL2ID   = _l1_cfg["label2id"]
L1_ID2LABEL   = {int(k): v for k, v in _l1_cfg["id2label"].items()}
L1_NUM_LABELS = _l1_cfg["num_labels"]

# ── Layer-3 config ─────────────────────────────────────────────────────────────
L3_MODEL_NAME = "NASA-AIML/MIKA_SafeAeroBERT"
L3_MAX_LEN    = 256

# Category labels in the order used during training
# (matches the milestone2.md classification report)
# Dynamically populated on load
L3_LABELS = []

# Full descriptions for the identified ADREP classes based on ECCAIRS taxonomy
L3_LABEL_FULL = {
    "MAC": "Mid-Air Collision",
    "CFIT": "Controlled Flight Into or Toward Terrain",
    "GCOL": "Ground Collision",
    "SEC": "Security Related",
    "ATM": "ATM/Communication or Ground Issue",
    "LOC-I": "Loss of Control - Inflight",
    "TURB": "Turbulence Encounter",
    "RE": "Runway Excursion",
    "USOS": "Undershoot/Overshoot",
    "OTHR": "Other"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ══════════════════════════════════════════════════════════════════════════════
#  Model loading  (called once at Flask startup)
# ══════════════════════════════════════════════════════════════════════════════

_l1_tokenizer = None
_l1_model     = None
_l3_tokenizer = None
_l3_model     = None


def load_models():
    global _l1_tokenizer, _l1_model, _l3_tokenizer, _l3_model, L3_LABELS
    if _l3_model is not None:
        return

    # Load Label Mapping dynamically
    mapping_path = os.path.join(MODELS_DIR, "id2cat.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            id2cat = json.load(f)
        # Sort by int keys ascending
        L3_LABELS = [id2cat[str(i)] for i in range(len(id2cat))]
    else:
        print("Warning: id2cat.json not found, falling back to default 8 classes.")
        L3_LABELS = ['ATM', 'CFIT', 'GCOL', 'LOC-I', 'MAC', 'OTHR', 'SEC', 'TURB']

    print("⏳ Loading Layer-1 NER model …")
    _l1_tokenizer = AutoTokenizer.from_pretrained(L1_MODEL_NAME)
    _l1_model = AutoModelForTokenClassification.from_pretrained(
        L1_MODEL_NAME,
        num_labels=L1_NUM_LABELS,
        id2label=L1_ID2LABEL,
        label2id=L1_LABEL2ID,
    )
    if os.path.exists(L1_WEIGHTS):
        _l1_model.load_state_dict(
            torch.load(L1_WEIGHTS, map_location=DEVICE, weights_only=True)
        )
        print(f"  ✅ Layer-1 weights loaded from {os.path.basename(L1_WEIGHTS)}")
    else:
        print(f"  ⚠️ Warning: No fine-tuned weights at {L1_WEIGHTS}. Using base model.")
    _l1_model.to(DEVICE)
    _l1_model.eval()

    print("⏳ Loading Layer-3 Classifier model …")
    _l3_tokenizer = AutoTokenizer.from_pretrained(L3_MODEL_NAME)
    _l3_model = AutoModelForSequenceClassification.from_pretrained(L3_MODEL_NAME, num_labels=len(L3_LABELS))
    if os.path.exists(L3_WEIGHTS):
        _l3_model.load_state_dict(
            torch.load(L3_WEIGHTS, map_location=DEVICE, weights_only=True)
        )
        print(f"  ✅ Layer-3 weights loaded from {os.path.basename(L3_WEIGHTS)}")
    else:
        print(f"  ⚠️ Warning: No fine-tuned weights at {L3_WEIGHTS}. Using base model.")
    _l3_model.to(DEVICE)
    _l3_model.eval()

    print("🚀 Models are loaded and ready on:", DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
#  Layer-1: NER entity extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_entities(text: str) -> dict:
    """
    Returns a dict like:
      {
        "ACTOR":   ["Captain", "First Officer"],
        "SYSTEM":  ["autopilot"],
        "TRIGGER": ["turbulence"],
        "spans":   [{"start":0,"end":7,"type":"ACTOR","text":"Captain"}, ...]
      }
    """
    encoding = _l1_tokenizer(
        text,
        max_length=L1_MAX_LEN,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    offset_mapping = encoding["offset_mapping"][0].tolist()

    with torch.no_grad():
        logits = _l1_model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds  = torch.argmax(logits, dim=-1)[0].tolist()

    # Decode BIO → entity spans with character offsets
    entities = defaultdict(list)
    char_spans = []
    current_entity = None
    current_start  = None
    tok_end_prev   = None

    for pred_id, (tok_start, tok_end) in zip(preds, offset_mapping):
        if tok_start == 0 and tok_end == 0:          # [CLS] / [SEP] / [PAD]
            if current_entity and current_start is not None:
                span_text = text[current_start:tok_end_prev].strip()
                if span_text:
                    entities[current_entity].append(span_text)
                    char_spans.append({
                        "start": current_start, "end": tok_end_prev,
                        "type": current_entity, "text": span_text,
                    })
                current_entity = None
            continue

        pred_label = L1_ID2LABEL.get(pred_id, "O")

        if pred_label.startswith("B-"):
            if current_entity and current_start is not None:
                span_text = text[current_start:tok_end_prev].strip()
                if span_text:
                    entities[current_entity].append(span_text)
                    char_spans.append({
                        "start": current_start, "end": tok_end_prev,
                        "type": current_entity, "text": span_text,
                    })
            current_entity = pred_label[2:]
            current_start  = tok_start
            tok_end_prev   = tok_end

        elif pred_label.startswith("I-") and current_entity == pred_label[2:]:
            tok_end_prev = tok_end

        else:
            if current_entity and current_start is not None:
                span_text = text[current_start:tok_end_prev].strip()
                if span_text:
                    entities[current_entity].append(span_text)
                    char_spans.append({
                        "start": current_start, "end": tok_end_prev,
                        "type": current_entity, "text": span_text,
                    })
            current_entity = None
            current_start  = None

    # Flush end-of-sequence
    if current_entity and current_start is not None:
        span_text = text[current_start:tok_end_prev].strip()
        if span_text:
            entities[current_entity].append(span_text)
            char_spans.append({
                "start": current_start, "end": tok_end_prev,
                "type": current_entity, "text": span_text,
            })

    # De-duplicate
    for etype in entities:
        entities[etype] = list(dict.fromkeys(entities[etype]))

    return {
        "ACTOR":   entities.get("ACTOR", []),
        "SYSTEM":  entities.get("SYSTEM", []),
        "TRIGGER": entities.get("TRIGGER", []),
        "spans":   char_spans,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Layer-3: occurrence category classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_occurrence(text: str) -> dict:
    """
    Returns:
      {
        "top_class": "Controlled Flight Into or Toward Terrain",
        "confidence": 0.94,
        "top_5": [{"class": "CFIT", "confidence": 0.94}, ...]
      }
    """
    encoding = _l3_tokenizer(
        text,
        max_length=L3_MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = _l3_model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs  = torch.softmax(logits, dim=-1)[0].tolist()

    pred_idx    = int(torch.argmax(torch.tensor(probs)).item())
    prediction  = L3_LABELS[pred_idx]
    confidence  = probs[pred_idx]

    prediction_full = L3_LABEL_FULL.get(prediction, prediction)

    probabilities = [
        {
            "class":      L3_LABEL_FULL.get(L3_LABELS[i], L3_LABELS[i]),
            "confidence": round(probs[i], 4),
        }
        for i in range(len(L3_LABELS))
    ]
    probabilities.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "top_class": prediction_full,
        "confidence": round(confidence, 4),
        "top_5": probabilities[:5],
    }
