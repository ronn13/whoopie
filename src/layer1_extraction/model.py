# %% [markdown]
# # ADREP Event Extraction Model
#
# **Project:** Causal Graph Reasoning for Explainable ADREP Classification

# ---
# Pipeline Overview
#
# This implements **Layer 1 — Event Extraction** of the causal refinement
# module. The goal is to extract structured event tuples of the form:
#
# > **(ACTOR, SYSTEM, PHASE, TRIGGER, OUTCOME)**
#
# from aviation safety narratives initially classified as "OTHER" by the SDCPS pipeline.
#
# ### Approach: Hybrid Extraction
#
# | Entity    | Source                        | Method                              |
# |-----------|-------------------------------|-------------------------------------|
# | **PHASE** | Existing `phase` column       | Direct mapping (98% coverage)       |
# | **OUTCOME** | Existing `events` + `events_6` columns | Direct mapping (100% coverage) |
# | **ACTOR** | Narrative text                | Fine-tuned BERT NER                 |
# | **SYSTEM** | Narrative text               | Fine-tuned BERT NER                 |
# | **TRIGGER** | Narrative text              | Fine-tuned BERT NER                 |
#
# ### Training Data Strategy
#
# Since we have **no gold-annotated NER labels**, we use a **silver-labeling** approach:
# 1. Run enhanced rule-based extractors (regex + dictionaries) over the narratives
# 2. Align extracted spans to token positions → BIO tags
# 3. Fine-tune BERT on these silver labels
# 4. The model generalizes BEYOND the rules (learns contextual patterns)
#
# ---

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os
import re
import json
import random
import warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import (
    classification_report as seq_classification_report,
    f1_score as seq_f1_score,
)

warnings.filterwarnings("ignore")

# ── Reproducibility ─────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── CUDA / GPU Configuration ───────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    # Enable cuDNN auto-tuner to find the fastest convolution algorithms
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Enable TF32 for faster matmul on Ampere+ GPUs (RTX 30xx, A100, etc.)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Ensure CUDA is ready
    torch.cuda.empty_cache()

    print(f"🖥️  Device: {DEVICE} ✅ GPU acceleration enabled")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   cuDNN benchmark: ON")
    print(f"   TF32: ON")
else:
    print(f"🖥️  Device: {DEVICE}")
    print(f"   ⚠️  No GPU detected — training will be slower on CPU")

# %% [markdown]
# ## 2. Load Data

# %%
df = pd.read_csv("data_aviation.csv")
print(f"📊 Dataset: {df.shape[0]} reports, {df.shape[1]} columns")
print(f"📋 All labeled: {df['final_category'].unique()}")

# Quick look at a sample narrative
sample = df.loc[0, "narrative_1"]
print(f"\n📝 Sample narrative (first 300 chars):\n{sample[:300]}...")

# %% [markdown]
# ## 3. Silver Label Generation
#
# We build enhanced rule-based extractors that return **character-level spans**
# `(start, end, entity_type)` — not just the extracted text. This lets us align
# them precisely with BERT sub-word tokens for BIO tagging.

# %%
# ══════════════════════════════════════════════════════════════════════════
# 3.1  ACTOR Patterns — Who is involved?
# ══════════════════════════════════════════════════════════════════════════
ACTOR_PATTERNS = [
    # Specific roles
    r"\b(Captain)\b",
    r"\b(First Officer)\b",
    r"\b(FO)\b",
    r"\b(Pilot Flying)\b",
    r"\b(Pilot Monitoring)\b",
    r"\b(PIC)\b",
    r"\b(SIC)\b",
    r"\b(Flight Instructor)\b",
    r"\b(CFI)\b",
    r"\b(Student Pilot)\b",
    r"\b(Private Pilot)\b",
    r"\b(Commercial Pilot)\b",
    # ATC roles
    r"\b(Controller)\b",
    r"\b(Tower Controller)\b",
    r"\b(Approach Controller)\b",
    r"\b(TRACON Controller)\b",
    r"\b(Center Controller)\b",
    r"\b(Ground Controller)\b",
    r"\b(ATC)\b",
    # Crew & generic
    r"\b(Flight Crew)\b",
    r"\b(flight crew)\b",
    r"\b(Flight Attendant)\b",
    r"\b(Dispatcher)\b",
    r"\b(RPIC)\b",
    r"\b(Remote Pilot)\b",
    r"\b(the pilot)\b",
    r"\b(the copilot)\b",
    r"\b(the crew)\b",
    r"\b(our crew)\b",
    # Pronouns that often refer to pilots (contextual — lower confidence)
    r"\b(I (?:was|am|had|noticed|observed|decided|reported|called|requested|initiated))",
    r"\b(we (?:were|had|noticed|observed|decided|reported|called|requested|initiated))",
]

# ══════════════════════════════════════════════════════════════════════════
# 3.2  SYSTEM Patterns — Aircraft/equipment systems
# ══════════════════════════════════════════════════════════════════════════
SYSTEM_PATTERNS = [
    # Navigation
    r"\b(GPS)\b", r"\b(RNAV)\b", r"\b(VOR)\b", r"\b(ILS)\b",
    r"\b(LOC(?:alizer)?)\b", r"\b(Glideslope)\b", r"\b(FMS)\b", r"\b(FMC)\b",
    r"\b(G1000)\b", r"\b(G3000)\b", r"\b(WAAS)\b",
    # Safety systems
    r"\b(TCAS)\b", r"\b(ACAS)\b", r"\b(RA)\b",  # Resolution Advisory
    r"\b(EGPWS)\b", r"\b(GPWS)\b", r"\b(TAWS)\b",
    r"\b(Traffic Alert)\b",
    # Autopilot / automation
    r"\b(autopilot)\b", r"\b(auto ?pilot)\b",
    r"\b(autothrottle)\b", r"\b(auto ?throttle)\b",
    r"\b(flight director)\b", r"\b(LNAV)\b", r"\b(VNAV)\b",
    r"\b(FLCH)\b", r"\b(V/S mode)\b",
    # Communication
    r"\b(radio)\b", r"\b(CTAF)\b", r"\b(CPDLC)\b",
    r"\b(frequency)\b", r"\b(transponder)\b",
    # Surveillance
    r"\b(ADS-?B)\b", r"\b(radar)\b",
    # Visual aids
    r"\b(PAPI)\b", r"\b(VASI)\b",
    # UAS
    r"\b(drone)\b", r"\b(UAS)\b", r"\b(sUAS)\b", r"\b(quadcopter)\b",
    # Aircraft components
    r"\b(engine)\b", r"\b(landing gear)\b", r"\b(flaps)\b",
    r"\b(elevator)\b", r"\b(rudder)\b", r"\b(aileron)\b",
    r"\b(altimeter)\b", r"\b(airspeed indicator)\b",
]

# ══════════════════════════════════════════════════════════════════════════
# 3.3  TRIGGER Patterns — What initiated the event chain?
# ══════════════════════════════════════════════════════════════════════════
TRIGGER_PATTERNS = [
    # Causal language
    r"(?:due to|because of|caused by|resulted from|attributed to)\s+([^.;,]{5,80})",
    r"(?:failure to|failed to)\s+([^.;,]{5,60})",
    r"(?:distracted by|confused by|overwhelmed by)\s+([^.;,]{5,60})",
    r"(?:led to|leading to|resulting in)\s+([^.;,]{5,60})",
    # Human factors triggers
    r"\b(communication breakdown)\b",
    r"\b(loss of situational awareness)\b",
    r"\b(situational awareness)\b",
    r"\b(pilot deviation)\b",
    r"\b(workload)\b",
    r"\b(fatigue)\b",
    r"\b(distraction)\b",
    r"\b(confusion)\b",
    r"\b(complacency)\b",
    r"\b(miscommunication)\b",
    # Environmental triggers
    r"\b(turbulence)\b", r"\b(wind ?shear)\b",
    r"\b(icing|ice accumulation)\b",
    r"\b(thunderstorm)\b", r"\b(convective weather)\b",
    r"\b(low visibility)\b", r"\b(IMC)\b", r"\b(VMC)\b",
    r"\b(wake turbulence)\b",
    # Equipment/mechanical triggers
    r"\b(malfunction(?:ing)?)\b",
    r"\b(mechanical failure)\b",
    r"\b(system failure)\b",
    r"\b(bird strike)\b",
    r"\b(power loss)\b",
    r"\b(engine failure)\b",
]


def extract_spans(text, patterns, entity_type):
    """Extract character-level spans from text using regex patterns.

    Returns list of (start, end, entity_type, matched_text) tuples.
    """
    spans = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Use the first capturing group if it exists, otherwise full match
            if match.lastindex and match.lastindex >= 1:
                start, end = match.start(1), match.end(1)
                matched = match.group(1)
            else:
                start, end = match.start(), match.end()
                matched = match.group()
            spans.append((start, end, entity_type, matched.strip()))
    return spans


def resolve_overlaps(spans):
    """Remove overlapping spans, keeping the longest match."""
    if not spans:
        return []
    # Sort by start position, then by length (longest first)
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    resolved = [spans[0]]
    for span in spans[1:]:
        prev = resolved[-1]
        if span[0] >= prev[1]:  # No overlap
            resolved.append(span)
        elif (span[1] - span[0]) > (prev[1] - prev[0]):
            # Current span is longer — replace previous
            resolved[-1] = span
    return resolved


def silver_label_narrative(text):
    """Generate silver-label entity spans for a single narrative.

    Returns a list of non-overlapping (start, end, entity_type, text) spans.
    """
    all_spans = []
    all_spans.extend(extract_spans(text, ACTOR_PATTERNS, "ACTOR"))
    all_spans.extend(extract_spans(text, SYSTEM_PATTERNS, "SYSTEM"))
    all_spans.extend(extract_spans(text, TRIGGER_PATTERNS, "TRIGGER"))
    return resolve_overlaps(all_spans)


# ── Test silver labeling on a sample ────────────────────────────────────
sample_text = df.loc[0, "narrative_1"]
sample_spans = silver_label_narrative(sample_text)

print("═" * 70)
print("SILVER LABEL TEST — Report #0")
print("═" * 70)
print(f"\nNarrative (first 500 chars):\n{sample_text[:500]}\n")
print(f"Extracted {len(sample_spans)} entity spans:")
for start, end, etype, text in sample_spans:
    print(f"  [{start:4d}-{end:4d}] {etype:8s} → \"{text}\"")

# %%
# ── Run silver labeling on ALL narratives ───────────────────────────────
silver_data = []
for idx, row in df.iterrows():
    text = row["narrative_1"]
    spans = silver_label_narrative(text)
    silver_data.append(
        {
            "idx": idx,
            "text": text,
            "spans": spans,
            "phase": row.get("phase", "Unknown"),
            "outcome_events": row.get("events", ""),
            "outcome_actions": row.get("events_6", ""),
        }
    )

# Coverage stats
n_with_actor = sum(1 for d in silver_data if any(s[2] == "ACTOR" for s in d["spans"]))
n_with_system = sum(
    1 for d in silver_data if any(s[2] == "SYSTEM" for s in d["spans"])
)
n_with_trigger = sum(
    1 for d in silver_data if any(s[2] == "TRIGGER" for s in d["spans"])
)

print("\n" + "═" * 60)
print("SILVER LABEL COVERAGE")
print("═" * 60)
print(f"  ACTOR spans found:   {n_with_actor:>3}/{len(df)} reports")
print(f"  SYSTEM spans found:  {n_with_system:>3}/{len(df)} reports")
print(f"  TRIGGER spans found: {n_with_trigger:>3}/{len(df)} reports")

# Entity distribution
all_entity_counts = Counter()
for d in silver_data:
    for span in d["spans"]:
        all_entity_counts[span[2]] += 1
print(f"\n  Total entity mentions:")
for etype, count in all_entity_counts.most_common():
    print(f"    {etype:>8s}: {count}")

# %% [markdown]
# ## 4. Tokenization & BIO Tag Alignment
#
# We tokenize the narratives with BERT's sub-word tokenizer, then align the
# character-level silver spans to token-level **BIO** tags:
# - **B-ACTOR**: Beginning of an ACTOR entity
# - **I-ACTOR**: Inside (continuation) of an ACTOR entity
# - **O**: Outside any entity
#
# This is the standard NER tagging scheme.

# %%
# ── Configuration ───────────────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"  # Good balance of speed & quality
MAX_LEN = 256  # Most narratives fit; longer ones are chunked
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1

# ── Label setup ─────────────────────────────────────────────────────────
ENTITY_TYPES = ["ACTOR", "SYSTEM", "TRIGGER"]
# Standard BIO label set
LABELS = ["O"]
for etype in ENTITY_TYPES:
    LABELS.append(f"B-{etype}")
    LABELS.append(f"I-{etype}")

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

print(f"📏 Label set ({NUM_LABELS} labels):")
for label, idx in LABEL2ID.items():
    print(f"  {idx}: {label}")

# %%
# ── Tokenizer ───────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"\n✅ Tokenizer loaded: {MODEL_NAME}")
print(f"   Vocab size: {tokenizer.vocab_size}")


# %% [markdown]
# ### 4.1 Character-Span to Token-Span Alignment
#
# BERT tokenizes text into sub-word pieces. We need to figure out which
# sub-word tokens fall within each entity's character span, then assign
# `B-XXX` to the first token and `I-XXX` to the rest.

# %%
def align_spans_to_tokens(text, spans, tokenizer, max_len=MAX_LEN):
    """Align character-level entity spans to BIO token labels.

    Args:
        text: Raw text string
        spans: List of (start, end, entity_type, matched_text) character spans
        tokenizer: HuggingFace tokenizer
        max_len: Maximum sequence length

    Returns:
        encoding: Tokenizer output dict
        labels: List of label IDs aligned to tokens
    """
    # Tokenize with offset mapping so we know char→token alignment
    encoding = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    # Get offset mapping by token
    offset_mapping = encoding["offset_mapping"][0].tolist()  # [(start, end), ...]
    labels = [LABEL2ID["O"]] * max_len

    # Assign BIO tags
    for char_start, char_end, entity_type, _ in spans:
        entity_started = False
        for token_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            # Skip special tokens (CLS, SEP, PAD — they have (0,0))
            if tok_start == 0 and tok_end == 0 and token_idx != 0:
                continue

            # Check if this token overlaps with the entity span
            if tok_start < char_end and tok_end > char_start:
                if not entity_started:
                    labels[token_idx] = LABEL2ID[f"B-{entity_type}"]
                    entity_started = True
                else:
                    labels[token_idx] = LABEL2ID[f"I-{entity_type}"]

    # Set special tokens to -100 (ignored by loss function)
    for token_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == 0 and tok_end == 0:
            labels[token_idx] = -100  # [CLS], [SEP], [PAD]

    # The first token [CLS] at position 0 is also special
    labels[0] = -100

    return encoding, labels


# ── Quick sanity check ──────────────────────────────────────────────────
sample_encoding, sample_labels = align_spans_to_tokens(
    sample_text, sample_spans, tokenizer
)

print("═" * 70)
print("TOKEN-LEVEL BIO ALIGNMENT — Report #0 (first 40 tokens)")
print("═" * 70)
tokens = tokenizer.convert_ids_to_tokens(sample_encoding["input_ids"][0])
for i in range(min(40, len(tokens))):
    label_id = sample_labels[i]
    label_str = ID2LABEL[label_id] if label_id >= 0 else "[SPECIAL]"
    if label_str != "O" and label_str != "[SPECIAL]":
        print(f"  Token {i:3d}: {tokens[i]:15s} → {label_str}  ◄◄◄")
    elif tokens[i] != "[PAD]":
        print(f"  Token {i:3d}: {tokens[i]:15s} → {label_str}")

# %% [markdown]
# ## 5. PyTorch Dataset & DataLoader
#
# We handle long narratives by **chunking**: if a narrative exceeds `MAX_LEN`
# tokens, we split it into overlapping windows so we don't lose entities near
# chunk boundaries.

# %%
class AviationNERDataset(Dataset):
    """PyTorch Dataset for aviation event extraction NER."""

    def __init__(self, data_records, tokenizer, max_len=MAX_LEN, stride=128):
        """
        Args:
            data_records: List of dicts with 'text' and 'spans' keys
            tokenizer: HuggingFace tokenizer
            max_len: Max tokens per sample
            stride: Overlap between chunks for long texts
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        for record in data_records:
            text = record["text"]
            spans = record["spans"]
            idx = record["idx"]

            # Tokenize to check length
            test_enc = tokenizer(text, add_special_tokens=False)
            n_tokens = len(test_enc["input_ids"])

            if n_tokens <= max_len - 2:  # -2 for [CLS] and [SEP]
                # Fits in a single chunk
                encoding, labels = align_spans_to_tokens(
                    text, spans, tokenizer, max_len
                )
                self.samples.append(
                    {
                        "input_ids": encoding["input_ids"].squeeze(0),
                        "attention_mask": encoding["attention_mask"].squeeze(0),
                        "labels": torch.tensor(labels, dtype=torch.long),
                        "report_idx": idx,
                    }
                )
            else:
                # Split into overlapping chunks
                # Use character-level chunking based on approximate boundaries
                chunk_char_len = (max_len - 2) * 5  # ~5 chars per token
                stride_char_len = stride * 5

                start = 0
                while start < len(text):
                    end = min(start + chunk_char_len, len(text))
                    chunk_text = text[start:end]

                    # Adjust spans for this chunk
                    chunk_spans = []
                    for s_start, s_end, etype, matched in spans:
                        # Check if span overlaps with this chunk
                        if s_start < end and s_end > start:
                            new_start = max(0, s_start - start)
                            new_end = min(end - start, s_end - start)
                            chunk_spans.append(
                                (new_start, new_end, etype, matched)
                            )
                    # Align spans to tokens
                    encoding, labels = align_spans_to_tokens(
                        chunk_text, chunk_spans, tokenizer, max_len
                    )
                    self.samples.append(
                        {
                            "input_ids": encoding["input_ids"].squeeze(0),
                            "attention_mask": encoding["attention_mask"].squeeze(0),
                            "labels": torch.tensor(labels, dtype=torch.long),
                            "report_idx": idx,
                        }
                    )

                    if end >= len(text):
                        break
                    start += stride_char_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# %% [markdown]
# ### 5.1 Train/Validation/Test Split
#
# We use a **70/15/15** split, stratified to maintain entity type distribution.

# %%
# Shuffle and split
indices = list(range(len(silver_data)))
random.shuffle(indices)

n_train = int(0.70 * len(indices))
n_val = int(0.15 * len(indices))

train_indices = indices[:n_train]
val_indices = indices[n_train : n_train + n_val]
test_indices = indices[n_train + n_val :]

train_data = [silver_data[i] for i in train_indices]
val_data = [silver_data[i] for i in val_indices]
test_data = [silver_data[i] for i in test_indices]

print(f"📊 Split sizes:")
print(f"   Train: {len(train_data)} reports")
print(f"   Val:   {len(val_data)} reports")
print(f"   Test:  {len(test_data)} reports")

# Create datasets
train_dataset = AviationNERDataset(train_data, tokenizer)
val_dataset = AviationNERDataset(val_data, tokenizer)
test_dataset = AviationNERDataset(test_data, tokenizer)

print(f"\n📦 Dataset samples (after chunking):")
print(f"   Train: {len(train_dataset)} samples")
print(f"   Val:   {len(val_dataset)} samples")
print(f"   Test:  {len(test_dataset)} samples")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %% [markdown]
# ## 6. Model Architecture
#
# We use **BERT + Token Classification Head** — the standard architecture for NER:
#
# ```
# [narrative tokens] → BERT Encoder → Hidden States → Linear(768 → 7) → BIO Labels
# ```
#
# The model predicts a BIO label for each sub-word token. At inference time,
# we decode the labels back into entity spans.

# %%
# ── Class weights for imbalanced labels ─────────────────────────────────
# The O (Outside) class heavily dominates. We compute class weights to
# address this during training.
label_counts = Counter()
for sample in train_dataset.samples:
    for label_id in sample["labels"].tolist():
        if label_id >= 0:  # Skip special tokens (-100)
            label_counts[label_id] += 1

total = sum(label_counts.values())
class_weights = torch.zeros(NUM_LABELS)
for label_id in range(NUM_LABELS):
    count = label_counts.get(label_id, 1)
    # Inverse frequency weighting, capped to prevent extreme weights
    weight = total / (NUM_LABELS * count)
    class_weights[label_id] = min(weight, 20.0)  # Cap at 20x

# Normalize so O gets weight ~1.0
o_weight = class_weights[LABEL2ID["O"]]
class_weights = class_weights / o_weight

print("⚖️  Class weights (normalized):")
for label, idx in LABEL2ID.items():
    count = label_counts.get(idx, 0)
    print(f"  {label:12s}  count={count:>7,}  weight={class_weights[idx]:.2f}")

class_weights = class_weights.to(DEVICE)

# %%
# ── Load pre-trained BERT for Token Classification ──────────────────────
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n🧠 Model: {MODEL_NAME}")
print(f"   Total parameters:     {total_params:>12,}")
print(f"   Trainable parameters: {trainable_params:>12,}")
print(f"   Label dimensions:     {NUM_LABELS}")

# %% [markdown]
# ## 7. Training Loop
#
# Training uses:
# - **Weighted cross-entropy loss** to handle label imbalance (O >> entity tags)
# - **Linear warmup + decay** learning rate schedule
# - **Gradient clipping** (max norm = 1.0) for stability
# - **Early stopping** on validation F1 score

# %%
# ── Optimizer & Scheduler ───────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

# Weighted loss function
loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

print(f"⚙️  Training config:")
print(f"   Epochs:        {NUM_EPOCHS}")
print(f"   Batch size:    {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Total steps:   {total_steps}")
print(f"   Warmup steps:  {warmup_steps}")


# %%
def evaluate(model, dataloader, loss_fn):
    """Evaluate model on a dataset, returning loss and seqeval metrics."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute loss
            loss = loss_fn(logits.view(-1, NUM_LABELS), labels.view(-1))
            total_loss += loss.item()

            # Decode predictions
            preds = torch.argmax(logits, dim=-1)

            # Collect per-token predictions and labels (skip special tokens)
            for i in range(preds.size(0)):
                pred_seq = []
                label_seq = []
                for j in range(preds.size(1)):
                    if labels[i, j].item() != -100:
                        pred_seq.append(ID2LABEL[preds[i, j].item()])
                        label_seq.append(ID2LABEL[labels[i, j].item()])
                all_preds.append(pred_seq)
                all_labels.append(label_seq)

    avg_loss = total_loss / len(dataloader)
    f1 = seq_f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, f1, all_preds, all_labels


# %%
# ══════════════════════════════════════════════════════════════════════════
#                          TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("                    🚀 TRAINING STARTED")
print("═" * 70)

best_val_f1 = 0.0
best_epoch = 0
patience = 3
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

for epoch in range(NUM_EPOCHS):
    # ── Train ───────────────────────────────────────────────────────────
    model.train()
    epoch_loss = 0
    n_batches = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = loss_fn(logits.view(-1, NUM_LABELS), labels.view(-1))
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_train_loss = epoch_loss / n_batches

    # ── Evaluate on train (sample) and validation ───────────────────────
    train_loss_eval, train_f1, _, _ = evaluate(model, train_loader, loss_fn)
    val_loss, val_f1, _, _ = evaluate(model, val_loader, loss_fn)

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(val_loss)
    history["train_f1"].append(train_f1)
    history["val_f1"].append(val_f1)

    current_lr = scheduler.get_last_lr()[0]

    print(
        f"  Epoch {epoch+1:>2}/{NUM_EPOCHS}  │  "
        f"Train Loss: {avg_train_loss:.4f}  │  "
        f"Val Loss: {val_loss:.4f}  │  "
        f"Train F1: {train_f1:.4f}  │  "
        f"Val F1: {val_f1:.4f}  │  "
        f"LR: {current_lr:.2e}"
    )

    # ── Early stopping / checkpointing ──────────────────────────────────
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch = epoch + 1
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), "best_event_extractor.pt")
        print(f"         ✅ New best! Saved checkpoint (Val F1: {val_f1:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n  ⏹️  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

print(f"\n{'═'*70}")
print(f"  🏆 Best model: Epoch {best_epoch}, Val F1 = {best_val_f1:.4f}")
print(f"{'═'*70}")

# %% [markdown]
# ## 8. Training Visualization

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
axes[0].plot(history["train_loss"], label="Train Loss", marker="o", linewidth=2)
axes[0].plot(history["val_loss"], label="Val Loss", marker="s", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training & Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# F1 curves
axes[1].plot(history["train_f1"], label="Train F1", marker="o", linewidth=2, color="green")
axes[1].plot(history["val_f1"], label="Val F1", marker="s", linewidth=2, color="orange")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("F1 Score")
axes[1].set_title("Training & Validation F1 (Weighted)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=best_val_f1, color="red", linestyle="--", alpha=0.5, label="Best Val F1")

plt.suptitle("BERT NER Event Extractor — Training Curves", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("📈 Training curves saved to training_curves.png")

# %% [markdown]
# ## 9. Test Set Evaluation

# %%
# Load best model
model.load_state_dict(torch.load("best_event_extractor.pt", map_location=DEVICE))
model.eval()

test_loss, test_f1, test_preds, test_labels = evaluate(model, test_loader, loss_fn)

print("═" * 70)
print("                    📊 TEST SET EVALUATION")
print("═" * 70)
print(f"\n  Test Loss: {test_loss:.4f}")
print(f"  Test F1:   {test_f1:.4f}")
print(f"\n{seq_classification_report(test_labels, test_preds, zero_division=0)}")

# %% [markdown]
# ## 10. Inference — Full Event Extraction Pipeline
#
# Now we combine everything into a single inference function that takes
# a raw narrative and returns structured event tuples:
# **(ACTOR, SYSTEM, PHASE, TRIGGER, OUTCOME)**

# %%
def extract_entities_from_text(text, model, tokenizer, max_len=MAX_LEN):
    """Run NER inference on a narrative and return extracted entity spans.

    Returns:
        dict mapping entity types to lists of extracted text spans
    """
    model.eval()
    encoding = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    offset_mapping = encoding["offset_mapping"][0].tolist()
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)[0].tolist()

    # Decode BIO predictions back to entity spans
    entities = defaultdict(list)
    current_entity = None
    current_start = None
    current_text_parts = []

    for token_idx, (pred_id, (tok_start, tok_end)) in enumerate(
        zip(preds, offset_mapping)
    ):
        if tok_start == 0 and tok_end == 0:  # Special token
            if current_entity:  # Close any open entity
                entity_text = text[current_start:tok_end_prev].strip()
                if entity_text:
                    entities[current_entity].append(entity_text)
                current_entity = None
            continue

        pred_label = ID2LABEL.get(pred_id, "O")
        tok_end_prev = tok_end

        if pred_label.startswith("B-"):
            # Close previous entity if any
            if current_entity and current_start is not None:
                entity_text = text[current_start:tok_end_prev_close].strip()
                if entity_text:
                    entities[current_entity].append(entity_text)

            # Start new entity
            current_entity = pred_label[2:]
            current_start = tok_start
            tok_end_prev_close = tok_end

        elif pred_label.startswith("I-") and current_entity == pred_label[2:]:
            tok_end_prev_close = tok_end  # Extend entity

        else:
            # O label — close any open entity
            if current_entity and current_start is not None:
                entity_text = text[current_start:tok_end_prev_close].strip()
                if entity_text:
                    entities[current_entity].append(entity_text)
            current_entity = None
            current_start = None

    # Close final entity if it reaches end of sequence
    if current_entity and current_start is not None:
        entity_text = text[current_start:].strip()
        if entity_text:
            entities[current_entity].append(entity_text)

    # Deduplicate
    for etype in entities:
        entities[etype] = list(dict.fromkeys(entities[etype]))

    return dict(entities)


def extract_full_event_tuple(row, model, tokenizer):
    """Extract the complete (ACTOR, SYSTEM, PHASE, TRIGGER, OUTCOME) tuple.

    Combines NER-extracted entities with existing structured columns.
    """
    text = row["narrative_1"]

    # ── NER-based extraction (ACTOR, SYSTEM, TRIGGER) ───────────────────
    ner_entities = extract_entities_from_text(text, model, tokenizer)

    # ── Structured column extraction (PHASE, OUTCOME) ───────────────────
    # PHASE: from existing column
    phase = row.get("phase", "Unknown")
    if pd.isna(phase):
        phase = "Unknown"

    # OUTCOME: merge events + events_6
    outcomes = []
    if pd.notna(row.get("events", "")):
        outcomes.extend(
            [e.strip() for e in str(row["events"]).split(";") if e.strip()]
        )
    if pd.notna(row.get("events_6", "")):
        outcomes.extend(
            [e.strip() for e in str(row["events_6"]).split(";") if e.strip()]
        )
    outcomes = list(dict.fromkeys(outcomes))  # Deduplicate

    return {
        "ACTOR": ner_entities.get("ACTOR", ["Unknown"]),
        "SYSTEM": ner_entities.get("SYSTEM", ["Not identified"]),
        "PHASE": phase,
        "TRIGGER": ner_entities.get("TRIGGER", ["Not explicitly stated"]),
        "OUTCOME": outcomes if outcomes else ["Not specified"],
    }


# %% [markdown]
# ### 10.1 Demo: Extract Events from Sample Reports

# %%
print("═" * 80)
print("           🔬 EVENT EXTRACTION DEMO — Sample Reports")
print("═" * 80)

demo_indices = [0, 2, 5, 30, 73, 100, 150]
demo_results = []

for idx in demo_indices:
    if idx < len(df):
        row = df.loc[idx]
        event_tuple = extract_full_event_tuple(row, model, tokenizer)
        demo_results.append({"idx": idx, **event_tuple})

        print(f"\n{'─'*75}")
        print(f"  Report #{idx}")
        print(f"{'─'*75}")
        print(f"  Synopsis: {str(row['synopsis'])[:120]}...")
        print(f"\n  📌 ACTOR:   {event_tuple['ACTOR']}")
        print(f"  📌 SYSTEM:  {event_tuple['SYSTEM']}")
        print(f"  📌 PHASE:   {event_tuple['PHASE']}")
        print(f"  📌 TRIGGER: {event_tuple['TRIGGER']}")
        print(f"  📌 OUTCOME: {event_tuple['OUTCOME']}")

# %% [markdown]
# ## 11. Full Dataset Extraction & Export
#
# Run the extractor on ALL 173 reports and save structured results for
# the next pipeline stage (causal graph construction).

# %%
print("\n🔄 Extracting events from all reports...")
all_events = []

for idx, row in df.iterrows():
    event_tuple = extract_full_event_tuple(row, model, tokenizer)
    all_events.append(
        {
            "report_idx": idx,
            "narrative_preview": str(row["narrative_1"])[:150],
            **event_tuple,
        }
    )
    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{len(df)} reports...")

print(f"  ✅ Done! Extracted events from {len(all_events)} reports.")

# ── Save to JSON ────────────────────────────────────────────────────────
output_path = "extracted_events.json"
with open(output_path, "w") as f:
    json.dump(all_events, f, indent=2, default=str)
print(f"  💾 Saved to {output_path}")

# ── Also save a summary DataFrame ──────────────────────────────────────
summary_rows = []
for evt in all_events:
    summary_rows.append(
        {
            "report_idx": evt["report_idx"],
            "n_actors": len(evt["ACTOR"]),
            "n_systems": len(evt["SYSTEM"]),
            "phase": evt["PHASE"],
            "n_triggers": len(evt["TRIGGER"]),
            "n_outcomes": len(evt["OUTCOME"]),
            "actors": "; ".join(evt["ACTOR"]),
            "systems": "; ".join(evt["SYSTEM"]),
            "triggers": "; ".join(evt["TRIGGER"]),
            "outcomes": "; ".join(evt["OUTCOME"]),
        }
    )

events_df = pd.DataFrame(summary_rows)
events_df.to_csv("extracted_events_summary.csv", index=False)
print(f"  📊 Summary CSV saved to extracted_events_summary.csv")

# %% [markdown]
# ## 12. Extraction Quality Analysis

# %%
print("═" * 70)
print("              📊 EXTRACTION QUALITY ANALYSIS")
print("═" * 70)

# Coverage: what % of reports have at least one entity of each type?
actor_coverage = sum(1 for e in all_events if e["ACTOR"] != ["Unknown"]) / len(all_events)
system_coverage = sum(1 for e in all_events if e["SYSTEM"] != ["Not identified"]) / len(all_events)
phase_coverage = sum(1 for e in all_events if e["PHASE"] != "Unknown") / len(all_events)
trigger_coverage = sum(1 for e in all_events if e["TRIGGER"] != ["Not explicitly stated"]) / len(all_events)
outcome_coverage = sum(1 for e in all_events if e["OUTCOME"] != ["Not specified"]) / len(all_events)

print(f"\n  Entity Coverage (% of reports with ≥1 entity):")
print(f"    ACTOR:   {actor_coverage:>5.1%}")
print(f"    SYSTEM:  {system_coverage:>5.1%}")
print(f"    PHASE:   {phase_coverage:>5.1%}")
print(f"    TRIGGER: {trigger_coverage:>5.1%}")
print(f"    OUTCOME: {outcome_coverage:>5.1%}")

# Reports with ALL five entity types
full_coverage = sum(
    1
    for e in all_events
    if e["ACTOR"] != ["Unknown"]
    and e["SYSTEM"] != ["Not identified"]
    and e["PHASE"] != "Unknown"
    and e["TRIGGER"] != ["Not explicitly stated"]
    and e["OUTCOME"] != ["Not specified"]
) / len(all_events)

print(f"\n  ✅ Full tuple coverage (all 5 entities): {full_coverage:.1%}")

# Average entities per report
avg_actors = np.mean([len(e["ACTOR"]) for e in all_events if e["ACTOR"] != ["Unknown"]])
avg_systems = np.mean([len(e["SYSTEM"]) for e in all_events if e["SYSTEM"] != ["Not identified"]])
avg_triggers = np.mean([len(e["TRIGGER"]) for e in all_events if e["TRIGGER"] != ["Not explicitly stated"]])

print(f"\n  Average entities per report (when found):")
print(f"    ACTOR:   {avg_actors:.1f}")
print(f"    SYSTEM:  {avg_systems:.1f}")
print(f"    TRIGGER: {avg_triggers:.1f}")

# %% [markdown]
# ## 13. Visualization: Entity Distribution

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 10))

# Most common actors
all_actors = []
for e in all_events:
    if e["ACTOR"] != ["Unknown"]:
        all_actors.extend(e["ACTOR"])
actor_counts = Counter(all_actors).most_common(10)
if actor_counts:
    labels, counts = zip(*actor_counts)
    axes[0].barh(range(len(labels)), counts, color="#4A90D9", edgecolor="white")
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_xlabel("Count")
    axes[0].set_title("Top 10 Extracted ACTORS", fontweight="bold")
    axes[0].invert_yaxis()

# Most common systems
all_systems = []
for e in all_events:
    if e["SYSTEM"] != ["Not identified"]:
        all_systems.extend(e["SYSTEM"])
system_counts = Counter(all_systems).most_common(10)
if system_counts:
    labels, counts = zip(*system_counts)
    axes[1].barh(range(len(labels)), counts, color="#50C878", edgecolor="white")
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlabel("Count")
    axes[1].set_title("Top 10 Extracted SYSTEMS", fontweight="bold")
    axes[1].invert_yaxis()

# Most common triggers
all_triggers = []
for e in all_events:
    if e["TRIGGER"] != ["Not explicitly stated"]:
        all_triggers.extend(e["TRIGGER"])
trigger_counts = Counter(all_triggers).most_common(10)
if trigger_counts:
    labels, counts = zip(*trigger_counts)
    # Truncate long trigger labels
    labels = [l[:40] + "..." if len(l) > 40 else l for l in labels]
    axes[2].barh(range(len(labels)), counts, color="#FF6B6B", edgecolor="white")
    axes[2].set_yticks(range(len(labels)))
    axes[2].set_yticklabels(labels, fontsize=9)
    axes[2].set_xlabel("Count")
    axes[2].set_title("Top 10 Extracted TRIGGERS", fontweight="bold")
    axes[2].invert_yaxis()

plt.suptitle(
    "BERT NER Event Extractor — Entity Distribution (173 reports)",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("entity_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 Entity distribution saved to entity_distribution.png")

# %% [markdown]
# ## 14. Sample Event Tuples for Causal Graph Construction
#
# Preview the structured output that will feed into Layer 2 (Graph Construction).

# %%
print("═" * 80)
print("     📋 SAMPLE EVENT TUPLES → Ready for Causal Graph (Layer 2)")
print("═" * 80)

for i, evt in enumerate(all_events[:5]):
    print(f"\n{'─'*75}")
    print(f"  Report #{evt['report_idx']}")
    print(f"  Narrative: {evt['narrative_preview']}...")
    print(f"{'─'*75}")
    print(f"  EVENT TUPLE:")
    print(f"    ACTOR:   {evt['ACTOR']}")
    print(f"    SYSTEM:  {evt['SYSTEM']}")
    print(f"    PHASE:   {evt['PHASE']}")
    print(f"    TRIGGER: {evt['TRIGGER']}")
    print(f"    OUTCOME: {evt['OUTCOME']}")
    print()
    # Show potential graph edges
    print(f"  POTENTIAL CAUSAL EDGES:")
    for actor in evt["ACTOR"][:2]:
        for trigger in evt["TRIGGER"][:2]:
            print(f"    {actor} ──[caused]──▶ {trigger}")
    for trigger in evt["TRIGGER"][:2]:
        for outcome in evt["OUTCOME"][:2]:
            print(f"    {trigger} ──[led_to]──▶ {outcome}")
    for system in evt["SYSTEM"][:2]:
        for trigger in evt["TRIGGER"][:2]:
            print(f"    {system} ──[involved_in]──▶ {trigger}")

# %% [markdown]
# ## 15. Save Model & Artifacts
#
# Save everything needed for the downstream pipeline.

# %%
# ── Save model configuration ───────────────────────────────────────────
config = {
    "model_name": MODEL_NAME,
    "max_len": MAX_LEN,
    "num_labels": NUM_LABELS,
    "label2id": LABEL2ID,
    "id2label": {str(k): v for k, v in ID2LABEL.items()},
    "entity_types": ENTITY_TYPES,
    "best_epoch": best_epoch,
    "best_val_f1": best_val_f1,
    "training_config": {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
    },
}

with open("event_extractor_config.json", "w") as f:
    json.dump(config, f, indent=2)
print("💾 Model config saved to event_extractor_config.json")
print("💾 Best model weights at best_event_extractor.pt")
print("💾 Extracted events at extracted_events.json")
print("💾 Summary CSV at extracted_events_summary.csv")

# %% [markdown]
# ## 16. SafeAeroBERT Comparison
#
# NASA's **SafeAeroBERT** (`NASA-AIML/MIKA_SafeAeroBERT`) is a domain-specific BERT
# model pre-trained on ASRS and NTSB aviation safety narratives — the same type of
# data we're working with. Let's train it on our silver labels and compare its
# performance against the general-purpose `bert-base-uncased`.

# %%
# ══════════════════════════════════════════════════════════════════════════
#                   SAFEAEROBERT — SETUP & TRAINING
# ══════════════════════════════════════════════════════════════════════════

AERO_MODEL_NAME = "NASA-AIML/MIKA_SafeAeroBERT"

print(f"🛩️  Loading domain-specific model: {AERO_MODEL_NAME}")
aero_tokenizer = AutoTokenizer.from_pretrained(AERO_MODEL_NAME)
print(f"   Tokenizer vocab size: {aero_tokenizer.vocab_size}")

# ── Build datasets with AeroBERT tokenizer ──────────────────────────────
# We reuse the same silver_data and train/val/test splits for a fair comparison
print("📦 Building AeroBERT datasets...")

aero_train_dataset = AviationNERDataset(train_data, aero_tokenizer, max_len=MAX_LEN)
aero_val_dataset = AviationNERDataset(val_data, aero_tokenizer, max_len=MAX_LEN)
aero_test_dataset = AviationNERDataset(test_data, aero_tokenizer, max_len=MAX_LEN)

print(f"   Train: {len(aero_train_dataset)} samples")
print(f"   Val:   {len(aero_val_dataset)} samples")
print(f"   Test:  {len(aero_test_dataset)} samples")

aero_train_loader = DataLoader(aero_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
aero_val_loader = DataLoader(aero_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
aero_test_loader = DataLoader(aero_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
# ── Load SafeAeroBERT for Token Classification ─────────────────────────
aero_model = AutoModelForTokenClassification.from_pretrained(
    AERO_MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
aero_model = aero_model.to(DEVICE)

aero_total_params = sum(p.numel() for p in aero_model.parameters())
aero_trainable_params = sum(p.numel() for p in aero_model.parameters() if p.requires_grad)
print(f"\n🧠 Model: {AERO_MODEL_NAME}")
print(f"   Total parameters:     {aero_total_params:>12,}")
print(f"   Trainable parameters: {aero_trainable_params:>12,}")

# ── Compute class weights for AeroBERT datasets ────────────────────────
aero_label_counts = Counter()
for sample in aero_train_dataset.samples:
    for label_id in sample["labels"].tolist():
        if label_id >= 0:
            aero_label_counts[label_id] += 1

aero_total = sum(aero_label_counts.values())
aero_class_weights = torch.zeros(NUM_LABELS)
for label_id in range(NUM_LABELS):
    count = aero_label_counts.get(label_id, 1)
    weight = aero_total / (NUM_LABELS * count)
    aero_class_weights[label_id] = min(weight, 20.0)

aero_o_weight = aero_class_weights[LABEL2ID["O"]]
aero_class_weights = aero_class_weights / aero_o_weight
aero_class_weights = aero_class_weights.to(DEVICE)

# ── Optimizer & Scheduler ──────────────────────────────────────────────
aero_optimizer = AdamW(aero_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

aero_total_steps = len(aero_train_loader) * NUM_EPOCHS
aero_warmup_steps = int(aero_total_steps * WARMUP_RATIO)

aero_scheduler = get_linear_schedule_with_warmup(
    aero_optimizer,
    num_warmup_steps=aero_warmup_steps,
    num_training_steps=aero_total_steps,
)

aero_loss_fn = nn.CrossEntropyLoss(weight=aero_class_weights, ignore_index=-100)

print(f"\n⚙️  AeroBERT Training config (same hyperparameters for fair comparison):")
print(f"   Epochs:        {NUM_EPOCHS}")
print(f"   Batch size:    {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")

# %%
# ══════════════════════════════════════════════════════════════════════════
#                   SAFEAEROBERT — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("              🛩️  SAFEAEROBERT TRAINING STARTED")
print("═" * 70)

aero_best_val_f1 = 0.0
aero_best_epoch = 0
aero_patience_counter = 0
aero_history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

for epoch in range(NUM_EPOCHS):
    # ── Train ───────────────────────────────────────────────────────────
    aero_model.train()
    epoch_loss = 0
    n_batches = 0

    for batch in aero_train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        aero_optimizer.zero_grad()
        outputs = aero_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = aero_loss_fn(logits.view(-1, NUM_LABELS), labels.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(aero_model.parameters(), max_norm=1.0)

        aero_optimizer.step()
        aero_scheduler.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_train_loss = epoch_loss / n_batches

    # ── Evaluate ────────────────────────────────────────────────────────
    train_loss_eval, train_f1, _, _ = evaluate(aero_model, aero_train_loader, aero_loss_fn)
    val_loss, val_f1, _, _ = evaluate(aero_model, aero_val_loader, aero_loss_fn)

    aero_history["train_loss"].append(avg_train_loss)
    aero_history["val_loss"].append(val_loss)
    aero_history["train_f1"].append(train_f1)
    aero_history["val_f1"].append(val_f1)

    current_lr = aero_scheduler.get_last_lr()[0]

    print(
        f"  Epoch {epoch+1:>2}/{NUM_EPOCHS}  │  "
        f"Train Loss: {avg_train_loss:.4f}  │  "
        f"Val Loss: {val_loss:.4f}  │  "
        f"Train F1: {train_f1:.4f}  │  "
        f"Val F1: {val_f1:.4f}  │  "
        f"LR: {current_lr:.2e}"
    )

    # ── Early stopping / checkpointing ──────────────────────────────────
    if val_f1 > aero_best_val_f1:
        aero_best_val_f1 = val_f1
        aero_best_epoch = epoch + 1
        aero_patience_counter = 0
        torch.save(aero_model.state_dict(), "best_aerobert_event_extractor.pt")
        print(f"         ✅ New best! Saved checkpoint (Val F1: {val_f1:.4f})")
    else:
        aero_patience_counter += 1
        if aero_patience_counter >= patience:
            print(f"\n  ⏹️  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

print(f"\n{'═'*70}")
print(f"  🏆 AeroBERT Best: Epoch {aero_best_epoch}, Val F1 = {aero_best_val_f1:.4f}")
print(f"{'═'*70}")

# %%
# ══════════════════════════════════════════════════════════════════════════
#                   SAFEAEROBERT — TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════
aero_model.load_state_dict(
    torch.load("best_aerobert_event_extractor.pt", map_location=DEVICE)
)
aero_model.eval()

aero_test_loss, aero_test_f1, aero_test_preds, aero_test_labels = evaluate(
    aero_model, aero_test_loader, aero_loss_fn
)

print("═" * 70)
print("              📊 SAFEAEROBERT — TEST SET EVALUATION")
print("═" * 70)
print(f"\n  Test Loss: {aero_test_loss:.4f}")
print(f"  Test F1:   {aero_test_f1:.4f}")
print(f"\n{seq_classification_report(aero_test_labels, aero_test_preds, zero_division=0)}")

# %% [markdown]
# ## 17. BERT vs SafeAeroBERT — Side-by-Side Comparison
#
# Compare the two models across all metrics to determine whether the
# domain-specific pre-training provides a measurable advantage.

# %%
# ══════════════════════════════════════════════════════════════════════════
#                  COMPARISON: BERT vs SafeAeroBERT
# ══════════════════════════════════════════════════════════════════════════
from seqeval.metrics import precision_score as seq_precision_score
from seqeval.metrics import recall_score as seq_recall_score

# ── Re-evaluate BERT on test set for consistent comparison ──────────────
model.load_state_dict(torch.load("best_event_extractor.pt", map_location=DEVICE))
model.eval()
bert_test_loss, bert_test_f1, bert_test_preds, bert_test_labels = evaluate(
    model, test_loader, loss_fn
)

bert_precision = seq_precision_score(bert_test_labels, bert_test_preds, average="weighted", zero_division=0)
bert_recall = seq_recall_score(bert_test_labels, bert_test_preds, average="weighted", zero_division=0)

aero_precision = seq_precision_score(aero_test_labels, aero_test_preds, average="weighted", zero_division=0)
aero_recall = seq_recall_score(aero_test_labels, aero_test_preds, average="weighted", zero_division=0)

print("═" * 70)
print("        ⚖️  BERT vs SafeAeroBERT — HEAD-TO-HEAD COMPARISON")
print("═" * 70)
print(f"\n  {'Metric':<20s} {'BERT-base':>12s} {'SafeAeroBERT':>14s} {'Δ (Aero - BERT)':>16s}")
print(f"  {'─'*62}")
print(f"  {'Test Loss':<20s} {bert_test_loss:>12.4f} {aero_test_loss:>14.4f} {aero_test_loss - bert_test_loss:>+16.4f}")
print(f"  {'Test F1 (weighted)':<20s} {bert_test_f1:>12.4f} {aero_test_f1:>14.4f} {aero_test_f1 - bert_test_f1:>+16.4f}")
print(f"  {'Precision':<20s} {bert_precision:>12.4f} {aero_precision:>14.4f} {aero_precision - bert_precision:>+16.4f}")
print(f"  {'Recall':<20s} {bert_recall:>12.4f} {aero_recall:>14.4f} {aero_recall - bert_recall:>+16.4f}")
print(f"  {'Best Val F1':<20s} {best_val_f1:>12.4f} {aero_best_val_f1:>14.4f} {aero_best_val_f1 - best_val_f1:>+16.4f}")
print(f"  {'Best Epoch':<20s} {best_epoch:>12d} {aero_best_epoch:>14d}")
print(f"  {'Parameters':<20s} {trainable_params:>12,} {aero_trainable_params:>14,}")

winner = "SafeAeroBERT 🛩️" if aero_test_f1 > bert_test_f1 else "BERT-base 🤖" if bert_test_f1 > aero_test_f1 else "Tie 🤝"
print(f"\n  🏆 Winner (by Test F1): {winner}")

# ── Per-entity type comparison ──────────────────────────────────────────
print(f"\n  {'─'*62}")
print(f"  Per-Entity F1 Breakdown:")
print(f"  {'─'*62}")

from seqeval.metrics import classification_report as seq_classification_report_dict

# Get per-entity metrics using seqeval
bert_report_str = seq_classification_report(bert_test_labels, bert_test_preds, zero_division=0, output_dict=False)
aero_report_str = seq_classification_report(aero_test_labels, aero_test_preds, zero_division=0, output_dict=False)

print(f"\n  BERT-base detailed report:")
print(f"  {bert_report_str}")
print(f"\n  SafeAeroBERT detailed report:")
print(f"  {aero_report_str}")

# %%
# ── Visualization: Training curves comparison ───────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Loss curves — BERT
axes[0, 0].plot(history["train_loss"], label="Train", marker="o", linewidth=2, color="#4A90D9")
axes[0, 0].plot(history["val_loss"], label="Val", marker="s", linewidth=2, color="#FF6B6B")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("BERT-base — Loss", fontweight="bold")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss curves — AeroBERT
axes[0, 1].plot(aero_history["train_loss"], label="Train", marker="o", linewidth=2, color="#4A90D9")
axes[0, 1].plot(aero_history["val_loss"], label="Val", marker="s", linewidth=2, color="#FF6B6B")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].set_title("SafeAeroBERT — Loss", fontweight="bold")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# F1 curves — BERT
axes[1, 0].plot(history["train_f1"], label="Train", marker="o", linewidth=2, color="#50C878")
axes[1, 0].plot(history["val_f1"], label="Val", marker="s", linewidth=2, color="#FFA500")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("F1 Score")
axes[1, 0].set_title("BERT-base — F1 (Weighted)", fontweight="bold")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=best_val_f1, color="red", linestyle="--", alpha=0.5, label="Best")

# F1 curves — AeroBERT
axes[1, 1].plot(aero_history["train_f1"], label="Train", marker="o", linewidth=2, color="#50C878")
axes[1, 1].plot(aero_history["val_f1"], label="Val", marker="s", linewidth=2, color="#FFA500")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("F1 Score")
axes[1, 1].set_title("SafeAeroBERT — F1 (Weighted)", fontweight="bold")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=aero_best_val_f1, color="red", linestyle="--", alpha=0.5, label="Best")

plt.suptitle(
    "BERT-base vs SafeAeroBERT — Training Curves Comparison",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("model_comparison_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("📈 Comparison curves saved to model_comparison_curves.png")

# %%
# ── Bar chart: Test metrics side by side ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ["F1 Score", "Precision", "Recall"]
bert_scores = [bert_test_f1, bert_precision, bert_recall]
aero_scores = [aero_test_f1, aero_precision, aero_recall]

x = np.arange(len(metrics))
width = 0.3

bars1 = ax.bar(x - width / 2, bert_scores, width, label="BERT-base", color="#4A90D9", edgecolor="white", linewidth=1.5)
bars2 = ax.bar(x + width / 2, aero_scores, width, label="SafeAeroBERT", color="#FF6B6B", edgecolor="white", linewidth=1.5)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

ax.set_ylabel("Score", fontsize=12)
ax.set_title("BERT-base vs SafeAeroBERT — Test Set Performance", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("model_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.show()
print("📊 Comparison bar chart saved to model_comparison_bar.png")

# ── Save comparison results ─────────────────────────────────────────────
comparison_results = {
    "bert_base": {
        "model_name": MODEL_NAME,
        "test_f1": bert_test_f1,
        "test_loss": bert_test_loss,
        "precision": bert_precision,
        "recall": bert_recall,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "trainable_params": trainable_params,
    },
    "safe_aero_bert": {
        "model_name": AERO_MODEL_NAME,
        "test_f1": aero_test_f1,
        "test_loss": aero_test_loss,
        "precision": aero_precision,
        "recall": aero_recall,
        "best_val_f1": aero_best_val_f1,
        "best_epoch": aero_best_epoch,
        "trainable_params": aero_trainable_params,
    },
    "winner": winner,
}

with open("model_comparison_results.json", "w") as f:
    json.dump(comparison_results, f, indent=2, default=str)
print("💾 Comparison results saved to model_comparison_results.json")

# %% [markdown]
# ## 18. Summary & Next Steps
#
# ### What we built:
# - ✅ **Silver label generator** — rule-based extractors that produce BIO-tagged training data
# - ✅ **BERT NER model** — fine-tuned `bert-base-uncased` for aviation event extraction
# - ✅ **SafeAeroBERT NER model** — fine-tuned NASA's domain-specific model for comparison
# - ✅ **Model comparison** — head-to-head evaluation on the same test set
# - ✅ **Hybrid pipeline** — NER for ACTOR/SYSTEM/TRIGGER, structured columns for PHASE/OUTCOME
# - ✅ **Inference function** — `extract_full_event_tuple()` takes a row and returns the 5-tuple
# - ✅ **Exported results** — JSON + CSV ready for Layer 2 (causal graph construction)
#
# ### Model Performance:
# - Trained on silver labels (rule-based), so F1 scores measure agreement with rules
# - The BERT model should **generalize beyond** the rules by learning contextual patterns
# - SafeAeroBERT may show improved performance due to aviation-domain pre-training
# - True quality assessment requires manual annotation of a subset (10-20 reports)
#
# ### Next Steps for the Pipeline:
# 1. **Manual validation** — annotate 10-20 reports to measure true extraction quality
# 2. **Layer 2 — Causal Graph Construction** — build DAGs from extracted event tuples
# 3. **Layer 3 — Graph Reasoning** — train GAT/HGT on the causal graphs
# 4. **Layer 4 — ADREP Reclassification** — map graph embeddings to ADREP categories
#
# ### Potential Improvements:
# - Add a **CRF layer** on top of BERT for structured prediction (enforces valid BIO sequences)
# - **Active learning** — use model uncertainty to select reports for manual annotation
# - **LLM-augmented silver labels** — use Gemini/GPT to generate richer training data
# - **Multi-task learning** — jointly predict entities and causal relations
