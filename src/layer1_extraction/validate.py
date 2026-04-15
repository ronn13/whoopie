# %% [markdown]
# # 🔍 Manual Validation of Event Extraction
#
# **Purpose:** Measure the *true* quality of our BERT NER event extractor by
# comparing its outputs against human judgment on a sample of 20 reports.
#
# **Why this matters:**
# The model was trained on silver labels (regex-generated), so the F1 scores
# from training only tell us how well the model learned to *imitate the rules*.
# This validation tells us how well the model extracts *actual* entities.
#
# **Workflow:**
# 1. Load the trained model and run inference on 20 sampled reports
# 2. For each report, display the narrative and model predictions
# 3. The annotator reviews each entity: ✅ Accept, ✏️ Correct, or ❌ Reject
# 4. The annotator can also add entities the model missed
# 5. Compute true Precision, Recall, and F1 against human annotations
# 6. Save everything for reproducibility
#
# ---

# %% [markdown]
# ## 1. Setup & Load Model

# %%
import os
import re
import json
import random
import warnings
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

warnings.filterwarnings("ignore")

# ── Reproducibility ─────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {DEVICE}")

# ── Configuration ───────────────────────────────────────────────────────
NUM_SAMPLES = 20  # Number of reports to validate
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256
ANNOTATION_FILE = "manual_annotations.json"

# Label setup (must match training)
ENTITY_TYPES = ["ACTOR", "SYSTEM", "TRIGGER"]
LABELS = ["O"]
for etype in ENTITY_TYPES:
    LABELS.append(f"B-{etype}")
    LABELS.append(f"I-{etype}")

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

# %% [markdown]
# ## 2. Load Data & Model

# %%
# ── Load dataset ────────────────────────────────────────────────────────
df = pd.read_csv("data_aviation.csv")
print(f"📊 Dataset: {df.shape[0]} reports")

# ── Load tokenizer & model ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# Load trained weights
model_path = "best_event_extractor.pt"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"✅ Loaded trained weights from {model_path}")
else:
    print(f"⚠️  {model_path} not found — using untrained model (results will be random)")

model = model.to(DEVICE)
model.eval()

# %% [markdown]
# ## 3. Inference Function

# %%
def extract_entities_from_text(text, model, tokenizer, max_len=MAX_LEN):
    """Run NER inference on a narrative and return extracted entity spans."""
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

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)[0].tolist()

    # Decode BIO predictions back to entity spans
    entities = defaultdict(list)
    current_entity = None
    current_start = None
    tok_end_prev_close = None

    for token_idx, (pred_id, (tok_start, tok_end)) in enumerate(
        zip(preds, offset_mapping)
    ):
        if tok_start == 0 and tok_end == 0:  # Special token
            if current_entity and current_start is not None:
                entity_text = text[current_start:tok_end_prev_close].strip()
                if entity_text:
                    entities[current_entity].append(entity_text)
                current_entity = None
            continue

        pred_label = ID2LABEL.get(pred_id, "O")

        if pred_label.startswith("B-"):
            if current_entity and current_start is not None:
                entity_text = text[current_start:tok_end_prev_close].strip()
                if entity_text:
                    entities[current_entity].append(entity_text)
            current_entity = pred_label[2:]
            current_start = tok_start
            tok_end_prev_close = tok_end

        elif pred_label.startswith("I-") and current_entity == pred_label[2:]:
            tok_end_prev_close = tok_end

        else:
            if current_entity and current_start is not None:
                entity_text = text[current_start:tok_end_prev_close].strip()
                if entity_text:
                    entities[current_entity].append(entity_text)
            current_entity = None
            current_start = None

    if current_entity and current_start is not None:
        entity_text = text[current_start:].strip()
        if entity_text:
            entities[current_entity].append(entity_text)

    for etype in entities:
        entities[etype] = list(dict.fromkeys(entities[etype]))

    return dict(entities)


# %% [markdown]
# ## 4. Sample Selection
#
# We select a **stratified sample** of 20 reports to get a representative mix.

# %%
# ── Check for existing annotations (resume support) ────────────────────
existing_annotations = {}
if os.path.exists(ANNOTATION_FILE):
    with open(ANNOTATION_FILE, "r") as f:
        existing_data = json.load(f)
        existing_annotations = {a["report_idx"]: a for a in existing_data.get("annotations", [])}
    print(f"📂 Found {len(existing_annotations)} existing annotations — will skip those")

# ── Sample reports ──────────────────────────────────────────────────────
all_indices = list(range(len(df)))
random.shuffle(all_indices)
sample_indices = sorted(all_indices[:NUM_SAMPLES])

print(f"\n📋 Selected {NUM_SAMPLES} reports for validation: {sample_indices}")

# ── Run model inference on all samples ──────────────────────────────────
print("\n🔄 Running model inference on sample reports...")
sample_predictions = {}
for idx in sample_indices:
    text = df.loc[idx, "narrative_1"]
    predictions = extract_entities_from_text(text, model, tokenizer)
    sample_predictions[idx] = predictions

print(f"✅ Inference complete for {len(sample_predictions)} reports")

# %% [markdown]
# ## 5. Interactive Annotation
#
# For each report, you'll see:
# - The narrative text
# - The model's predictions for each entity type
# - Options to **accept**, **correct**, or **reject** each entity
# - Option to **add missed** entities
#
# ### Input Guide:
# - Press **Enter** or type `y` → Accept all predictions for this entity type
# - Type `n` → Reject all predictions (model got it wrong)
# - Type corrections as comma-separated values → Replace predictions
# - Type `+entity1, entity2` → Add missed entities
# - Type `s` → Skip this report
# - Type `q` → Quit and save progress

# %%
annotations = []
skipped = 0
quit_early = False

print("\n" + "═" * 80)
print("        🔍 MANUAL VALIDATION — Interactive Annotation")
print("═" * 80)
print("\nFor each entity type, you'll see the model's predictions.")
print("  [Enter] or 'y'  → Accept predictions as correct")
print("  'n'             → Reject all (model was wrong, no entities here)")
print("  'type, new, values' → Replace with your corrections")
print("  '+missed1, missed2' → Add entities the model missed")
print("  's'             → Skip this report")
print("  'q'             → Quit and save progress")
print("─" * 80)

for i, idx in enumerate(sample_indices):
    # Skip if already annotated
    if idx in existing_annotations:
        annotations.append(existing_annotations[idx])
        print(f"\n  ⏭️  Report #{idx} — already annotated, skipping")
        continue

    row = df.loc[idx]
    text = row["narrative_1"]
    preds = sample_predictions[idx]

    print(f"\n{'═' * 80}")
    print(f"  📝 Report #{idx}  ({i+1}/{NUM_SAMPLES})")
    print(f"{'═' * 80}")
    print(f"\n  Narrative (first 600 chars):")
    print(f"  {text[:600]}{'...' if len(text) > 600 else ''}")
    print()

    annotation = {
        "report_idx": idx,
        "model_predictions": preds,
        "human_judgments": {},
        "timestamp": datetime.now().isoformat(),
    }

    skip_report = False

    for etype in ENTITY_TYPES:
        predicted = preds.get(etype, [])
        print(f"\n  📌 {etype}: {predicted if predicted else '[none predicted]'}")

        response = input(f"     Accept? [Enter=yes / n=reject / corrections / +missed / s=skip / q=quit]: ").strip()

        if response.lower() == "q":
            quit_early = True
            break
        elif response.lower() == "s":
            skip_report = True
            skipped += 1
            break
        elif response == "" or response.lower() == "y":
            # Accept — predictions are correct
            annotation["human_judgments"][etype] = {
                "status": "accepted",
                "correct_entities": predicted,
                "model_correct": True,
                "missed": [],
                "false_positives": [],
            }
            print(f"     ✅ Accepted")
        elif response.lower() == "n":
            # Reject — no valid entities of this type
            missed_input = input(f"     Any {etype}s the model missed? (comma-separated, or Enter for none): ").strip()
            missed = [m.strip() for m in missed_input.split(",") if m.strip()] if missed_input else []

            annotation["human_judgments"][etype] = {
                "status": "rejected",
                "correct_entities": missed,
                "model_correct": False,
                "missed": missed,
                "false_positives": predicted,
            }
            print(f"     ❌ Rejected. {len(predicted)} false positive(s), {len(missed)} missed")
        elif response.startswith("+"):
            # Add missed entities while keeping predictions
            added = [m.strip() for m in response[1:].split(",") if m.strip()]
            all_correct = predicted + added

            annotation["human_judgments"][etype] = {
                "status": "partially_correct",
                "correct_entities": all_correct,
                "model_correct": True,  # Predictions were right, just incomplete
                "missed": added,
                "false_positives": [],
            }
            print(f"     ✏️  Accepted predictions + added {len(added)} missed: {added}")
        else:
            # Replace with corrections
            corrections = [c.strip() for c in response.split(",") if c.strip()]

            # Determine which predictions were right and which were wrong
            correct_preds = [p for p in predicted if p.lower() in [c.lower() for c in corrections]]
            false_positives = [p for p in predicted if p.lower() not in [c.lower() for c in corrections]]
            missed = [c for c in corrections if c.lower() not in [p.lower() for p in predicted]]

            annotation["human_judgments"][etype] = {
                "status": "corrected",
                "correct_entities": corrections,
                "model_correct": len(false_positives) == 0 and len(missed) == 0,
                "missed": missed,
                "false_positives": false_positives,
            }
            print(f"     ✏️  Corrected → {corrections}")
            if false_positives:
                print(f"        False positives: {false_positives}")
            if missed:
                print(f"        Missed: {missed}")

    if quit_early:
        print("\n  ⏹️  Quitting — saving progress...")
        break

    if not skip_report:
        annotations.append(annotation)

print(f"\n{'═' * 80}")
print(f"  ✅ Annotated: {len(annotations)} reports")
print(f"  ⏭️  Skipped:   {skipped} reports")
print(f"{'═' * 80}")

# %% [markdown]
# ## 6. Save Annotations

# %%
# ── Save annotations to JSON ───────────────────────────────────────────
output = {
    "metadata": {
        "model": MODEL_NAME,
        "model_weights": model_path,
        "num_samples": NUM_SAMPLES,
        "num_annotated": len(annotations),
        "num_skipped": skipped,
        "annotation_date": datetime.now().isoformat(),
        "entity_types": ENTITY_TYPES,
    },
    "annotations": annotations,
}

with open(ANNOTATION_FILE, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"💾 Annotations saved to {ANNOTATION_FILE}")

# %% [markdown]
# ## 7. Compute True Extraction Quality
#
# Now we calculate **true Precision, Recall, and F1** by comparing the model's
# predictions against the human annotations.
#
# - **Precision** = correct predictions / total predictions
# - **Recall** = correct predictions / total actual entities
# - **F1** = harmonic mean of Precision and Recall

# %%
# ── Load annotations (in case running separately) ──────────────────────
if not annotations:
    if os.path.exists(ANNOTATION_FILE):
        with open(ANNOTATION_FILE, "r") as f:
            data = json.load(f)
            annotations = data["annotations"]
        print(f"📂 Loaded {len(annotations)} annotations from {ANNOTATION_FILE}")
    else:
        print("⚠️  No annotations found. Run the annotation cells first.")

# ── Calculate metrics per entity type ───────────────────────────────────
if annotations:
    print("\n" + "═" * 70)
    print("        📊 TRUE EXTRACTION QUALITY (vs Human Annotations)")
    print("═" * 70)

    overall_tp = 0  # True positives
    overall_fp = 0  # False positives
    overall_fn = 0  # False negatives (missed)

    per_entity_metrics = {}

    for etype in ENTITY_TYPES:
        tp = 0  # Model predicted correctly
        fp = 0  # Model predicted but wrong
        fn = 0  # Model missed

        for ann in annotations:
            if etype not in ann.get("human_judgments", {}):
                continue

            judgment = ann["human_judgments"][etype]
            fp += len(judgment.get("false_positives", []))
            fn += len(judgment.get("missed", []))

            # True positives: predictions that were correct
            predicted = ann.get("model_predictions", {}).get(etype, [])
            false_pos = judgment.get("false_positives", [])
            tp += len(predicted) - len(false_pos)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_entity_metrics[etype] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

        print(f"\n  {etype}:")
        print(f"    True Positives:  {tp:>3d}")
        print(f"    False Positives: {fp:>3d}")
        print(f"    Missed (FN):     {fn:>3d}")
        print(f"    Precision:       {precision:.3f}")
        print(f"    Recall:          {recall:.3f}")
        print(f"    F1 Score:        {f1:.3f}")

    # Overall metrics
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    print(f"\n  {'─' * 50}")
    print(f"  OVERALL (macro-averaged):")
    print(f"    True Positives:  {overall_tp:>3d}")
    print(f"    False Positives: {overall_fp:>3d}")
    print(f"    Missed (FN):     {overall_fn:>3d}")
    print(f"    Precision:       {overall_precision:.3f}")
    print(f"    Recall:          {overall_recall:.3f}")
    print(f"    F1 Score:        {overall_f1:.3f}")

    # ── Quality rating ──────────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    if overall_f1 >= 0.80:
        print(f"  🟢 EXCELLENT — F1 ≥ 0.80. Model extractions are reliable.")
    elif overall_f1 >= 0.60:
        print(f"  🟡 GOOD — F1 ≥ 0.60. Model is useful but has notable gaps.")
    elif overall_f1 >= 0.40:
        print(f"  🟠 FAIR — F1 ≥ 0.40. Consider improving silver labels or using a larger model.")
    else:
        print(f"  🔴 POOR — F1 < 0.40. Silver labels may be too noisy. Consider LLM-based labeling.")

# %% [markdown]
# ## 8. Validation Visualization

# %%
import matplotlib.pyplot as plt

if annotations and per_entity_metrics:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Per-entity F1 bar chart ─────────────────────────────────────────
    entity_names = list(per_entity_metrics.keys())
    f1_scores = [per_entity_metrics[e]["f1"] for e in entity_names]
    precision_scores = [per_entity_metrics[e]["precision"] for e in entity_names]
    recall_scores = [per_entity_metrics[e]["recall"] for e in entity_names]

    x = np.arange(len(entity_names))
    width = 0.25

    axes[0].bar(x - width, precision_scores, width, label="Precision", color="#4A90D9", edgecolor="white")
    axes[0].bar(x, recall_scores, width, label="Recall", color="#50C878", edgecolor="white")
    axes[0].bar(x + width, f1_scores, width, label="F1", color="#FF6B6B", edgecolor="white")

    axes[0].set_xlabel("Entity Type")
    axes[0].set_ylabel("Score")
    axes[0].set_title("True Extraction Quality by Entity Type", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(entity_names)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar_group in [
        zip(x - width, precision_scores),
        zip(x, recall_scores),
        zip(x + width, f1_scores),
    ]:
        for xpos, score in bar_group:
            axes[0].annotate(
                f"{score:.2f}", xy=(xpos, score), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=9, fontweight="bold",
            )

    # ── Error breakdown: TP / FP / FN stacked bar ──────────────────────
    tps = [per_entity_metrics[e]["tp"] for e in entity_names]
    fps = [per_entity_metrics[e]["fp"] for e in entity_names]
    fns = [per_entity_metrics[e]["fn"] for e in entity_names]

    axes[1].bar(entity_names, tps, label="True Positive ✅", color="#50C878", edgecolor="white")
    axes[1].bar(entity_names, fps, bottom=tps, label="False Positive ❌", color="#FF6B6B", edgecolor="white")
    axes[1].bar(
        entity_names, fns,
        bottom=[t + f for t, f in zip(tps, fps)],
        label="Missed (FN) ⚠️", color="#FFA500", edgecolor="white",
    )

    axes[1].set_xlabel("Entity Type")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Error Breakdown by Entity Type", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Manual Validation Results ({len(annotations)} reports)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("manual_validation_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("📊 Validation results saved to manual_validation_results.png")

# %% [markdown]
# ## 9. Save Final Validation Report

# %%
if annotations and per_entity_metrics:
    report = {
        "metadata": {
            "model": MODEL_NAME,
            "num_reports_validated": len(annotations),
            "total_reports_in_dataset": len(df),
            "validation_date": datetime.now().isoformat(),
        },
        "overall_metrics": {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "true_positives": overall_tp,
            "false_positives": overall_fp,
            "false_negatives": overall_fn,
        },
        "per_entity_metrics": per_entity_metrics,
        "quality_rating": (
            "EXCELLENT" if overall_f1 >= 0.80
            else "GOOD" if overall_f1 >= 0.60
            else "FAIR" if overall_f1 >= 0.40
            else "POOR"
        ),
    }

    with open("validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"💾 Validation report saved to validation_report.json")

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  📋 VALIDATION SUMMARY")
    print(f"{'═' * 70}")
    print(f"  Reports validated: {len(annotations)}/{len(df)}")
    print(f"  Overall Precision: {overall_precision:.3f}")
    print(f"  Overall Recall:    {overall_recall:.3f}")
    print(f"  Overall F1:        {overall_f1:.3f}")
    print(f"  Quality Rating:    {report['quality_rating']}")
    print(f"{'═' * 70}")
