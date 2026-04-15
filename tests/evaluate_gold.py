import json
import os

def normalize_entities(entities):
    """Lowercases and cleans up entities for comparison."""
    if not entities:
        return set()
    cleaned = set()
    for e in entities:
        if e.lower().strip() not in ["unknown", "not identified", "not explicitly stated"]:
            cleaned.add(e.lower().strip())
    return cleaned

def evaluate():
    gold_path = os.path.join("tests", "output", "gold_100_labels.json")
    pred_path = os.path.join("tests", "output", "extracted_events_asn.json")

    print("📊 Loading Gold Benchmark...")
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    print("🤖 Loading SafeAeroBERT Predictions (this may take a second)...")
    if not os.path.exists(pred_path):
        print(f"Waiting for {pred_path} to finish generating...")
        return
        
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    # Convert predictions list of dicts into a dictionary keyed by uid
    predictions = {p["uid"]: p for p in pred_data if "uid" in p}

    metrics = {"ACTOR": {"tp": 0, "fp": 0, "fn": 0},
               "SYSTEM": {"tp": 0, "fp": 0, "fn": 0},
               "TRIGGER": {"tp": 0, "fp": 0, "fn": 0}}

    evaluated_count = 0

    for uid, gold_entities in gold_data.items():
        if uid not in predictions:
            continue
            
        pred_entities = predictions[uid]
        evaluated_count += 1

        for label in ["ACTOR", "SYSTEM", "TRIGGER"]:
            g_set = normalize_entities(gold_entities.get(label, []))
            p_set = normalize_entities(pred_entities.get(label, []))

            # Since exact wording might differ slightly (e.g., "the pilot" vs "pilot"), 
            # we use a soft-match technique to count hits.
            for p in p_set:
                # If prediction is an exact match or substring of a gold label
                # or gold label is a substring of prediction (relaxed matching).
                if any(p in g or g in p for g in g_set):
                    metrics[label]["tp"] += 1
                else:
                    metrics[label]["fp"] += 1
            
            for g in g_set:
                if not any(p in g or g in p for p in p_set):
                    metrics[label]["fn"] += 1

    print(f"\n✅ Evaluated {evaluated_count} matching records from the Gold Standard.\n")
    print("-" * 60)
    print(f"{'CATEGORY':<12} | {'PRECISION':<12} | {'RECALL':<12} | {'F1-SCORE':<12}")
    print("-" * 60)

    total_tp, total_fp, total_fn = 0, 0, 0

    for label in ["ACTOR", "SYSTEM", "TRIGGER"]:
        tp = metrics[label]["tp"]
        fp = metrics[label]["fp"]
        fn = metrics[label]["fn"]
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{label:<12} | {precision:.4f}{'':<6} | {recall:.4f}{'':<6} | {f1:.4f}")

    # Macro Averages
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_p * overall_r) / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0

    print("-" * 60)
    print(f"{'OVERALL':<12} | {overall_p:.4f}{'':<6} | {overall_r:.4f}{'':<6} | {overall_f1:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    evaluate()
