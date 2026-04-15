# Adding Two More Models to the SafeAeroBERT Demo UI

## First question: What does each teammate need to send you?

### The Short Answer
Each teammate needs to send you **4 things** per model.

---

## 1. What They Need to Send (per model)

### A. The trained weights file
```
their_model_weights.pt       ← PyTorch state_dict
```
Generated with:
```python
torch.save(model.state_dict(), "their_model_weights.pt")
```

> [!IMPORTANT]
> Make sure they save only `model.state_dict()`, **not** the whole model object (`torch.save(model, ...)`). The whole-object format is fragile across PyTorch versions, and your server likely has a different environment than theirs.

---

### B. A config JSON file
This is the most important thing to standardize. Tell them to fill in this template exactly:

```json
{
  "model_id": "model_b",
  "display_name": "DistilBERT Classifier",
  "base_model": "distilbert-base-uncased",
  "task": "sequence_classification",
  "max_len": 256,
  "num_labels": 6,
  "labels": ["w/o", "unk", "sub", "mis", "non", "min"],
  "label_order_matches_logits": true,
  "description": "DistilBERT fine-tuned on aviation narratives with focal loss",
  "training": {
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 3e-5,
    "optimizer": "AdamW"
  },
  "metrics": {
    "weighted_accuracy": 0.85,
    "macro_f1": 0.68
  }
}
```

> [!CAUTION]
> **`labels` order is critical.** The label at index `i` must correspond to logit index `i` exactly as it was during their training. A mismatch will silently produce wrong predictions — no error will be thrown. Ask them to double-check by printing `model.config.id2label` from their training environment.

---

### C. The base HuggingFace model name (or local tokenizer files)
If they used a standard HF model (e.g., `bert-base-uncased`, `distilbert-base-uncased`, `roberta-base`), the string name is enough — the tokenizer loads from the hub.

If they fine-tuned a **custom/private tokenizer**, they need to send the full tokenizer folder:
```
tokenizer/
  ├── tokenizer_config.json
  ├── vocab.txt  (or vocab.json + merges.txt for BPE)
  └── special_tokens_map.json
```

---

### D. A short validation snippet (optional but recommended)
Ask them to paste the inference block they used locally so you can verify the weights load correctly on your end:

```python
model.eval()
with torch.no_grad():
    logits = model(input_ids=..., attention_mask=...).logits
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax().item()
print(id2label[pred])  # Should print a valid category
```

---

## 2. How the UI Changes

### Layout concept: Side-by-side comparison columns

The current single "Layer 3 — Damage Severity" card becomes **3 columns**, one per model. The narrative input and Layer 1 NER panel stay unchanged.

```
┌─────────────────────────────────────────────────────┐
│              Incident Narrative (input)             │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│        Layer 1 — Entity Extraction (shared)         │
└─────────────────────────────────────────────────────┘
┌───────────────┬───────────────┬─────────────────────┐
│  SafeAeroBERT │  Model B      │  Model C            │
│  (yours)      │  (teammate)   │  (teammate)         │
│               │               │                     │
│  sub  91.1%  │  sub  87.3%  │  w/o  54.2%         │
│  ▓▓▓▓▓▓▓     │  ▓▓▓▓▓▓      │  ▓▓▓                │
│  ...bars...   │  ...bars...   │  ...bars...         │
└───────────────┴───────────────┴─────────────────────┘
│         ✦ All 3 agree: Substantial Damage           │
└─────────────────────────────────────────────────────┘
```

Key UI additions:
- **Consensus badge** — shows when all 3 models agree, or flags disagreement
- **Winner highlight** — the column with highest confidence gets a subtle border highlight
- **Comparison table** — collapsible row below showing Accuracy / F1 side-by-side

---

## 3. Things to Keep in Mind

### ⚠️ Label alignment is the #1 silent failure risk
If your model was trained with labels `["w/o", "unk", "sub", ...]` and their model used `["sub", "w/o", "non", ...]`, their model will appear to run fine but predict completely wrong categories. **Enforce the config JSON above and cross-check it.**

### ⚠️ Startup memory / VRAM
Three 400MB BERT-class models loaded simultaneously = **~1.2–1.5 GB RAM** on CPU, or VRAM if GPU. If your machine is tight, consider **lazy loading** (load a model only when its column is requested) rather than all-at-startup.

### 🔄 Base model tokenizers must match the weights
If teammate used `roberta-base` but you try loading their weights with a `bert-base-uncased` tokenizer, it will fail or produce garbage. The base model name in the config is used to load the tokenizer — they must match exactly.

### 📊 Normalize your comparison metrics
Ask both teammates to compute accuracy and macro F1 on the **same held-out test set** (your 4,658-record set) so the numbers in the comparison table are apples-to-apples. Numbers from their own private split are not comparable.

### 🏷 Label set must be identical
All three models must predict the **same 6 categories**. If a teammate's model has 5 labels (e.g., they dropped `mis` due to class imbalance), the comparison UI will be asymmetric. Decide in advance: either all 6 labels or a reduced shared set.

### 🔌 Extensibility in the code
The current `inference.py` hardcodes two models. It will need to be refactored into a **model registry** — a list of model configs that the server iterates over. This makes adding a 4th model later trivial. See the suggested registry pattern below.

---

## 4. Suggested Code Structure After Refactor

```python
# inference.py — registry pattern
MODEL_REGISTRY = [
    {
        "id":           "safeaerobert",
        "display_name": "SafeAeroBERT",
        "config_path":  "outputs/models/safeaerobert_config.json",
        "weights_path": "outputs/models/safeaerobert_classifier.pt",
    },
    {
        "id":           "model_b",
        "display_name": "DistilBERT",
        "config_path":  "outputs/models/model_b_config.json",
        "weights_path": "outputs/models/model_b.pt",
    },
    {
        "id":           "model_c",
        "display_name": "RoBERTa",
        "config_path":  "outputs/models/model_c_config.json",
        "weights_path": "outputs/models/model_c.pt",
    },
]
```
The `/api/compare` endpoint then runs all registered models and returns an array of results — the frontend maps each result to a column.

---

## 5. Teammate Handoff Checklist

Send this to each teammate:

- [ ] `their_model.pt` — `torch.save(model.state_dict(), ...)`
- [ ] `their_model_config.json` — filled template above (especially `labels` order)
- [ ] Base HuggingFace model string (e.g. `"distilbert-base-uncased"`)
- [ ] Weighted accuracy + macro F1 on the **shared 4,658-record test set**
- [ ] Confirm PyTorch version: `torch.__version__` (for weights compatibility)
- [ ] Tokenizer folder if they used a custom/local tokenizer
