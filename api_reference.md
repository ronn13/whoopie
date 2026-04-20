# SafeAeroBERT Demo — `/predict` API Reference

This document defines the **exact JSON contract** your model's `/predict` endpoint must satisfy to display correctly in the multi-model demo UI. Model 1 (SafeAeroBERT) is the reference implementation.

---

## Endpoint

```
POST /predict
Content-Type: application/json
```

---

## Request

```json
{
  "narrative": "string — the full incident narrative text"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `narrative` | string | yes | Raw text. Max 512 tokens recommended. Truncation is your model's responsibility. |

---

## Response — Required Shape

```json
{
  "model_id":        "string",
  "display_name":    "string",
  "prediction": {
    "top_class":   "string",
    "confidence":  0.0000,
    "top_5": [
      { "class": "string", "confidence": 0.0000 },
      { "class": "string", "confidence": 0.0000 },
      { "class": "string", "confidence": 0.0000 },
      { "class": "string", "confidence": 0.0000 },
      { "class": "string", "confidence": 0.0000 }
    ]
  },
  "inference_time_ms": 0
}
```

---

## Field-by-Field Spec

### Top-level

| Field | Type | Example | Notes |
|---|---|---|---|
| `model_id` | string | `"exp12_ensemble"` | Unique snake_case identifier for your model. |
| `display_name` | string | `"EXP-12 Ensemble (RoBERTa + Longformer)"` | Shown as the panel title in the UI. |
| `prediction` | object | — | See below. |
| `inference_time_ms` | integer | `143` | Wall-clock time for the forward pass in milliseconds. Shown in the UI footer. |

---

### `prediction` object

| Field | Type | Example | Notes |
|---|---|---|---|
| `top_class` | string | `"Controlled Flight Into or Toward Terrain"` | **Full human-readable ADREP description** — shown as the large label under the category badge. Must be one of the 10 canonical descriptions in the table below. |
| `confidence` | float | `0.9421` | Softmax probability of the top class. Range `[0.0, 1.0]`, 4 decimal places. |
| `top_5` | array | — | Exactly 5 entries, sorted **descending** by confidence. Each entry: `{ "class": <ICAO code>, "confidence": <float> }`. |

### `top_5` entry

| Field | Type | Example | Notes |
|---|---|---|---|
| `class` | string | `"CFIT"` | **ICAO/ADREP short code** — used for the colour-coded category badge and bar chart labels. Must be one of the 10 canonical codes in the table below. |
| `confidence` | float | `0.9421` | Softmax probability for this class. |

> **Critical distinction:**
> - `prediction.top_class` → **full description** (e.g. `"Controlled Flight Into or Toward Terrain"`)
> - `top_5[i].class` → **ICAO short code** (e.g. `"CFIT"`)
>
> Getting these swapped is the most common integration mistake. If both fields contain only the ICAO code, the description line in the UI will be blank/wrong.

---

## Canonical Label Table

All 14 accepted classes. Your model must use **exactly** these strings — casing and punctuation included.

| ICAO Code (`top_5[i].class`) | Full Description (`prediction.top_class`) | UI Badge Colour |
|---|---|---|
| `MAC`    | `Mid-Air Collision` | red `#f87171` |
| `CFIT`   | `Controlled Flight Into or Toward Terrain` | orange `#f97316` |
| `GCOL`   | `Ground Collision` | yellow `#facc15` |
| `SEC`    | `Security Related` | purple `#a78bfa` |
| `ATM`    | `ATM/Communication or Ground Issue` | sky `#38bdf8` |
| `LOC-I`  | `Loss of Control - Inflight` | red `#ef4444` |
| `TURB`   | `Turbulence Encounter` | green `#4ade80` |
| `RE`     | `Runway Excursion` | rose `#f43f5e` |
| `USOS`   | `Undershoot/Overshoot` | amber `#fb923c` |
| `RI`     | `Runway Incursion` | fuchsia `#e879f9` |
| `SCF-NP` | `System/Component Failure or Malfunction (Non-Powerplant)` | cyan `#67e8f9` |
| `RAMP`   | `Ground Handling / Ramp Occurrence` | yellow `#fde68a` |
| `OTHR`   | `Other` | slate `#94a3b8` |
| `UNK`    | `Other` | slate `#94a3b8` |

> `OTHR` and `UNK` are treated identically — both mean the model could not confidently assign a specific category. Use `OTHR` as the preferred code; `UNK` is accepted for models whose label set includes it.

---

## Complete Example Response

```json
{
  "model_id": "exp12_ensemble",
  "display_name": "EXP-12 Ensemble (RoBERTa + Longformer)",
  "prediction": {
    "top_class": "Controlled Flight Into or Toward Terrain",
    "confidence": 0.9421,
    "top_5": [
      { "class": "CFIT",  "confidence": 0.9421 },
      { "class": "LOC-I", "confidence": 0.0312 },
      { "class": "TURB",  "confidence": 0.0118 },
      { "class": "ATM",   "confidence": 0.0089 },
      { "class": "OTHR",  "confidence": 0.0060 }
    ]
  },
  "inference_time_ms": 143
}
```

---

## Error Response

If your endpoint cannot process the request, return HTTP 4xx/5xx with:

```json
{
  "error": "human-readable description of what went wrong"
}
```

The UI detects the `error` key and renders it as a red banner in your model's panel instead of crashing.

---

## Minimal Flask Implementation Template

```python
from flask import Flask, request, jsonify
import time

app = Flask(__name__)

LABEL_FULL = {
    "MAC":   "Mid-Air Collision",
    "CFIT":  "Controlled Flight Into or Toward Terrain",
    "GCOL":  "Ground Collision",
    "SEC":   "Security Related",
    "ATM":   "ATM/Communication or Ground Issue",
    "LOC-I": "Loss of Control - Inflight",
    "TURB":  "Turbulence Encounter",
    "RE":    "Runway Excursion",
    "USOS":  "Undershoot/Overshoot",
    "OTHR":  "Other",
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    narrative = (data.get("narrative") or "").strip()
    if not narrative:
        return jsonify({"error": "No narrative provided"}), 400

    t0 = time.time()

    # --- run your model here ---
    # probs: list of float, one per class, in label order
    # labels: list of ICAO codes in the same order as probs
    probs, labels = run_my_model(narrative)

    paired = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
    top_code, top_conf = paired[0]

    return jsonify({
        "model_id":     "your_model_id",
        "display_name": "Your Model Display Name",
        "prediction": {
            "top_class":  LABEL_FULL[top_code],          # full description
            "confidence": round(top_conf, 4),
            "top_5": [
                {"class": code, "confidence": round(conf, 4)}
                for code, conf in paired[:5]
            ],                                            # ICAO codes, not full names
        },
        "inference_time_ms": int((time.time() - t0) * 1000),
    })
```

---

## Checklist Before Connecting

- [ ] `POST /predict` returns HTTP 200 with `Content-Type: application/json`
- [ ] `prediction.top_class` is the **full description** string (not the ICAO code)
- [ ] `top_5[i].class` is the **ICAO code** (not the full description)
- [ ] `top_5` has exactly 5 entries, sorted descending by confidence
- [ ] All confidences sum to ≤ 1.0 (they are softmax probabilities)
- [ ] `inference_time_ms` is an integer (milliseconds)
- [ ] Invalid/empty narrative returns `{ "error": "..." }` with a 4xx status
- [ ] CORS is enabled if the demo UI and your server run on different origins:
  ```python
  from flask_cors import CORS
  CORS(app)
  ```
- [ ] Your label set matches the 10 canonical codes exactly (no extras, no renames)
