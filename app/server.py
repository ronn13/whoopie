"""
server.py — Flask backend for the SafeAeroBERT Demo UI.

Endpoints:
  GET  /                    → serves index.html
  POST /api/extract         → Layer-1 NER  {text} → entities + spans
  POST /api/classify        → Layer-3 classifier {text} → severity + probs
  POST /api/pipeline        → runs both layers and returns combined result
  GET  /api/config          → returns external model endpoint URLs
  POST /api/multi-predict   → fans out {narrative} to all 3 models concurrently
"""

import os
import time
import concurrent.futures

import urllib.request
import urllib.error
import json as _json

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from inference import load_models, extract_entities, classify_occurrence

app = Flask(__name__, static_folder="static")
CORS(app)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# External model URLs — set via environment variables before starting the server.
# Leave blank to show the model as "not configured" in the UI.
MODEL_2_URL = os.environ.get("MODEL_2_URL", "").rstrip("/")
MODEL_3_URL = os.environ.get("MODEL_3_URL", "").rstrip("/")


def _call_external(url, narrative, timeout=30):
    """POST {narrative} to url/predict and return the JSON response dict.
    Returns an error dict on failure."""
    endpoint = url.rstrip("/") + "/predict"
    payload = _json.dumps({"narrative": narrative}).encode()
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return {"error": f"HTTP {e.code}: {body[:200]}"}
    except urllib.error.URLError as e:
        return {"error": f"Connection error: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


# ── Static serving ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


# ── API: Layer-1 entity extraction ──────────────────────────────────────────
@app.route("/api/extract", methods=["POST"])
def api_extract():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = extract_entities(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Layer-3 classification ─────────────────────────────────────────────
@app.route("/api/classify", methods=["POST"])
def api_classify():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = classify_occurrence(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── API: Standalone predict (API contract — for other UIs to call this model) ──
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = (data.get("narrative") or "").strip()
    if not text:
        return jsonify({"error": "No narrative provided"}), 400
    try:
        t0 = time.time()
        result = classify_occurrence(text)
        return jsonify({
            "model_id": "safeaerobert_damage_classifier",
            "display_name": "SafeAeroBERT Damage Classifier",
            "prediction": result,
            "inference_time_ms": int((time.time() - t0) * 1000),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── API: Config (external model URLs for the frontend) ────────────────────────
@app.route("/api/config", methods=["GET"])
def api_config():
    return jsonify({
        "models": [
            {
                "slot": 1,
                "model_id": "safeaerobert_damage_classifier",
                "display_name": "SafeAeroBERT",
                "local": True,
                "url": None,
            },
            {
                "slot": 2,
                "model_id": None,
                "display_name": "Model 2",
                "local": False,
                "url": MODEL_2_URL or None,
            },
            {
                "slot": 3,
                "model_id": None,
                "display_name": "Model 3",
                "local": False,
                "url": MODEL_3_URL or None,
            },
        ]
    })


# ── API: Multi-predict (fan-out to all 3 models concurrently) ─────────────────
@app.route("/api/multi-predict", methods=["POST"])
def api_multi_predict():
    data = request.get_json(force=True)
    narrative = (data.get("narrative") or "").strip()
    if not narrative:
        return jsonify({"error": "No narrative provided"}), 400

    results = {}

    # Model 1 — local
    try:
        t0 = time.time()
        pred = classify_occurrence(narrative)
        results["1"] = {
            "model_id": "safeaerobert_damage_classifier",
            "display_name": "SafeAeroBERT",
            "prediction": pred,
            "inference_time_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        results["1"] = {"error": str(e)}

    # Models 2 & 3 — external, called concurrently
    external = {}
    if MODEL_2_URL:
        external["2"] = MODEL_2_URL
    if MODEL_3_URL:
        external["3"] = MODEL_3_URL

    if external:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                slot: pool.submit(_call_external, url, narrative)
                for slot, url in external.items()
            }
            for slot, fut in futures.items():
                try:
                    results[slot] = fut.result()
                except Exception as e:
                    results[slot] = {"error": str(e)}

    # Mark unconfigured slots explicitly
    for slot in ("2", "3"):
        if slot not in results:
            results[slot] = {"error": "not_configured"}

    return jsonify(results)



# ── API: Full pipeline ────────────────────────────────────────────────────────
@app.route("/api/pipeline", methods=["POST"])
def api_pipeline():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        ner    = extract_entities(text)
        damage = classify_occurrence(text)
        return jsonify({"ner": ner, "classification": damage})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Startup ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
