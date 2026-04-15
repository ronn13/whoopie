"""
server.py — Flask backend for the SafeAeroBERT Demo UI.

Endpoints:
  GET  /                 → serves index.html
  POST /api/extract      → Layer-1 NER  {text} → entities + spans
  POST /api/classify     → Layer-3 classifier {text} → severity + probs
  POST /api/pipeline     → runs both layers and returns combined result
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from inference import load_models, extract_entities, classify_occurrence

app = Flask(__name__, static_folder="static")
CORS(app)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


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

# ── API: Standalone Model Integration ─────────────────────────────────────────
import time
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    data = request.get_json(force=True)
    
    # Contract uses 'narrative' instead of 'text'
    text = (data.get("narrative") or "").strip()
    if not text:
        return jsonify({"error": "No narrative provided"}), 400
        
    try:
        result = classify_occurrence(text)
        inference_time = int((time.time() - start_time) * 1000)
        
        return jsonify({
            "model_id": "safeaerobert_damage_classifier",
            "display_name": "SafeAeroBERT Damage Classifier",
            "prediction": result,
            "inference_time_ms": inference_time
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



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
    app.run(host="0.0.0.0", port=5050, debug=False)
