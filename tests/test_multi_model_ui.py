"""
test_multi_model_ui.py
======================
Tests for the multi-model display endpoints in server.py.

Uses Flask's test client and mocks out:
  - inference.load_models   (no-op — weights not needed for API tests)
  - inference.classify_occurrence (returns a deterministic stub)
  - inference.extract_entities    (returns a deterministic stub)
  - urllib.request.urlopen        (simulates external model responses)

Run with:
    python -m pytest tests/test_multi_model_ui.py -v
"""

import json
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import importlib

# Ensure app/ is importable
APP_DIR = os.path.join(os.path.dirname(__file__), '..', 'app')
sys.path.insert(0, os.path.abspath(APP_DIR))


# ── Stubs ─────────────────────────────────────────────────────────────────────

STUB_PREDICTION = {
    "top_class": "Mid-Air Collision",
    "confidence": 0.85,
    "top_5": [
        {"class": "MAC",   "confidence": 0.85},
        {"class": "CFIT",  "confidence": 0.07},
        {"class": "TURB",  "confidence": 0.04},
        {"class": "LOC-I", "confidence": 0.03},
        {"class": "OTHR",  "confidence": 0.01},
    ],
}

STUB_NER = {
    "ACTOR":   ["captain", "first officer"],
    "SYSTEM":  ["TCAS"],
    "TRIGGER": ["resolution advisory"],
    "spans":   [],
}

def make_external_response(slot):
    """Simulate a valid response from an external model endpoint."""
    return {
        "model_id":          f"team_{slot}_model",
        "display_name":      f"Team {slot} DistilBERT",
        "prediction":        STUB_PREDICTION,
        "inference_time_ms": 120 + slot * 10,
    }


# ── Test suite ────────────────────────────────────────────────────────────────

class TestMultiModelUI(unittest.TestCase):

    def _make_app(self, model2_url="", model3_url=""):
        """Import server fresh with given env vars, return test client."""
        # Patch env before import
        env_patch = patch.dict(os.environ, {
            "MODEL_2_URL": model2_url,
            "MODEL_3_URL": model3_url,
        })
        env_patch.start()
        self.addCleanup(env_patch.stop)

        # Mock heavy inference functions so no GPU/weights needed
        mock_load   = patch('inference.load_models',         return_value=None)
        mock_class  = patch('inference.classify_occurrence', return_value=STUB_PREDICTION)
        mock_ner    = patch('inference.extract_entities',    return_value=STUB_NER)
        mock_load.start();  self.addCleanup(mock_load.stop)
        mock_class.start(); self.addCleanup(mock_class.stop)
        mock_ner.start();   self.addCleanup(mock_ner.stop)

        # Re-import server so module-level env reads pick up patches
        import server as srv_module
        importlib.reload(srv_module)
        srv_module.load_models()  # no-op via mock

        return srv_module.app.test_client(), srv_module

    # ── /api/config ───────────────────────────────────────────────────────────

    def test_config_no_external_urls(self):
        client, _ = self._make_app()
        resp = client.get('/api/config')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        models = {m['slot']: m for m in data['models']}
        self.assertTrue(models[1]['local'])
        self.assertIsNone(models[2]['url'])
        self.assertIsNone(models[3]['url'])

    def test_config_with_external_urls(self):
        client, _ = self._make_app(
            model2_url="https://model2.example.com",
            model3_url="https://model3.example.com",
        )
        resp = client.get('/api/config')
        data = resp.get_json()
        models = {m['slot']: m for m in data['models']}
        self.assertEqual(models[2]['url'], "https://model2.example.com")
        self.assertEqual(models[3]['url'], "https://model3.example.com")

    # ── /api/multi-predict — no external models configured ────────────────────

    def test_multi_predict_local_only(self):
        client, _ = self._make_app()
        resp = client.post('/api/multi-predict',
                           json={"narrative": "The aircraft experienced turbulence."})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Slot 1: local result present
        self.assertIn('1', data)
        self.assertEqual(data['1']['model_id'], 'safeaerobert_damage_classifier')
        self.assertIn('prediction', data['1'])
        self.assertEqual(data['1']['prediction']['top_class'], 'Mid-Air Collision')

        # Slots 2 & 3: not configured
        self.assertEqual(data['2']['error'], 'not_configured')
        self.assertEqual(data['3']['error'], 'not_configured')

    def test_multi_predict_missing_narrative(self):
        client, _ = self._make_app()
        resp = client.post('/api/multi-predict', json={})
        self.assertEqual(resp.status_code, 400)
        self.assertIn('error', resp.get_json())

    # ── /api/multi-predict — with external models ─────────────────────────────

    def _mock_urlopen(self, slot_responses):
        """
        Returns a context manager that patches urllib.request.urlopen.
        slot_responses: dict mapping URL substring -> response dict
        """
        def fake_urlopen(req, timeout=30):
            url = req.full_url
            for key, payload in slot_responses.items():
                if key in url:
                    mock_resp = MagicMock()
                    mock_resp.read.return_value = json.dumps(payload).encode()
                    mock_resp.__enter__ = lambda s: mock_resp
                    mock_resp.__exit__  = MagicMock(return_value=False)
                    return mock_resp
            raise Exception(f"Unexpected URL in test: {url}")
        return patch('server.urllib.request.urlopen', side_effect=fake_urlopen)

    def test_multi_predict_with_both_external_models(self):
        client, _ = self._make_app(
            model2_url="https://model2.example.com",
            model3_url="https://model3.example.com",
        )
        ext_responses = {
            "model2.example.com": make_external_response(2),
            "model3.example.com": make_external_response(3),
        }
        with self._mock_urlopen(ext_responses):
            resp = client.post('/api/multi-predict',
                               json={"narrative": "The aircraft experienced turbulence."})

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Slot 1 — local
        self.assertEqual(data['1']['model_id'], 'safeaerobert_damage_classifier')

        # Slot 2 — external
        self.assertNotIn('error', data['2'])
        self.assertEqual(data['2']['model_id'], 'team_2_model')
        self.assertEqual(data['2']['display_name'], 'Team 2 DistilBERT')
        self.assertEqual(data['2']['prediction']['top_class'], 'Mid-Air Collision')
        self.assertIsInstance(data['2']['inference_time_ms'], int)

        # Slot 3 — external
        self.assertNotIn('error', data['3'])
        self.assertEqual(data['3']['model_id'], 'team_3_model')
        self.assertIn('prediction', data['3'])
        top5 = data['3']['prediction']['top_5']
        self.assertEqual(len(top5), 5)
        self.assertEqual(top5[0]['class'], 'MAC')

    def test_multi_predict_external_model_error(self):
        """If one external model is down, the other and model 1 still return results."""
        client, _ = self._make_app(
            model2_url="https://model2.example.com",
            model3_url="https://model3.example.com",
        )

        def fake_urlopen(req, timeout=30):
            url = req.full_url
            if "model2" in url:
                import urllib.error
                raise urllib.error.URLError("Connection refused")
            # model3 succeeds
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(make_external_response(3)).encode()
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__  = MagicMock(return_value=False)
            return mock_resp

        with patch('server.urllib.request.urlopen', side_effect=fake_urlopen):
            resp = client.post('/api/multi-predict',
                               json={"narrative": "The aircraft experienced turbulence."})

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('prediction', data['1'])          # model 1 OK
        self.assertIn('error',      data['2'])          # model 2 failed
        self.assertNotIn('error',   data['3'])          # model 3 OK

    # ── /predict (standalone API contract) ────────────────────────────────────

    def test_standalone_predict_contract(self):
        client, _ = self._make_app()
        resp = client.post('/predict', json={"narrative": "Engine failure on climb-out."})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['model_id'], 'safeaerobert_damage_classifier')
        self.assertIn('display_name', data)
        self.assertIn('prediction',   data)
        self.assertIn('top_class',    data['prediction'])
        self.assertIn('confidence',   data['prediction'])
        self.assertIn('top_5',        data['prediction'])
        self.assertIsInstance(data['inference_time_ms'], int)

    def test_standalone_predict_missing_narrative(self):
        client, _ = self._make_app()
        resp = client.post('/predict', json={"narrative": ""})
        self.assertEqual(resp.status_code, 400)

    # ── /api/pipeline ─────────────────────────────────────────────────────────

    def test_pipeline_returns_ner_and_classification(self):
        client, _ = self._make_app()
        resp = client.post('/api/pipeline', json={"text": "TCAS RA issued during approach."})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn('ner',            data)
        self.assertIn('classification', data)
        self.assertIn('ACTOR',   data['ner'])
        self.assertIn('SYSTEM',  data['ner'])
        self.assertIn('TRIGGER', data['ner'])
        self.assertIn('top_class', data['classification'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
