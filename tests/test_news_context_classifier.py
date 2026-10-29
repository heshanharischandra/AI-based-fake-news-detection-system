import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.services.news_context_classifier import pair_features, predict_context
from app.services.scoring_service import calculate_verification_score


class ContextClassifierTests(unittest.TestCase):
    def test_features_are_scale_invariant(self):
        image = np.arange(1, 513, dtype=np.float32)
        text = image[::-1].copy()
        np.testing.assert_allclose(pair_features(image, text), pair_features(image * 3, text * 5), atol=1e-6)

    def test_invalid_embeddings_are_rejected(self):
        for vector in (np.zeros(512), np.full(512, np.nan), np.ones(511)):
            with self.assertRaises(ValueError):
                pair_features(vector, np.ones(512))

    def test_artifact_inference_and_unknown_schema(self):
        artifact = {"schema_version": 1, "encoder": "openai/clip-vit-base-patch32",
                    "feature_version": "normalized_product_absdiff_cosine_v1", "text_mode": "raw_caption_no_prompt",
                    "coef": [0.0] * 1025, "intercept": 2.0, "threshold": 0.6,
                    "dataset": "test fixture", "calibration_status": "not_probability_calibrated"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            result = predict_context(np.ones(512), np.ones(512), path)
            self.assertEqual(result["prediction"], "possible_out_of_context")
            self.assertAlmostEqual(result["mismatch_score"], 0.8808)
            self.assertFalse(result["verified_evidence"])
            artifact["schema_version"] = 99
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError):
                predict_context(np.ones(512), np.ones(512), path)

    def test_missing_model_and_high_scores_do_not_create_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = predict_context(np.ones(512), np.ones(512), Path(directory) / "absent.json")
            self.assertEqual(missing["status"], "context_model_not_trained")
        model_output = {"status": "dataset_model_prediction", "mismatch_score": 0.9999}
        result = calculate_verification_score(None, {"trained_context": model_output})
        self.assertIsNone(result["verification_score"])
        self.assertEqual(result["trained_context"], model_output)
        self.assertFalse(result["source_eligible"])


if __name__ == "__main__":
    unittest.main()
