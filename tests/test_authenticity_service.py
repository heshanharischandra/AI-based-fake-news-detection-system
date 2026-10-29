import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from app.services import authenticity_service as service


class AuthenticityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image_path = Path(self.directory.name) / "sample.png"
        Image.new("RGB", (100, 80), "white").save(self.image_path)

    def analyze(self, scores, *, labels=None, overlays=False, raw_logits=None):
        labels = labels if labels is not None else {0: "REAL", 1: "FAKE"}
        tensors = raw_logits if raw_logits is not None else [
            torch.tensor([[1 - score, score]], dtype=torch.float32).log()
            for score in scores
        ]
        fake_model = Mock()
        fake_model.config = SimpleNamespace(id2label=labels, problem_type=None)
        fake_model.side_effect = [SimpleNamespace(logits=value) for value in tensors]
        fake_processor = Mock()
        fake_processor.return_value.to.return_value = {}
        with patch.object(service, "load_authenticity_model"), \
                patch.object(service, "processor", fake_processor), \
                patch.object(service, "model", fake_model):
            return service.check_ai_generation(
                self.image_path, input_contains_text_overlay=overlays
            )

    def test_high_average_does_not_hide_a_weak_view(self):
        # These averages and deviations passed the former high-score gates.
        for scores in ([0.99, 0.85, 0.75], [0.01, 0.15, 0.25]):
            with self.subTest(scores=scores):
                result = self.analyze(scores)
                self.assertGreater(result["model_score"], 0.85)
                self.assertLess(result["augmentation_disagreement"], 0.10)
                self.assertEqual(result["status"], "needs_manual_review")
                self.assertTrue(result["review_reasons"])

    def test_stable_predictions_and_reversed_label_order(self):
        for scores, labels, status in (
            ([0.95, 0.94, 0.96], None, "possible_ai_generated_high_score"),
            ([0.05, 0.06, 0.04], None, "likely_real_high_score"),
            ([0.05, 0.06, 0.04], {"0": "AI-generated", "1": "REAL"},
             "possible_ai_generated_high_score"),
        ):
            with self.subTest(status=status, labels=labels):
                result = self.analyze(scores, labels=labels)
                self.assertEqual(result["status"], status)
                self.assertAlmostEqual(result["ai_generated_score"] + result["real_score"], 1)
                self.assertFalse(result["requires_review"])

    def test_overlaid_screenshot_keeps_scores_but_requires_review(self):
        result = self.analyze([0.98, 0.98, 0.98], overlays=True)
        self.assertEqual(result["status"], "needs_manual_review")
        self.assertAlmostEqual(result["ai_generated_score"], 0.98)
        self.assertTrue(result["input_contains_text_overlay"])

    def test_unrecognized_or_ambiguous_labels_do_not_produce_scores(self):
        for label in ("PAINTING", "AI OR REAL", "NOT REAL", "LABEL_1"):
            with self.subTest(label=label):
                result = self.analyze([0.95] * 3, labels={0: "REAL", 1: label})
                self.assertEqual(result["status"], "analysis_unavailable")
                self.assertIsNone(result["ai_generated_score"])

    def test_invalid_logits_fail_without_a_classification(self):
        for tensor in (torch.tensor([[float("nan"), 1.0]]),
                       torch.tensor([[float("inf"), 1.0]]),
                       torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([1.0, 2.0])):
            with self.subTest(shape=tensor.shape):
                result = self.analyze([], raw_logits=[tensor])
                self.assertEqual(result["status"], "analysis_unavailable")
                self.assertIsNone(result["real_score"])

    def test_unreadable_image_does_not_load_model(self):
        with patch.object(service, "load_authenticity_model") as loader:
            result = service.check_ai_generation(self.image_path.with_name("absent.png"))
        self.assertEqual(result["status"], "analysis_unavailable")
        loader.assert_not_called()

    def test_transparency_and_orientation_preserve_the_source(self):
        Image.new("RGBA", (10, 20), (255, 0, 0, 0)).save(self.image_path)
        original_bytes = self.image_path.read_bytes()
        prepared = service._prepare_image(self.image_path)
        self.assertEqual(prepared.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(self.image_path.read_bytes(), original_bytes)
        rotated_path = self.image_path.with_suffix(".jpg")
        source = Image.new("RGB", (10, 20), "white")
        exif = source.getexif()
        exif[274] = 6
        source.save(rotated_path, exif=exif)
        self.assertEqual(service._prepare_image(rotated_path).size, (20, 10))


if __name__ == "__main__":
    unittest.main()
