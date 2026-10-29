from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch
import io
from contextlib import redirect_stdout

import evaluate_system


class EvaluateSystemTests(unittest.TestCase):
    def test_unlabelled_evaluation_stops_before_pipeline_loading(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "ground_truth.json"
            manifest.write_text(json.dumps({"samples": [{"image": "sample.png"}]}), encoding="utf-8")
            with patch("sys.argv", ["evaluate_system.py", "--test-dir", directory]), \
                    patch.object(evaluate_system, "run_pipeline") as pipeline, \
                    redirect_stdout(io.StringIO()) as output:
                self.assertEqual(evaluate_system.main(), 2)
            pipeline.assert_not_called()
            self.assertIn("Models were not loaded", output.getvalue())

    def test_false_labels_count_as_annotations(self):
        readiness = evaluate_system.label_readiness([
            {"id": "negative", "reuse_label": False, "clip_match_label": False},
            {"id": "empty", "ocr_text": "  "},
        ])
        self.assertEqual(readiness["status"], "partially_labelled")
        self.assertEqual(readiness["metrics"]["image_reuse_retrieval"]["labelled_samples"], 1)
        self.assertEqual(readiness["metrics"]["ocr_character_error_rate"]["labelled_samples"], 0)

    def test_summary_separates_missing_predictions_and_labels(self):
        missing = {"id": "missing", "pipeline_status": "image_missing",
                   "ground_truth": {"language": "si", "reuse_label": True}}
        summary = evaluate_system.build_summary([missing], 0.3)
        self.assertEqual(summary["text_classification"]["label_coverage"], 1.0)
        self.assertEqual(summary["text_classification"]["status"], "predictions_unavailable")
        self.assertEqual(summary["clip_similarity_threshold"]["status"], "labels_required")

    def test_binary_f1_is_zero_when_all_positive_predictions_are_wrong(self):
        self.assertEqual(evaluate_system.binary_metrics([True], [False])["f1"], 0.0)
        self.assertEqual(evaluate_system.binary_metrics([False], [True])["f1"], 0.0)

    def test_ocr_cer_preserves_sinhala_unicode_and_normalizes_whitespace(self):
        distance, length, cer = evaluate_system.ocr_cer("ශ්‍රී  ලංකා", "ශ්‍රී ලංකා")
        self.assertEqual(distance, 0)
        self.assertEqual(length, len("ශ්‍රී ලංකා"))
        self.assertEqual(cer, 0.0)

    def test_binary_metrics_has_expected_confusion_matrix(self):
        metrics = evaluate_system.binary_metrics([True, True, False, False], [True, False, True, False])
        self.assertEqual(metrics["confusion_matrix"], {
            "true_positive": 1,
            "false_positive": 1,
            "true_negative": 1,
            "false_negative": 1,
        })
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)

    def test_load_ground_truth_rejects_paths_outside_test_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary_directory:
            test_dir = Path(temporary_directory) / "test_images"
            test_dir.mkdir()
            manifest = test_dir / "ground_truth.json"
            manifest.write_text(json.dumps({"samples": [{"image": "../outside.png"}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stay inside test_images"):
                evaluate_system.load_ground_truth(manifest, test_dir)

    def test_template_uses_only_images_inside_test_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary_directory:
            test_dir = Path(temporary_directory) / "test_images"
            nested_directory = test_dir / "sinhala"
            nested_directory.mkdir(parents=True)
            (nested_directory / "headline.png").write_bytes(b"placeholder")
            (test_dir / "ignore.txt").write_text("not an image", encoding="utf-8")
            manifest = test_dir / "ground_truth.json"

            count = evaluate_system.create_ground_truth_template(manifest, test_dir)
            template = json.loads(manifest.read_text(encoding="utf-8"))

            self.assertEqual(count, 1)
            self.assertEqual(template["samples"], [{"id": "sinhala-headline", "image": "sinhala/headline.png"}])

    def test_discover_source_images_excludes_processed_and_ui_outputs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "source.png").write_bytes(b"source")
            (root / "processed").mkdir()
            (root / "processed" / "preview.png").write_bytes(b"derived")
            (root / "ui").mkdir()
            (root / "ui" / "input.jpg").write_bytes(b"derived")

            self.assertEqual(
                [image.name for image in evaluate_system.discover_source_images(root)],
                ["source.png"],
            )

    def test_summary_calculates_requested_labelled_metrics(self):
        record = {
            "pipeline_status": "completed",
            "ground_truth": {
                "language": "si",
                "ocr_text": "ශ්‍රී ලංකා",
                "clip_match_label": True,
                "reuse_label": False,
                "authenticity_label": "real",
            },
            "ocr": {"status": "success", "language": "si", "extracted_text": "ශ්‍රී ලංකා"},
            "translation": {"translation_status": "unverified_translation_candidate"},
            "clip": {"status": "similarity_calculated", "similarity_score": 0.4},
            "authenticity": {"status": "likely_real_high_score", "predicted_class": "real"},
            "reuse": {"status": "no_close_local_match", "possible_reuse": False, "matched_reference": None},
        }

        summary = evaluate_system.build_summary([record], 0.3)
        self.assertEqual(summary["text_classification"]["macro_f1"], 1.0)
        self.assertEqual(summary["ocr_character_error_rate"]["corpus_cer"], 0.0)
        self.assertEqual(summary["clip_similarity_threshold"]["accuracy"], 1.0)
        self.assertEqual(summary["image_reuse_retrieval"]["accuracy"], 1.0)
        self.assertEqual(summary["ai_authenticity_detection"]["accuracy"], 1.0)
        self.assertEqual(summary["stage_status_counts"]["translation"], {"unverified_translation_candidate": 1})


if __name__ == "__main__":
    unittest.main()
