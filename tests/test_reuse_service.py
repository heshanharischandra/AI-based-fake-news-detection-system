"""Local matching regressions; these are not a news-image accuracy benchmark."""
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from app.services import reuse_service as service


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.references = root / "references"
        self.references.mkdir()
        self.query = root / "query.png"
        self.photo = Image.new("RGB", (640, 480), "white")
        draw = ImageDraw.Draw(self.photo)
        rng = random.Random(42)
        for _ in range(180):
            x, y = rng.randrange(580), rng.randrange(420)
            color = tuple(rng.randrange(256) for _ in range(3))
            draw.ellipse((x, y, x + rng.randrange(8, 60), y + rng.randrange(8, 60)), fill=color)
        self.photo.save(self.query)
        folder_patch = patch.object(service, "REFERENCE_DIR", self.references)
        folder_patch.start()
        self.addCleanup(folder_patch.stop)

    @staticmethod
    def hashes(distance, votes):
        return {
            "hash_distances": {"phash": round(distance * 256)},
            "normalized_hash_distances": {"phash": distance},
            "hash_similarity_score": 1 - distance,
            "weighted_hash_distance": distance,
            "hash_match_votes": votes,
        }

    @staticmethod
    def features(inliers=0, ratio=0.0):
        return {"feature_status": "feature_match_calculated", "feature_good_matches": 100,
                "feature_inliers": inliers, "feature_inlier_ratio": ratio}

    def test_exact_pixels_include_dimensions_and_are_not_downsampled(self):
        self.assertNotEqual(service._pixel_digest(Image.new("RGB", (16, 32), "white")),
                            service._pixel_digest(Image.new("RGB", (32, 16), "white")))
        self.assertNotEqual(service._pixel_digest(Image.new("RGB", (1024, 1024), "white")),
                            service._pixel_digest(Image.new("RGB", (2048, 2048), "white")))

    def test_exact_copy_does_not_depend_on_opencv(self):
        self.photo.save(self.references / "copy.png")
        with patch.object(service, "_feature_match", side_effect=RuntimeError("unavailable")) as matcher:
            result = service.check_image_reuse(self.query)
        self.assertEqual(result["status"], "exact_duplicate")
        self.assertEqual(result["matched_reference"], "copy.png")
        matcher.assert_not_called()

    def test_lossless_reencoding_is_pixel_exact_not_file_exact(self):
        self.photo.save(self.references / "copy.bmp")
        result = service.check_image_reuse(self.query)
        self.assertEqual(result["status"], "exact_duplicate")
        self.assertFalse(result["file_digest_match"])
        self.assertTrue(result["pixel_digest_match"])

    def test_unrelated_flat_images_are_not_confirmed_by_hashes(self):
        Image.new("RGB", (128, 128), "white").save(self.query)
        Image.new("RGB", (128, 128), (160, 160, 160)).save(self.references / "gray.png")
        result = service.check_image_reuse(self.query)
        self.assertIsNot(result["possible_reuse"], True)
        self.assertIsNone(result["matched_reference"])

    def test_valid_geometric_match_wins_over_many_inconsistent_inliers(self):
        Image.new("RGB", (128, 128), "red").save(self.references / "a_bad.png")
        Image.new("RGB", (128, 128), "blue").save(self.references / "b_good.png")
        with patch.object(service, "_hash_comparison", side_effect=[self.hashes(0.07, 4), self.hashes(0.30, 1)]), \
                patch.object(service, "_feature_match", side_effect=[self.features(100, 0.01), self.features(20, 0.8)]):
            result = service.check_image_reuse(self.query)
        self.assertTrue(result["possible_reuse"])
        self.assertEqual(result["matched_reference"], "b_good.png")

    def test_feature_failure_does_not_turn_into_a_negative_result(self):
        Image.new("RGB", (128, 128), "red").save(self.references / "reference.png")
        with patch.object(service, "_hash_comparison", return_value=self.hashes(0.45, 0)), \
                patch.object(service, "_feature_match", side_effect=RuntimeError("broken feature backend")):
            result = service.check_image_reuse(self.query)
        self.assertEqual(result["status"], "reference_comparison_incomplete")
        self.assertIsNone(result["possible_reuse"])
        self.assertEqual(result["reference_count"], 1)
        self.assertEqual(len(result["failed_references"]), 1)

    def test_bad_reference_is_reported_without_losing_valid_match(self):
        (self.references / "bad.png").write_bytes(b"not an image")
        self.photo.save(self.references / "good.png")
        result = service.check_image_reuse(self.query)
        self.assertEqual(result["status"], "exact_duplicate")
        self.assertEqual(result["reference_count"], 1)
        self.assertEqual(result["reference_collection_count"], 2)
        self.assertEqual(len(result["failed_references"]), 1)

    def test_failed_or_empty_collection_returns_unknown_not_no_reuse(self):
        self.assertEqual(service.check_image_reuse(self.query)["status"], "reference_collection_empty")
        (self.references / "bad.png").write_bytes(b"not an image")
        result = service.check_image_reuse(self.query)
        self.assertEqual(result["status"], "reference_comparison_failed")
        self.assertIsNone(result["possible_reuse"])

    def test_nonmatch_is_not_published_as_a_matched_reference(self):
        Image.new("RGB", (128, 128), "red").save(self.references / "other.png")
        with patch.object(service, "_hash_comparison", return_value=self.hashes(0.45, 0)), \
                patch.object(service, "_feature_match", return_value=self.features()):
            result = service.check_image_reuse(self.query)
        self.assertFalse(result["possible_reuse"])
        self.assertIsNone(result["matched_reference"])
        self.assertEqual(result["nearest_reference_candidate"], "other.png")

    def test_cropped_and_recompressed_images_with_real_feature_matching(self):
        self.photo.save(self.references / "original.png")
        variants = [self.photo.crop((80, 50, 560, 420)), self.photo.resize((480, 360))]
        for index, variant in enumerate(variants):
            with self.subTest(index=index):
                transformed = self.query.with_suffix(".jpg")
                variant.save(transformed, quality=70)
                result = service.check_image_reuse(transformed)
                self.assertTrue(result["possible_reuse"], result)
                self.assertEqual(result["matched_reference"], "original.png")
                self.assertFalse(result["pixel_digest_match"])


if __name__ == "__main__":
    unittest.main()
