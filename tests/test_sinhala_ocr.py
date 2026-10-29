import json
import unittest
import unicodedata
from unittest.mock import patch

from PIL import Image

from app.services.claim_service import prepare_news_claim
from app.services.language_service import detect_language, normalize_text
from app.services.ocr_service import _clean_text, _read_pass, select_ocr_language
from scripts.train_sinhala_ocr import distance, metrics, normalize, parse_psm_modes, parse_tsv


class SinhalaOCRTests(unittest.TestCase):
    def test_scripts_and_combining_marks(self):
        self.assertEqual(detect_language("ශ්‍රී ලංකාවේ පුවත්"), "si")
        self.assertEqual(detect_language("ශ්‍රී ලංකාවේ COVID news"), "si+en")
        self.assertEqual(detect_language("News in Sri Lanka"), "en")
        self.assertEqual(detect_language("123 ! \u0dca"), "unknown")

    def test_cleaner_preserves_conjuncts_lines_and_short_words(self):
        text = "ශ්‍රී ලංකාවේ\nNews in a city to be shown"
        self.assertEqual(_clean_text(text), text)
        self.assertIn("\u200d", normalize_text(text))
        self.assertEqual(normalize_text(unicodedata.normalize("NFD", "කෝ")), "කෝ")
        serialized = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
        self.assertEqual(json.loads(serialized.decode("utf-8"))["text"], text)

    def test_lines_and_low_confidence_words_are_not_deleted(self):
        data = {"text": ["ශ්‍රී", "ලංකාවේ", "in", "a", "city"],
                "conf": [90, 88, 15, 95, 92], "block_num": [1]*5,
                "par_num": [1]*5, "line_num": [1, 1, 2, 2, 2]}
        with patch("app.services.ocr_service.pytesseract.image_to_data", return_value=data):
            text, confidence = _read_pass(Image.new("RGB", (10, 10)), "sin+eng", 6)
        self.assertEqual(text, "ශ්‍රී ලංකාවේ\nin a city")
        # Confidence is character-weighted, while the uncertain Sinhala token remains present.
        self.assertAlmostEqual(confidence, 81.7222, places=3)

    def test_training_tsv_parser_preserves_sinhala_and_weights_confidence(self):
        tsv = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
               "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90\tශ්‍රී\n"
               "5\t1\t1\t1\t1\t2\t10\t0\t20\t10\t50\tලංකාවේ\n")
        text, confidence = parse_tsv(tsv)
        self.assertEqual(text, "ශ්‍රී ලංකාවේ")
        self.assertGreater(confidence, 60)
        self.assertLess(confidence, 70)

    def test_psm_argument_validation(self):
        self.assertEqual(parse_psm_modes("6,11,6"), (6, 11))
        with self.assertRaises(Exception):
            parse_psm_modes("6,14")

    def test_high_confidence_sinhala_is_translated_but_not_verified(self):
        original = "ශ්‍රී ලංකාවේ නව පුවත්"
        result = prepare_news_claim({"extracted_text": original, "confidence": 91, "status": "success"},
                                    translator=lambda text, lang: "Latest news in Sri Lanka")
        self.assertEqual(result["original_claim"], original)
        self.assertEqual(result["english_claim"], "Latest news in Sri Lanka")
        self.assertFalse(result["translation_verified"])
        self.assertEqual(result["translation_status"], "unverified_translation_candidate")

    def test_mixed_and_english_routing(self):
        result = prepare_news_claim({"extracted_text": "ශ්‍රී ලංකාවේ COVID update", "confidence": 90, "status": "success"},
                                    translator=lambda text, lang: "COVID update in Sri Lanka")
        self.assertEqual(result["language"], "si+en")
        self.assertTrue(result["english_claim"])
        result = prepare_news_claim({"extracted_text": "News in a city", "confidence": 90, "status": "success"})
        self.assertEqual(result["english_claim"], "News in a city")
        self.assertEqual(result["translation_status"], "not_needed")

    def test_uncertain_ocr_and_failed_translation_do_not_become_claims(self):
        for confidence in (20, float("nan"), "invalid"):
            result = prepare_news_claim({"extracted_text": "ශ්‍රී ලංකාවේ පුවත්", "confidence": confidence, "status": "success"},
                                        translator=lambda *_: self.fail("Should not translate uncertain OCR"))
            self.assertFalse(result["english_claim"])
        for translated in ("", "ශ්‍රී ලංකාවේ පුවත්", "broken \ufffd text"):
            result = prepare_news_claim({"extracted_text": "ශ්‍රී ලංකාවේ පුවත්", "confidence": 90, "status": "success"},
                                        translator=lambda *_: translated)
            self.assertEqual(result["translation_status"], "translation_failed")
            self.assertFalse(result["english_claim"])

    def test_language_selection(self):
        self.assertEqual(select_ocr_language({"sin", "eng"}), "sin+eng")
        self.assertEqual(select_ocr_language({"sin"}), "sin")
        self.assertIsNone(select_ocr_language(set()))

    def test_metrics_and_normalization(self):
        self.assertEqual(distance("abc", "adc"), 1)
        self.assertEqual(normalize("ශ්‍රී  ලංකාවේ\nපුවත්"), "ශ්‍රී ලංකාවේ පුවත්")
        result = metrics([("abc def", "abc def"), ("abc", "adc")])
        self.assertEqual(result["character_error_rate"], 0.1)
        self.assertEqual(result["exact_line_accuracy"], 0.5)
        self.assertAlmostEqual(result["word_error_rate"], 1/3)


if __name__ == "__main__":
    unittest.main()
