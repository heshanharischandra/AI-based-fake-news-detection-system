import unittest

from app.services.pipeline_service import run_pipeline


class PipelineServiceTests(unittest.TestCase):
    def fake_services(self):
        return {
            "ocr": lambda path: {"status": "success", "extracted_text": "සිංහල පුවත් ශීර්ෂය", "detected_language": "si", "confidence": 0.91},
            "claim": lambda candidate, translator=None: {"status": "success", "original_claim": candidate.get("headline") or candidate.get("extracted_text"), "english_claim": "A verified comparison headline"},
            "visual": lambda path: {"status": "success", "description": "A news photograph"},
            "consistency": lambda path, text: {"status": "success", "similarity_score": 0.71, "model": "test"},
            "authenticity": lambda path, **kwargs: {"status": "success", "real_score": 0.6, "ai_score": 0.4},
            "reuse": lambda path: {"status": "success", "match_found": False},
            "context": lambda **kwargs: {"status": "insufficient_evidence", "evidence_status": "missing"},
            "score": lambda **kwargs: {"status": "insufficient_verified_evidence", "verification_score": None},
            "explanation": lambda **kwargs: {"status": "success", "summary": "Test explanation"},
            "evidence": lambda **kwargs: {"status": "no_verified_evidence_available", "records_found": 0},
        }

    def test_pipeline_keeps_sinhala_ocr_and_runs_core_stages(self):
        events = []
        result = run_pipeline("sample.png", {"include_details": False}, lambda *event: events.append(event), self.fake_services())
        self.assertEqual(result["ocr"]["detected_language"], "si")
        self.assertEqual(result["consistency"]["status"], "success")
        self.assertNotIn("explanation", result)
        self.assertNotIn("evidence", result)
        self.assertEqual(len(events), 16)  # running + completed for eight visible stages

    def test_optional_headline_and_comparison_are_passed_to_pipeline(self):
        services = self.fake_services()
        seen = {}

        def claim(candidate, translator=None):
            seen["headline"] = candidate.get("headline")
            seen["comparison"] = translator(candidate) if translator else None
            return {"status": "success", "original_claim": seen["headline"], "english_claim": seen["comparison"]}

        services["claim"] = claim
        result = run_pipeline("sample.png", {"headline": "Reviewed headline", "comparison_text": "English comparison", "include_details": True}, lambda *event: None, services)
        self.assertEqual(seen, {"headline": "Reviewed headline", "comparison": "English comparison"})
        self.assertEqual(result["claim"]["english_claim"], "English comparison")
        self.assertIn("explanation", result)
        self.assertIn("evidence", result)


if __name__ == "__main__":
    unittest.main()
