import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.reference_registry import register_verified_reference
from app.services import context_service
from app.services.scoring_service import calculate_verification_score


class ReferenceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.jpg"
        self.source.write_bytes(b"reference-image")
        self.references = self.root / "references"
        self.metadata = self.references / "reference_metadata.json"

    def register(self):
        return register_verified_reference(
            self.source,
            source_name="Trusted publisher",
            source_url="https://publisher.example/article",
            publication_date="2025-06-02",
            event_date="2025-06-01",
            event_name="City Bridge Flood",
            location="Colombo",
            original_claim="City Bridge Flood 2025 in Colombo",
            filename="bridge-flood.jpg",
            reference_dir=self.references,
            metadata_path=self.metadata,
        )

    def test_verified_reference_enables_full_context_coverage_for_matching_claim(self):
        record = self.register()
        self.assertEqual(record["evidence_status"], "verified")
        self.assertTrue((self.references / "bridge-flood.jpg").is_file())

        with patch.object(context_service, "METADATA_PATH", self.metadata):
            context = context_service.verify_context(
                "City Bridge Flood 2025 in Colombo",
                {"possible_reuse": True, "matched_reference": "bridge-flood.jpg", "status": "exact_duplicate"},
            )
        score = calculate_verification_score(context)
        self.assertEqual(context["evidence_status"], "verified")
        self.assertEqual(score["context_evidence_coverage_percent"], 100.0)
        self.assertEqual(score["verification_score"], 100.0)

    def test_registry_rejects_incomplete_or_duplicate_records(self):
        with self.assertRaisesRegex(ValueError, "valid HTTP"):
            register_verified_reference(
                self.source, source_name="Publisher", source_url="not-a-url",
                publication_date="2025-01-01", event_date="2025-01-01",
                event_name="Event", location="Colombo", original_claim="Claim",
                reference_dir=self.references, metadata_path=self.metadata,
            )
        self.register()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.register()


if __name__ == "__main__":
    unittest.main()
