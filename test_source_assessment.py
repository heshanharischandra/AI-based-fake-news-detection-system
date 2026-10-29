
import unittest

from services.source_assessment import (
    WEIGHTS,
    build_source_assessment,
)

from unified_api import social_identifier


class SourceCredibilityTests(unittest.TestCase):

    def test_total_weights(self):
        self.assertEqual(
            sum(WEIGHTS.values()),
            100,
        )

        self.assertEqual(
            len(WEIGHTS),
            8,
        )

    def test_expected_seventy_points(self):
        raw = {
            "publisher": {
                "name": "Test Publisher",
                "domain": "example.com",
                "website": "https://example.com/",
                "name_evidence": (
                    "explicit_site_declaration"
                ),
            },
            "article": {
                "url": "https://example.com/story",
                "headline": "Example headline",
            },
            "social_accounts": [
                {
                    "platform": "facebook",
                    "url": (
                        "https://www.facebook.com/examplepublisher"
                    ),
                }
            ],
            "transparency": {
                "indicators": {},
            },
            "evidence": {
                "page_inspection": {
                    "contact": {
                        "status": "inspected",
                        "content_status": "supported",
                        "url": (
                            "https://example.com/contact"
                        ),
                    },
                    "about": {
                        "status": "inspected",
                        "content_status": "supported",
                        "ownership_statement_status": (
                            "declared_not_verified"
                        ),
                        "url": (
                            "https://example.com/about"
                        ),
                    },
                    "editorial": {
                        "status": "not_established",
                    },
                    "corrections": {
                        "status": "not_established",
                    },
                }
            },
        }

        result = build_source_assessment(
            raw,
            language="en",
        )

        value = result["transparency"]

        self.assertEqual(
            value["observed_disclosure_points"],
            70,
        )

        self.assertEqual(
            value["possible_points"],
            100,
        )

        self.assertEqual(
            sum(
                item["observed_points"]
                for item in value[
                    "indicators"
                ].values()
            ),
            70,
        )

        self.assertEqual(
            value["indicators"][
                "editorial_policy"
            ]["observed_points"],
            0,
        )

        self.assertEqual(
            value["indicators"][
                "corrections_policy"
            ]["observed_points"],
            0,
        )

    def test_facebook_page_and_youtube_channel(self):
        self.assertEqual(
            social_identifier(
                "https://www.facebook.com/hirunews"
            )[0],
            "facebook",
        )

        self.assertEqual(
            social_identifier(
                "https://www.youtube.com/@HiruNewsOfficial"
            )[0],
            "youtube",
        )

    def test_facebook_post_rejected_as_page(self):
        with self.assertRaises(ValueError):
            social_identifier(
                "https://www.facebook.com/hirunews/posts/123456"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
