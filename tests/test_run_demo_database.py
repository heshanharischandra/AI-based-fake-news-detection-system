import unittest
from pathlib import Path

from run_demo import persist_demo_analysis


class FakeDatabase:
    def __init__(self):
        self.saved = None

    def connect(self):
        return True

    def save_analysis(self, analysis):
        self.saved = analysis


class RunDemoDatabaseTests(unittest.TestCase):
    def test_completed_analysis_is_normalized_and_saved(self):
        database = FakeDatabase()
        analysis = {
            "id": "demo-1",
            "filename": "news.png",
            "image": {"saved_path": Path("uploads/news.png")},
            "results": {"score": float("nan")},
        }

        self.assertTrue(persist_demo_analysis(analysis, database=database))
        self.assertEqual(database.saved["id"], "demo-1")
        self.assertEqual(database.saved["image"]["saved_path"], "uploads\\news.png")
        self.assertIsNone(database.saved["results"]["score"])


if __name__ == "__main__":
    unittest.main()
