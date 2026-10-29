import os
import tempfile
import unittest
from pathlib import Path
from threading import Lock
from unittest.mock import patch

from app.config import Settings
from app.database import MongoDatabase
from app.ui_jobs import JobStore


class FakeCollection:
    def __init__(self):
        self.documents = {}
        self.indexes = []

    def create_index(self, fields, name):
        self.indexes.append((fields, name))

    def replace_one(self, query, document, upsert=False):
        self.documents[query["_id"]] = document

    def find_one(self, query):
        return self.documents.get(query["_id"])


class FakeDatabase:
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, name):
        return self.collection


class FakeAdmin:
    def command(self, name):
        if name != "ping":
            raise AssertionError(name)


class FakeClient:
    def __init__(self):
        self.admin = FakeAdmin()
        self.collection = FakeCollection()
        self.closed = False

    def __getitem__(self, name):
        return FakeDatabase(self.collection)

    def close(self):
        self.closed = True


class RecordingRepository:
    def __init__(self):
        self.saved = []
        self.lock = Lock()

    def save_analysis(self, analysis):
        with self.lock:
            self.saved.append(analysis)

    def get_analysis(self, job_id):
        return None

    def recent_analyses(self, limit=20):
        return []


class DatabaseTests(unittest.TestCase):
    def test_settings_read_mongodb_environment(self):
        values = {
            "MONGODB_URI": "mongodb://db.example:27017",
            "MONGODB_DATABASE": "verification_test",
            "MONGODB_COLLECTION": "jobs",
            "MONGODB_CONNECT_TIMEOUT_MS": "1500",
        }
        with patch.dict(os.environ, values, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.mongodb_uri, values["MONGODB_URI"])
        self.assertEqual(settings.mongodb_database, "verification_test")
        self.assertEqual(settings.mongodb_collection, "jobs")
        self.assertEqual(settings.mongodb_connect_timeout_ms, 1500)

    def test_connect_save_and_load_analysis(self):
        client = FakeClient()
        database = MongoDatabase(
            Settings(mongodb_uri="mongodb://fake"), client_factory=lambda *args, **kwargs: client
        )
        self.assertTrue(database.connect())
        database.save_analysis({"id": "job-1", "filename": "news.png", "created_at": "2026-01-01T00:00:00+00:00"})
        result = database.get_analysis("job-1")
        self.assertEqual(result["id"], "job-1")
        self.assertEqual(result["filename"], "news.png")
        self.assertNotIn("updated_at", result)
        self.assertEqual(database.health()["status"], "connected")
        database.close()
        self.assertTrue(client.closed)

    def test_unconfigured_database_is_disabled(self):
        database = MongoDatabase(Settings(mongodb_uri=None))
        self.assertFalse(database.connect())
        self.assertEqual(database.health(), {"status": "disabled"})

    def test_collection_name_can_be_overridden_for_cli_records(self):
        client = FakeClient()
        database = MongoDatabase(
            Settings(mongodb_uri="mongodb://fake"),
            client_factory=lambda *args, **kwargs: client,
            collection_name="demo_analyses",
        )
        self.assertTrue(database.connect())
        self.assertEqual(database.health()["collection"], "demo_analyses")

    def test_job_lifecycle_is_persisted(self):
        repository = RecordingRepository()
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "input.png"
            image_path.write_bytes(b"image")
            store = JobStore(runner=lambda *args: None, repository=repository)
            store.create("job-1", image_path, "input.png", {"format": "PNG"}, {"include_details": False})
            store.close()
        statuses = [record["status"] for record in repository.saved]
        self.assertEqual(statuses[0], "queued")
        self.assertIn("running", statuses)
        self.assertEqual(statuses[-1], "completed")


if __name__ == "__main__":
    unittest.main()
