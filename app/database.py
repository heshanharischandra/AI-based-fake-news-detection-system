"""MongoDB connection and analysis persistence."""
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from time import monotonic

from bson.errors import BSONError
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError

from app.config import Settings


class MongoDatabase:
    """Small synchronous MongoDB gateway suitable for FastAPI sync/thread workers."""

    def __init__(self, settings=None, client_factory=MongoClient, collection_name=None):
        self.settings = settings or Settings.from_env()
        self.collection_name = collection_name or self.settings.mongodb_collection
        self._client_factory = client_factory
        self._client = None
        self._collection = None
        self._error = None
        self._retry_after = 0
        self._lock = Lock()

    @property
    def configured(self):
        return bool(self.settings.mongodb_uri)

    @property
    def connected(self):
        return self._collection is not None

    def connect(self):
        """Connect, verify the server, and prepare indexes."""
        if not self.configured:
            return False
        with self._lock:
            if self.connected:
                return True
            if monotonic() < self._retry_after:
                return False
            client = None
            try:
                client = self._client_factory(
                    self.settings.mongodb_uri,
                    serverSelectionTimeoutMS=self.settings.mongodb_connect_timeout_ms,
                    connectTimeoutMS=self.settings.mongodb_connect_timeout_ms,
                    appname="image-verification-system",
                )
                client.admin.command("ping")
                collection = client[self.settings.mongodb_database][self.collection_name]
                collection.create_index([("created_at", DESCENDING)], name="created_at_desc")
                collection.create_index([("status", ASCENDING)], name="status_idx")
                self._client = client
                self._collection = collection
                self._error = None
                self._retry_after = 0
                return True
            except (PyMongoError, OSError, ValueError) as error:
                if client is not None:
                    client.close()
                self._error = str(error)[:500]
                self._retry_after = monotonic() + 30
                return False

    def save_analysis(self, analysis):
        collection = self._require_collection()
        document = deepcopy(analysis)
        job_id = document.pop("id")
        document["updated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            collection.replace_one({"_id": job_id}, {"_id": job_id, **document}, upsert=True)
        except (PyMongoError, BSONError) as error:
            self._record_operation_error(error)
            raise RuntimeError(self._error) from error

    def get_analysis(self, job_id):
        try:
            document = self._require_collection().find_one({"_id": job_id})
        except (PyMongoError, BSONError) as error:
            self._record_operation_error(error)
            raise RuntimeError(self._error) from error
        return self._to_analysis(document)

    def recent_analyses(self, limit=20):
        try:
            cursor = self._require_collection().find(
                {}, {"filename": 1, "created_at": 1, "status": 1, "image_url": 1}
            ).sort("created_at", DESCENDING).limit(limit)
            return [self._to_analysis(document) for document in cursor]
        except (PyMongoError, BSONError) as error:
            self._record_operation_error(error)
            raise RuntimeError(self._error) from error

    def health(self):
        if not self.configured:
            return {"status": "disabled"}
        if self.connected:
            return {
                "status": "connected",
                "database": self.settings.mongodb_database,
                "collection": self.collection_name,
            }
        return {"status": "unavailable", "error": self._error or "Connection has not been attempted."}

    def close(self):
        with self._lock:
            if self._client is not None:
                self._client.close()
            self._client = None
            self._collection = None

    def _require_collection(self):
        if self._collection is None and not self.connect():
            raise RuntimeError(self._error or "MongoDB is not configured.")
        return self._collection

    def _record_operation_error(self, error):
        with self._lock:
            client = self._client
            self._client = None
            self._collection = None
            self._error = str(error)[:500]
            self._retry_after = monotonic() + 30
        if client is not None:
            client.close()

    @staticmethod
    def _to_analysis(document):
        if document is None:
            return None
        result = dict(document)
        result["id"] = str(result.pop("_id"))
        result.pop("updated_at", None)
        return result
