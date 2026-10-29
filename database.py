
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING
from pymongo.errors import PyMongoError

load_dotenv()

_client = None


def database():
    global _client

    uri = os.getenv("MONGODB_URI", "").strip()

    if not uri:
        raise RuntimeError("MONGODB_URI is not configured")

    if _client is None:
        _client = MongoClient(
            uri,
            serverSelectionTimeoutMS=6000,
            connectTimeoutMS=6000
        )

    _client.admin.command("ping")

    return _client[
        os.getenv("MONGODB_DB", "source_credibility_db")
    ]


def healthy():
    try:
        database()
        return True
    except (PyMongoError, RuntimeError, OSError):
        return False


def close_database():
    global _client

    if _client is not None:
        _client.close()
        _client = None


def store_assessment(result):
    collection = database()["source_assessments"]

    collection.create_index([
        ("publisher.domain", 1),
        ("assessed_at", DESCENDING)
    ])

    document = {
        "publisher": result["publisher"],
        "article": result["article"],
        "transparency": result["transparency"],
        "social_accounts": result["social_accounts"],
        "article_verification": result["article_verification"],
        "evidence": result["evidence"],
        "assessment_version": "research-v3",
        "assessed_at": datetime.now(timezone.utc)
    }

    inserted = collection.insert_one(document)

    return {
        "status": "saved",
        "id": str(inserted.inserted_id),
        "assessed_at": document["assessed_at"].isoformat()
    }


def get_history(domain, limit=10):
    if (
        not isinstance(domain, str)
        or not domain
        or "/" in domain
        or len(domain) > 253
    ):
        raise ValueError("Invalid publisher domain")

    cursor = (
        database()["source_assessments"]
        .find(
            {"publisher.domain": domain.lower().strip()},
            {"article.text": 0, "evidence": 0}
        )
        .sort("assessed_at", DESCENDING)
        .limit(limit)
    )

    results = []

    for item in cursor:
        item["_id"] = str(item["_id"])

        date = item.get("assessed_at")

        if isinstance(date, datetime):
            item["assessed_at"] = date.replace(
                tzinfo=date.tzinfo or timezone.utc
            ).isoformat()

        results.append(item)

    return results
