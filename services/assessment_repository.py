
import re
from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING

from database import get_database


COLLECTION_NAME = "source_assessments"


def validate_domain(domain):
    if not isinstance(domain, str):
        raise ValueError(
            "Publisher domain is required"
        )

    value = domain.strip().lower()

    if (
        len(value) > 253
        or not re.fullmatch(
            r"[a-z0-9-]+(?:\.[a-z0-9-]+)+",
            value
        )
    ):
        raise ValueError(
            "Invalid publisher domain"
        )

    return value


def get_collection():
    return get_database()[COLLECTION_NAME]


def ensure_assessment_indexes():
    get_collection().create_index(
        [
            ("domain", ASCENDING),
            ("assessed_at", DESCENDING)
        ],
        name="publisher_assessment_history"
    )


def serialize_assessment(document):
    result = dict(document)

    if "_id" in result:
        result["_id"] = str(
            result["_id"]
        )

    timestamp = result.get(
        "assessed_at"
    )

    if isinstance(timestamp, datetime):
        result["assessed_at"] = timestamp.replace(
            tzinfo=timestamp.tzinfo or timezone.utc
        ).isoformat()

    return result


def save_assessment(
    scoring_result,
    source=None,
    cross_platform=None
):
    if (
        not isinstance(scoring_result, dict)
        or scoring_result.get("analysis_status")
        != "scoring_completed"
    ):
        raise ValueError(
            "Completed scoring result required"
        )

    source = source or {}

    domain = validate_domain(
        source.get("domain")
        or scoring_result.get("domain")
    )

    document = {
        "domain": domain,
        "source_name": (
            source.get("source_name")
            or scoring_result.get("source_name")
        ),
        "source_url": (
            source.get("source_url")
            or scoring_result.get("source_url")
        ),
        "article_url": source.get("final_url"),
        "article_language": source.get(
            "article_language"
        ),
        "scores": {
            "evidence_score": scoring_result[
                "evidence_score"
            ],
            "credibility_score": scoring_result[
                "credibility_score"
            ],
            "evidence_coverage_percent": scoring_result[
                "evidence_coverage_percent"
            ],
            "trust_level": scoring_result[
                "trust_level"
            ],
            "score_status": scoring_result[
                "score_status"
            ]
        },
        "indicator_breakdown": scoring_result[
            "indicator_breakdown"
        ],
        "cross_platform": cross_platform or {},
        "assessment_version": "pp1-transparency-v2",
        "assessed_at": datetime.now(
            timezone.utc
        )
    }

    ensure_assessment_indexes()

    inserted = get_collection().insert_one(
        document
    )

    return {
        "saved": True,
        "assessment_id": str(
            inserted.inserted_id
        ),
        "assessed_at": document[
            "assessed_at"
        ].isoformat()
    }


def get_source_history(domain, limit=10):
    domain = validate_domain(
        domain
    )

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 100
    ):
        raise ValueError(
            "Limit must be 1 to 100"
        )

    cursor = (
        get_collection()
        .find({"domain": domain})
        .sort("assessed_at", DESCENDING)
        .limit(limit)
    )

    return [
        serialize_assessment(item)
        for item in cursor
    ]


def get_latest_assessment(domain):
    domain = validate_domain(
        domain
    )

    document = get_collection().find_one(
        {"domain": domain},
        sort=[("assessed_at", DESCENDING)]
    )

    return (
        serialize_assessment(document)
        if document
        else None
    )


def compare_recent_assessments(domain):
    history = get_source_history(
        domain,
        2
    )

    if len(history) < 2:
        return {
            "status": "insufficient_history",
            "changes": []
        }

    latest, previous = (
        history[0],
        history[1]
    )

    changes = []

    old_breakdown = previous.get(
        "indicator_breakdown",
        {}
    )

    new_breakdown = latest.get(
        "indicator_breakdown",
        {}
    )

    names = sorted(
        set(new_breakdown)
        | set(old_breakdown)
    )

    for name in names:
        old = old_breakdown.get(
            name,
            {}
        ).get("status")

        new = new_breakdown.get(
            name,
            {}
        ).get("status")

        if old != new:
            changes.append({
                "indicator": name,
                "previous": old,
                "current": new
            })

    return {
        "status": "compared",
        "previous_assessment": previous["_id"],
        "latest_assessment": latest["_id"],
        "changes": changes
    }
