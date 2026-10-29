
import csv
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[1]

FILE = (
    ROOT
    / "data"
    / "source_accuracy_reviews.csv"
)

router = APIRouter(
    tags=["Research evidence"]
)


def valid_domain(value):
    value = (
        value or ""
    ).strip().casefold()

    if (
        not re.fullmatch(
            r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?",
            value
        )
        or ".." in value
        or "." not in value
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid publisher domain"
        )

    return value


def valid_evidence_url(value):
    try:
        p = urlsplit(
            (value or "").strip()
        )

        return (
            p.scheme == "https"
            and bool(p.hostname)
            and not p.username
            and not p.password
        )

    except ValueError:
        return False


def reviewed_accuracy(domain):
    domain = valid_domain(domain)

    if not FILE.is_file():
        return {
            "status": "insufficient_reviewed_evidence",
            "domain": domain,
            "reviewed_articles": 0,
            "minimum_articles": 10,
            "reviewers": 0
        }

    selected = {}

    required = {
        "domain",
        "article_url",
        "verdict",
        "reviewer_id",
        "review_date",
        "independent_evidence_url",
        "notes",
        "review_status"
    }

    try:
        with FILE.open(
            encoding="utf-8-sig",
            newline=""
        ) as stream:
            reader = csv.DictReader(
                stream
            )

            if not required.issubset(
                set(
                    reader.fieldnames
                    or []
                )
            ):
                return {
                    "status": "review_file_missing_columns",
                    "domain": domain,
                    "required_columns": sorted(required)
                }

            for r in reader:
                if (
                    r.get("domain")
                    or ""
                ).strip().casefold() != domain:
                    continue

                if (
                    r.get("review_status")
                    or ""
                ).strip().lower() != "reviewed":
                    continue

                verdict = (
                    r.get("verdict")
                    or ""
                ).strip().lower()

                if verdict not in {
                    "corroborated",
                    "contradicted"
                }:
                    continue

                if not valid_evidence_url(
                    r.get("article_url")
                ):
                    continue

                if not valid_evidence_url(
                    r.get("independent_evidence_url")
                ):
                    continue

                if (
                    len(
                        (
                            r.get("notes")
                            or ""
                        ).strip()
                    ) < 30
                ):
                    continue

                if not (
                    r.get("reviewer_id")
                    or ""
                ).strip():
                    continue

                if not (
                    r.get("review_date")
                    or ""
                ).strip():
                    continue

                article = (
                    r["article_url"]
                    .strip()
                    .rstrip("/")
                    .casefold()
                )

                reviewer = (
                    r["reviewer_id"]
                    .strip()
                    .casefold()
                )

                selected[
                    (
                        article,
                        reviewer
                    )
                ] = {
                    "article_url": r["article_url"].strip(),
                    "reviewer_id": reviewer,
                    "verdict": verdict,
                    "evidence_url": r[
                        "independent_evidence_url"
                    ].strip()
                }

    except (
        OSError,
        csv.Error
    ):
        return {
            "status": "review_file_unavailable",
            "domain": domain
        }

    by_article = {}

    for review in selected.values():
        key = review[
            "article_url"
        ].lower()

        by_article.setdefault(
            key,
            []
        ).append(review)

    agreed = []

    for article, assessments in by_article.items():
        voters = {
            x["reviewer_id"]
            for x in assessments
        }

        labels = {
            x["verdict"]
            for x in assessments
        }

        if (
            len(voters) >= 2
            and len(labels) == 1
        ):
            agreed.append({
                "article_url": assessments[0]["article_url"],
                "verdict": assessments[0]["verdict"],
                "evidence_urls": sorted(
                    {
                        x["evidence_url"]
                        for x in assessments
                    }
                )
            })

    counts = Counter(
        x["verdict"]
        for x in agreed
    )

    ready = (
        len(agreed) >= 10
    )

    return {
        "status": (
            "exploratory_sample"
            if ready
            else "insufficient_reviewed_evidence"
        ),
        "domain": domain,
        "reviewed_articles": len(agreed),
        "corroborated_articles": counts[
            "corroborated"
        ],
        "contradicted_articles": counts[
            "contradicted"
        ],
        "minimum_articles": 10,
        "reviewers_per_article_required": 2,
        "sample_corroboration_percent": (
            round(
                100
                * counts["corroborated"]
                / len(agreed),
                1
            )
            if ready
            else None
        ),
        "warning": (
            "Exploratory evidence sample, not an "
            "overall verified publisher trust score. "
            "Article selection and reviewer bias "
            "must be evaluated."
        )
    }


@router.get("/reviewed-accuracy/{domain}")
def reviewed_accuracy_route(domain: str):
    return reviewed_accuracy(domain)
