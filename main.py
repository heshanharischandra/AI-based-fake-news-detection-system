
import logging

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from database import (
    close_database,
    get_history,
    healthy,
    store_assessment
)

from services.analysis import (
    analyze_article,
    analyze_source,
    fetch_html
)

from services.discovery import find_related_coverage

from services.source_assessment import (
    build_source_assessment
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    close_database()


app = FastAPI(
    title="Cross-Platform News Source Credibility Intelligence",
    version="4.0-research-prototype",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


class AssessmentRequest(BaseModel):
    url: str = Field(
        min_length=8,
        max_length=2048
    )

    output_language: Literal["si", "en"] = "si"


def ensure_news_url(url):
    parsed = urlsplit(
        url
        if "://" in url
        else "https://" + url
    )

    host = (
        parsed.hostname or ""
    ).lower()

    if host in {
        "facebook.com",
        "www.facebook.com",
        "m.facebook.com",
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com"
    }:
        raise HTTPException(
            status_code=501,
            detail=(
                "Direct social-account assessment requires "
                "authorized platform integration; "
                "provide a publisher website URL"
            )
        )


@app.get("/")
def home():
    return {
        "system": (
            "Cross-Platform News Source Credibility Intelligence"
        ),
        "version": "4.0-research-prototype",
        "primary_task": "publisher_source_assessment",
        "supported_languages": [
            "si",
            "en"
        ],
        "article_truth_detection": "separate_component"
    }


@app.get("/health")
def health():
    return {
        "api_status": "running",
        "database_connected": healthy(),
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    }


@app.post("/assess")
def assess(request: AssessmentRequest):
    ensure_news_url(
        request.url
    )

    try:
        raw = analyze_source(
            request.url
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        logging.exception(
            "Unexpected source analysis error"
        )

        raise HTTPException(
            status_code=500,
            detail="Source analysis failed unexpectedly"
        ) from exc

    previous = []
    history_available = True

    try:
        previous = get_history(
            raw["publisher"]["domain"],
            10
        )

    except (
        PyMongoError,
        RuntimeError,
        OSError
    ):
        history_available = False

    try:
        report = build_source_assessment(
            raw,
            language=request.output_language,
            previous_assessments=(
                previous
                if history_available
                else None
            )
        )

    except (
        KeyError,
        TypeError,
        ValueError
    ) as exc:
        logging.exception(
            "Source profile assembly error"
        )

        raise HTTPException(
            status_code=500,
            detail="Source profile assembly failed"
        ) from exc

    if not history_available:
        report["source_history"] = {
            "status": "unavailable",
            "note_en": (
                "Previous assessments could not "
                "be retrieved from MongoDB."
            ),
            "note_si": (
                "MongoDB වෙතින් පෙර ඇගයීම් "
                "ලබාගත නොහැකි විය."
            ),
            "changed_indicators": []
        }

    try:
        report["database"] = store_assessment(
            report
        )

    except (
        PyMongoError,
        RuntimeError,
        OSError
    ) as exc:
        report["database"] = {
            "status": "not_saved",
            "error_type": type(exc).__name__
        }

    return report


@app.post("/coverage")
def coverage(request: AssessmentRequest):
    ensure_news_url(
        request.url
    )

    try:
        article = analyze_article(
            fetch_html(request.url)
        )["article"]

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc)
        ) from exc

    return {
        "article": {
            "url": article.get("url"),
            "headline": article.get("headline"),
            "publication_date": article.get(
                "publication_date"
            )
        },
        "discovery": find_related_coverage(
            article
        )
    }


@app.get("/history/{domain}")
def history(
    domain: str,
    limit: int = Query(
        10,
        ge=1,
        le=100
    )
):
    try:
        records = get_history(
            domain,
            limit
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except (
        PyMongoError,
        RuntimeError,
        OSError
    ) as exc:
        raise HTTPException(
            status_code=503,
            detail="Assessment history unavailable"
        ) from exc

    return {
        "domain": domain,
        "total": len(records),
        "assessments": records
    }
