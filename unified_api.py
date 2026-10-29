
import csv
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fastapi import HTTPException
from pydantic import BaseModel, Field

from main import app
from services.facebook_news_assessment import facebook_assess
from services.identity_prediction_api import router as identity_router
from services.source_assessment import WEIGHTS, METHOD_VERSION

ROOT = Path(__file__).resolve().parent

FACEBOOK_HOSTS = {
    "facebook.com",
    "www.facebook.com",
    "m.facebook.com",
    "web.facebook.com",
    "mbasic.facebook.com",
    "fb.com",
    "www.fb.com",
}

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
}


class SocialRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2048)


def classify_account(raw):
    value = raw.strip()

    if not value.startswith(("https://", "http://")):
        value = "https://" + value

    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Invalid account URL") from exc

    if (
        parsed.scheme not in ("http", "https")
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 80, 443)
    ):
        raise ValueError("Enter a normal public account URL")

    parts = [p for p in parsed.path.split("/") if p]

    if host in FACEBOOK_HOSTS:
        if not parts:
            raise ValueError(
                "Enter a Facebook Page URL, not the homepage"
            )

        lower = [p.lower() for p in parts]

        if (
            lower[0] in {
                "watch",
                "reel",
                "reels",
                "share",
                "groups",
                "events",
                "stories",
                "story.php",
                "permalink.php",
                "photo.php",
                "login",
                "help",
                "marketplace",
            }
            or (
                len(lower) >= 3
                and lower[1] in {
                    "posts",
                    "videos",
                    "photos",
                    "reels",
                }
            )
        ):
            raise ValueError(
                "This is a Facebook post or non-Page URL; "
                "use the news post input"
            )

        if lower[0] == "profile.php":
            ident = parse_qs(
                parsed.query
            ).get("id", [""])[0]

            if not ident.isdigit():
                raise ValueError(
                    "Facebook profile.php URL requires a numeric ID"
                )

        elif (
            lower[0] in {"pages", "people"}
            and len(parts) >= 3
        ):
            ident = lower[-1]

        else:
            ident = lower[0]

        if not re.fullmatch(r"[a-z0-9._-]{2,100}", ident):
            raise ValueError(
                "Invalid Facebook Page identifier"
            )

        return "facebook", ident, value

    if host in YOUTUBE_HOSTS:
        if (
            len(parts) == 1
            and parts[0].startswith("@")
        ):
            return (
                "youtube",
                parts[0].lower(),
                value,
            )

        if (
            len(parts) == 2
            and parts[0].lower() in {
                "channel",
                "user",
                "c",
            }
        ):
            return (
                "youtube",
                parts[1].lower(),
                value,
            )

        raise ValueError(
            "Enter a YouTube channel URL, not a video URL"
        )

    raise ValueError(
        "Only Facebook Pages and YouTube channels "
        "are supported here"
    )


def registry_matches(platform, identifier):
    paths = [
        (
            ROOT / "data" / "identity_pairs.csv",
            "Sri Lankan candidate dataset",
        ),
        (
            ROOT / "data" / "identity_pairs_reviewed.csv",
            "Sri Lankan review dataset",
        ),
        (
            ROOT
            / "data"
            / "imported_1000"
            / "identity_pairs_reviewed.csv",
            "Historical US candidate dataset",
        ),
    ]

    found = {}

    for path, origin in paths:
        if not path.exists():
            continue

        try:
            with path.open(
                encoding="utf-8-sig",
                newline="",
            ) as stream:
                for row in csv.DictReader(stream):
                    if (
                        (row.get("platform") or "")
                        .lower()
                        .strip()
                        != platform
                    ):
                        continue

                    try:
                        _, candidate, _ = classify_account(
                            row.get("account_url") or ""
                        )
                    except (ValueError, AttributeError):
                        continue

                    if candidate != identifier:
                        continue

                    key = (
                        row.get("publisher_id") or "",
                        platform,
                        identifier,
                    )

                    entry = {
                        "publisher_name": row.get(
                            "publisher_name",
                            "",
                        ),
                        "publisher_website": row.get(
                            "website_url",
                            "",
                        ),
                        "account_url": row.get(
                            "account_url",
                            "",
                        ),
                        "review_status": (
                            row.get("review_status")
                            or "pending"
                        ),
                        "identity_label": (
                            row.get("identity_label")
                            or "unknown"
                        ),
                        "evidence_url": (
                            row.get("review_evidence_url")
                            or row.get("evidence_url")
                            or ""
                        ),
                        "dataset": origin,
                    }

                    if (
                        key not in found
                        or (
                            entry["review_status"] == "reviewed"
                            and found[key]["review_status"]
                            != "reviewed"
                        )
                    ):
                        found[key] = entry

        except (OSError, csv.Error):
            continue

    return list(found.values())[:20]


@app.post(
    "/social-source",
    tags=["Cross-platform identity"],
)
def social_source(request: SocialRequest):
    try:
        platform, identifier, normalized = (
            classify_account(request.url)
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    matches = registry_matches(
        platform,
        identifier,
    )

    return {
        "platform": platform,
        "account_url": normalized,
        "source_identifier": identifier,
        "matches": matches,
        "identity_status": (
            "candidate_associations_only"
            if matches
            else "no_registry_match"
        ),
        "account_ownership_verified": False,
        "credibility_score": None,
        "explanation": (
            "A registered or model-predicted association "
            "is not independent proof of account ownership "
            "or factual reliability."
        ),
    }


@app.get(
    "/scoring-method",
    tags=["Research methodology"],
)
def scoring_method():
    return {
        "version": METHOD_VERSION,
        "weights": WEIGHTS,
        "maximum": sum(WEIGHTS.values()),
        "meaning": (
            "Observed website transparency/disclosure, "
            "not article truth or publisher factuality"
        ),
    }


app.add_api_route(
    "/facebook-assess",
    facebook_assess,
    methods=["POST"],
    tags=["Facebook News"],
    name="facebook_assess",
)

app.include_router(identity_router)
from services.facebook_page_scoring import router as facebook_page_router

app.include_router(facebook_page_router)

from services.reviewed_accuracy import router as reviewed_accuracy_router

app.include_router(reviewed_accuracy_router)

