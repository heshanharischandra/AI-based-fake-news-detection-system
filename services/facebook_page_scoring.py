
import csv
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import requests
from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

router = APIRouter(
    tags=["Facebook Page Evidence"]
)

WEIGHTS = {
    "page_identity": 15,
    "about_information": 15,
    "contact_information": 15,
    "linked_website": 15,
    "reviewed_publisher_association": 15,
    "page_category": 10,
    "location_information": 5,
    "page_transparency_information": 10,
}

PAGE_HOSTS = {
    "facebook.com",
    "www.facebook.com",
    "m.facebook.com",
    "web.facebook.com",
    "mbasic.facebook.com",
}

EVIDENCE_FILE = (
    ROOT / "data" / "facebook_page_evidence.csv"
)

IDENTITY_FILE = (
    ROOT / "data" / "identity_pairs_reviewed.csv"
)


class PageRequest(BaseModel):
    url: str = Field(
        min_length=12,
        max_length=2048
    )


class PostRequest(BaseModel):
    url: str = Field(
        min_length=12,
        max_length=2048
    )


def page_identifier(url, allow_post=True):
    try:
        parsed = urlsplit(
            url.strip()
        )

        host = (
            parsed.hostname or ""
        ).casefold().rstrip(".")

        if (
            parsed.scheme not in {
                "https",
                "http"
            }
            or host not in PAGE_HOSTS
            or parsed.username
            or parsed.password
            or parsed.port not in {
                None,
                80,
                443
            }
        ):
            return ""

        parts = [
            part
            for part in parsed.path.split("/")
            if part
        ]

        if not parts:
            return ""

        first = parts[0].lower()

        if first == "profile.php":
            candidate = parse_qs(
                parsed.query
            ).get("id", [""])[0]

        elif (
            first in {"pages", "people"}
            and len(parts) >= 3
        ):
            candidate = parts[-1]

        elif first in {
            "watch",
            "reel",
            "reels",
            "share",
            "groups",
            "events",
            "story.php",
            "photo.php",
            "permalink.php",
            "login",
            "help",
        }:
            return ""

        else:
            if (
                not allow_post
                and len(parts) > 1
                and parts[1].casefold() in {
                    "posts",
                    "photos",
                    "videos",
                    "reels",
                }
            ):
                return ""

            candidate = parts[0]

        if re.fullmatch(
            r"[a-zA-Z0-9._-]{2,110}",
            candidate or ""
        ):
            return candidate.casefold()

    except (
        ValueError,
        AttributeError
    ):
        pass

    return ""


def normalized_site(value):
    try:
        p = urlsplit(
            (value or "").strip()
        )

        host = (
            p.hostname or ""
        ).casefold()

        if (
            p.scheme in {
                "http",
                "https"
            }
            and host
            and not p.username
            and not p.password
        ):
            return p.geturl()

    except ValueError:
        pass

    return ""


def reviewed_associations(page_id):
    if not IDENTITY_FILE.is_file():
        return []

    found = []

    try:
        with IDENTITY_FILE.open(
            encoding="utf-8-sig",
            newline=""
        ) as stream:

            for row in csv.DictReader(stream):
                if (
                    row.get("platform")
                    or ""
                ).strip().casefold() != "facebook":
                    continue

                if page_identifier(
                    row.get("account_url") or ""
                ) != page_id:
                    continue

                if (
                    row.get("review_status")
                    or ""
                ).strip().casefold() != "reviewed":
                    continue

                if (
                    row.get("identity_label")
                    or ""
                ).strip().casefold() != "same_publisher":
                    continue

                evidence = normalized_site(
                    row.get("review_evidence_url")
                    or row.get("evidence_url")
                    or ""
                )

                site = normalized_site(
                    row.get("website_url")
                    or ""
                )

                if not evidence or not site:
                    continue

                found.append({
                    "publisher_name": (
                        row.get("publisher_name")
                        or ""
                    ),
                    "website_url": site,
                    "evidence_url": evidence,
                    "basis": (
                        "human_reviewed_association"
                    ),
                })

    except (
        OSError,
        csv.Error
    ):
        return []

    unique = {}

    for item in found:
        unique[item["website_url"]] = item

    return list(
        unique.values()
    )[:10]


def recent_review(value):
    try:
        date = datetime.fromisoformat(
            (value or "")
            .strip()
            .replace("Z", "+00:00")
        )

        if (
            date.tzinfo is None
            or date > datetime.now(timezone.utc)
        ):
            return None

        return date

    except ValueError:
        return None


def manual_evidence(page_id):
    if not EVIDENCE_FILE.is_file():
        return {}

    records = {}

    try:
        with EVIDENCE_FILE.open(
            encoding="utf-8-sig",
            newline=""
        ) as stream:

            for row in csv.DictReader(stream):
                if page_identifier(
                    row.get("page_url")
                    or ""
                ) != page_id:
                    continue

                indicator = (
                    row.get("indicator")
                    or ""
                ).strip().lower()

                status = (
                    row.get("status")
                    or ""
                ).strip().lower()

                evidence_url = normalized_site(
                    row.get("evidence_url")
                    or ""
                )

                reviewed_at = recent_review(
                    row.get("reviewed_at")
                )

                if (
                    indicator not in WEIGHTS
                    or indicator
                    == "reviewed_publisher_association"
                    or status not in {
                        "present",
                        "absent"
                    }
                    or not evidence_url
                    or reviewed_at is None
                    or not (
                        row.get("reviewer_id")
                        or ""
                    ).strip()
                    or len(
                        (
                            row.get("notes")
                            or ""
                        ).strip()
                    ) < 20
                ):
                    continue

                old = records.get(
                    indicator
                )

                if (
                    old is None
                    or old["reviewed_at"] < reviewed_at
                ):
                    records[indicator] = {
                        "status": status,
                        "evidence_url": evidence_url,
                        "reviewed_at": reviewed_at,
                        "basis": "human_reviewed",
                    }

    except (
        OSError,
        csv.Error
    ):
        return {}

    return records


def graph_metadata(page_id):
    token = os.getenv(
        "META_GRAPH_ACCESS_TOKEN",
        ""
    ).strip()

    if (
        not token
        or token.upper().startswith("YOUR_")
    ):
        return {}, "not_configured"

    version = os.getenv(
        "META_GRAPH_VERSION",
        "v26.0"
    ).strip()

    if not re.fullmatch(
        r"v\d+\.\d+",
        version
    ):
        version = "v26.0"

    field_sets = (
        "id,name,about,description,category,"
        "website,phone,emails,location",
        "id,name,about,website",
        "id,name",
    )

    for fields in field_sets:
        try:
            response = requests.get(
                "https://graph.facebook.com/"
                + version
                + "/"
                + quote(page_id, safe=""),
                params={
                    "fields": fields
                },
                headers={
                    "Authorization": (
                        "Bearer " + token
                    )
                },
                timeout=(4, 10),
            )

            if response.ok:
                data = response.json()

                if (
                    isinstance(data, dict)
                    and data.get("id")
                    and data.get("name")
                ):
                    return data, "retrieved"

        except (
            requests.RequestException,
            ValueError
        ):
            continue

    return {}, "access_unavailable"


def make_indicators(
    page_id,
    metadata,
    manually_reviewed,
    associations
):
    evidence = dict(
        manually_reviewed
    )

    if metadata:
        values = {
            "page_identity": bool(
                metadata.get("id")
                and metadata.get("name")
            ),
            "about_information": bool(
                (
                    metadata.get("about")
                    or metadata.get("description")
                    or ""
                ).strip()
            ),
            "contact_information": bool(
                metadata.get("phone")
                or metadata.get("emails")
            ),
            "linked_website": bool(
                normalized_site(
                    metadata.get("website")
                    or ""
                )
            ),
            "page_category": bool(
                (
                    metadata.get("category")
                    or ""
                ).strip()
            ),
            "location_information": bool(
                metadata.get("location")
            ),
        }

        for key, is_present in values.items():
            if is_present:
                evidence[key] = {
                    "status": "present",
                    "basis": "authorized_meta_api",
                    "evidence_url": "",
                    "reviewed_at": None,
                }

    if associations:
        evidence[
            "reviewed_publisher_association"
        ] = {
            "status": "present",
            "basis": (
                "human_reviewed_identity_dataset"
            ),
            "evidence_url": (
                associations[0]["evidence_url"]
            ),
            "reviewed_at": None,
        }

    indicators = {}

    for key, weight in WEIGHTS.items():
        observation = evidence.get(key)

        status = (
            observation["status"]
            if observation
            else "not_checked"
        )

        indicators[key] = {
            "label": (
                key.replace("_", " ").title()
            ),
            "weight": weight,
            "status": status,
            "awarded_points": (
                weight
                if status == "present"
                else 0
            ),
            "basis": (
                observation.get("basis")
                if observation
                else "not_checked"
            ),
            "evidence_url": (
                observation.get(
                    "evidence_url",
                    ""
                )
                if observation
                else ""
            ),
        }

    return indicators


@router.post("/facebook-page-score")
def facebook_page_score(
    request: PageRequest
):
    ident = page_identifier(
        request.url
    )

    if not ident:
        return {
            "status": "invalid_facebook_page_url",
            "score": None,
            "indicators": {},
        }

    associations = reviewed_associations(
        ident
    )

    manual = manual_evidence(
        ident
    )

    meta, meta_status = graph_metadata(
        ident
    )

    indicators = make_indicators(
        ident,
        meta,
        manual,
        associations
    )

    checked = [
        value
        for value in indicators.values()
        if value["status"] in {
            "present",
            "absent"
        }
    ]

    assessed_weight = sum(
        value["weight"]
        for value in checked
    )

    points = sum(
        value["awarded_points"]
        for value in indicators.values()
    )

    sufficient = (
        assessed_weight >= 60
        and len(checked) >= 4
    )

    return {
        "status": (
            "provisional_disclosure_score"
            if sufficient
            else "insufficient_reviewed_page_evidence"
        ),
        "platform": "facebook",
        "page_url": (
            "https://www.facebook.com/"
            + ident
        ),
        "page_name": meta.get("name"),
        "about": (
            meta.get("about")
            or meta.get("description")
        ),
        "score": (
            points
            if sufficient
            else None
        ),
        "observed_points": points,
        "assessed_weight": assessed_weight,
        "maximum_points": 100,
        "checked_indicators": len(checked),
        "meta_access": meta_status,
        "indicators": indicators,
        "reviewed_associations": associations,
        "interpretation": (
            "Documented Facebook Page profile information, "
            "not article factual accuracy."
        ),
    }


@router.post("/facebook-post-metadata")
def facebook_post_metadata(
    request: PostRequest
):
    token = os.getenv(
        "META_GRAPH_ACCESS_TOKEN",
        ""
    ).strip()

    if (
        not token
        or token.upper().startswith("YOUR_")
    ):
        return {
            "status": "not_configured",
            "created_time": None,
            "message": None,
        }

    try:
        parsed = urlsplit(
            request.url
        )

        parts = [
            x
            for x in parsed.path.split("/")
            if x
        ]

        if (
            not page_identifier(
                request.url
            )
            or len(parts) < 3
            or parts[1].casefold() != "posts"
            or not parts[-1].isdecimal()
        ):
            return {
                "status": "unsupported_post_url",
                "created_time": None,
                "message": None,
            }

        meta, state = graph_metadata(
            parts[0].casefold()
        )

        page_id = str(
            meta.get("id", "")
        )

        if not page_id.isdecimal():
            return {
                "status": state,
                "created_time": None,
                "message": None,
            }

        version = os.getenv(
            "META_GRAPH_VERSION",
            "v26.0"
        )

        if not re.fullmatch(
            r"v\d+\.\d+",
            version
        ):
            version = "v26.0"

        object_id = (
            page_id
            + "_"
            + parts[-1]
        )

        response = requests.get(
            "https://graph.facebook.com/"
            + version
            + "/"
            + quote(object_id, safe=""),
            params={
                "fields": (
                    "id,message,created_time,permalink_url"
                )
            },
            headers={
                "Authorization": (
                    "Bearer " + token
                )
            },
            timeout=(4, 10),
        )

        if response.ok:
            data = response.json()

            if (
                isinstance(data, dict)
                and data.get("id")
            ):
                return {
                    "status": "retrieved",
                    "created_time": data.get(
                        "created_time"
                    ),
                    "message": data.get(
                        "message"
                    ),
                    "permalink_url": data.get(
                        "permalink_url"
                    ),
                }

    except (
        requests.RequestException,
        ValueError
    ):
        pass

    return {
        "status": "access_unavailable",
        "created_time": None,
        "message": None,
    }
