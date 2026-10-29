
import csv
import html
import json
import os
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal


router = APIRouter(tags=["Facebook News"])
BASE_DIR = Path(__file__).resolve().parents[1]
FACEBOOK_HOSTS = {
    "facebook.com",
    "www.facebook.com",
    "m.facebook.com",
    "web.facebook.com",
    "mbasic.facebook.com",
    "fb.watch"
}


class FacebookRequest(BaseModel):
    url: str = Field(min_length=12, max_length=2048)
    output_language: Literal["en", "si"] = "en"


class PreviewParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.meta = {}

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "meta":
            return
        data = dict(attrs)
        key = str(data.get("property") or data.get("name") or "").lower().strip()
        value = str(data.get("content") or "").strip()
        if key and value and key not in self.meta:
            self.meta[key] = html.unescape(value)


def validate_facebook_post(url):
    try:
        parsed = urlsplit(url.strip())
        host = (parsed.hostname or "").lower().rstrip(".")

        if parsed.scheme != "https" or host not in FACEBOOK_HOSTS:
            raise ValueError("Use an HTTPS Facebook post or video URL")

        if parsed.username or parsed.password or parsed.port is not None:
            raise ValueError("URL credentials and custom ports are not allowed")

        parts = [part for part in parsed.path.split("/") if part]
        query = parse_qs(parsed.query)

        is_post = (
            (
                len(parts) >= 3
                and parts[1].lower() in {
                    "posts", "videos", "photos", "reels"
                }
            )
            or (
                len(parts) >= 2
                and parts[0].lower() in {"reel", "share"}
            )
            or (
                len(parts) >= 4
                and parts[0].lower() == "groups"
                and parts[2].lower() == "posts"
            )
            or (
                parts == ["watch"]
                and bool(query.get("v"))
            )
            or (
                parts in (["story.php"], ["permalink.php"])
                and bool(query.get("story_fbid"))
            )
            or (
                parts in (["photo.php"], ["photo"])
                and bool(query.get("fbid"))
            )
            or (
                host == "fb.watch"
                and bool(parts)
            )
        )

        if not is_post:
            raise ValueError(
                "Enter a Facebook post/video URL, not a Facebook Page homepage"
            )

        return parsed.geturl()

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc


def find_publisher(post_url):
    parts = [
        p for p in urlsplit(post_url).path.strip("/").split("/")
        if p
    ]

    if (
        len(parts) < 3
        or parts[1].lower() not in {
            "posts", "videos", "photos", "reels"
        }
    ):
        return None

    handle = parts[0].casefold()

    for name in (
        "identity_pairs_reviewed.csv",
        "identity_pairs.csv"
    ):
        csv_path = BASE_DIR / "data" / name

        if not csv_path.is_file():
            continue

        try:
            with csv_path.open(
                encoding="utf-8-sig",
                newline=""
            ) as file:
                for record in csv.DictReader(file):
                    if (
                        (record.get("platform") or "").strip().lower()
                        != "facebook"
                    ):
                        continue

                    account = (
                        record.get("account_url") or ""
                    ).strip()

                    account_host = (
                        urlsplit(account).hostname or ""
                    ).lower()

                    account_handle = (
                        urlsplit(account)
                        .path.strip("/")
                        .casefold()
                    )

                    if (
                        account_host not in FACEBOOK_HOSTS
                        or account_handle != handle
                    ):
                        continue

                    return {
                        "name": record.get("publisher_name") or "",
                        "website_url": record.get("website_url") or "",
                        "facebook_account_url": account,
                        "association_status": "dataset_candidate",
                        "human_review_status": (
                            record.get("review_status") or "pending"
                        ),
                        "human_review_label": (
                            record.get("identity_label") or "unknown"
                        ),
                        "evidence_url": (
                            record.get("review_evidence_url")
                            or record.get("evidence_url")
                            or ""
                        ),
                    }

        except (OSError, csv.Error):
            continue

    return None


def get_post_number(url):
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)

    candidates = (
        query.get("story_fbid", [])
        + query.get("fbid", [])
        + query.get("v", [])
    )

    parts = [
        part for part in parsed.path.split("/")
        if part
    ]

    if (
        len(parts) >= 3
        and parts[1].lower() in {
            "posts", "videos", "photos", "reels"
        }
    ):
        candidates.append(parts[-1])

    if (
        len(parts) == 2
        and parts[0].lower() == "reel"
    ):
        candidates.append(parts[1])

    return next(
        (value for value in candidates if value.isdecimal()),
        None
    )


def authorized_graph_post(url):
    token = os.getenv(
        "META_PAGE_ACCESS_TOKEN",
        ""
    ).strip()

    page_id = os.getenv(
        "META_PAGE_ID",
        ""
    ).strip()

    page_handle = os.getenv(
        "META_PAGE_HANDLE",
        ""
    ).strip().casefold()

    parts = [
        p for p in urlsplit(url).path.strip("/").split("/")
        if p
    ]

    actual_handle = (
        parts[0].casefold()
        if len(parts) >= 3 and parts[1].lower() == "posts"
        else ""
    )

    post_id = get_post_number(url)

    if not (
        token
        and page_id.isdecimal()
        and page_handle
        and actual_handle == page_handle
        and post_id
    ):
        return None

    version = os.getenv(
        "META_GRAPH_API_VERSION",
        "v26.0"
    ).strip()

    if not re.fullmatch(r"v\d+\.\d+", version):
        version = "v26.0"

    try:
        response = requests.get(
            f"https://graph.facebook.com/{version}/{page_id}_{post_id}",
            headers={
                "Authorization": f"Bearer {token}"
            },
            params={
                "fields": (
                    "id,message,story,created_time,"
                    "permalink_url,full_picture,from"
                )
            },
            timeout=(5, 12),
        )

        if response.status_code != 200:
            return None

        post = response.json()
        returned_id = str(post.get("id") or "")
        author_id = str(
            (post.get("from") or {}).get("id") or ""
        )

        if (
            returned_id != f"{page_id}_{post_id}"
            or (author_id and author_id != page_id)
        ):
            return None

        body = str(
            post.get("message")
            or post.get("story")
            or ""
        )

        return {
            "title": body[:160],
            "text": body,
            "published_at": post.get("created_time"),
            "image_url": post.get("full_picture") or "",
            "post_url": post.get("permalink_url") or url,
            "page_name": str(
                (post.get("from") or {}).get("name") or ""
            ),
            "method": "authorized_meta_graph_api",
        }

    except (
        requests.RequestException,
        ValueError,
        TypeError
    ):
        return None


def get_public_preview(url):
    current = url
    session = requests.Session()
    session.trust_env = False

    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml"
    })

    try:
        for _ in range(4):
            if (
                (urlsplit(current).hostname or "").lower()
                not in FACEBOOK_HOSTS
            ):
                return None

            with session.get(
                current,
                stream=True,
                allow_redirects=False,
                timeout=(5, 10)
            ) as response:

                if response.status_code in {
                    301, 302, 303, 307, 308
                }:
                    location = response.headers.get("Location")

                    if not location:
                        return None

                    current = urljoin(current, location)
                    continue

                if (
                    response.status_code != 200
                    or "html" not in response.headers.get(
                        "Content-Type", ""
                    ).lower()
                ):
                    return None

                data = bytearray()

                for chunk in response.iter_content(16384):
                    data.extend(chunk)

                    if len(data) > 1_200_000:
                        return None

                parser = PreviewParser()

                parser.feed(
                    data.decode(
                        "utf-8",
                        errors="replace"
                    )
                )

                meta = parser.meta
                title = meta.get("og:title", "").strip()
                text = meta.get("og:description", "").strip()

                generic = (
                    "log in to facebook",
                    "log into facebook",
                    "create an account",
                    "see more on facebook"
                )

                if title.casefold() in {
                    "facebook",
                    "log in",
                    "log into facebook"
                }:
                    title = ""

                if any(
                    phrase in text.casefold()
                    for phrase in generic
                ):
                    text = ""

                if len(title) < 12 and len(text) < 25:
                    return None

                return {
                    "title": title,
                    "text": text,
                    "published_at": meta.get(
                        "article:published_time"
                    ),
                    "image_url": meta.get("og:image") or "",
                    "post_url": url,
                    "page_name": "",
                    "method": "public_preview_metadata",
                }

    except (
        requests.RequestException,
        OSError,
        ValueError
    ):
        return None

    finally:
        session.close()

    return None


@router.post("/facebook-assess")
def facebook_assess(request: FacebookRequest):
    url = validate_facebook_post(request.url)

    publisher = find_publisher(url)

    graph_post = authorized_graph_post(url)
    post = graph_post or get_public_preview(url)

    status = (
        "extracted"
        if graph_post
        else "partial_public_metadata"
        if post
        else "restricted_or_unavailable"
    )

    return {
        "platform": "facebook",
        "submitted_url": url,
        "extraction_status": status,
        "extraction_method": (
            post["method"] if post else "none"
        ),
        "post": post or {
            "title": "",
            "text": "",
            "published_at": None,
            "image_url": "",
            "post_url": url,
            "page_name": ""
        },
        "publisher": publisher or {
            "name": "",
            "website_url": "",
            "association_status": "unknown"
        },
        "source_reliability": {
            "status": "not_assessed"
        },
        "explanation": (
            "Post details were retrieved with authorized Meta access."
            if graph_post
            else "Only public preview metadata was accessible."
            if post
            else "Facebook post content was unavailable; this is not a truth verdict."
        ),
        "explanation_en": (
            "Post details were retrieved using authorized Meta Page access."
            if graph_post
            else "Only public preview metadata was retrieved, not the complete post."
            if post
            else "Facebook did not expose usable post content. This does not mean the post is false."
        ),
        "explanation_si": (
            "අවසරලත් Meta API මගින් පෝස්ට් තොරතුරු ලබාගන්නා ලදී."
            if graph_post
            else "පෝස්ට් එකේ පොදු පෙරදසුන් තොරතුරු පමණක් ලබාගන්නා ලදී."
            if post
            else "Facebook පෝස්ට් අන්තර්ගතය ලබාගත නොහැකි විය. එය අසත්‍ය බවට සාක්ෂියක් නොවේ."
        ),
    }
