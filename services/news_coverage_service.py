
import os
import re
from difflib import SequenceMatcher
from urllib.parse import urlencode
from xml.etree import ElementTree

import requests


def clean_query(value):
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            "Article title is required"
        )

    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()[:180]


def headline_similarity(original, candidate):
    a = clean_query(
        original
    ).casefold()

    b = (
        clean_query(candidate).casefold()
        if str(candidate or "").strip()
        else ""
    )

    return round(
        SequenceMatcher(
            None,
            a,
            b
        ).ratio(),
        3
    )


def search_news_websites(
    title,
    language="si",
    max_results=6
):
    query = clean_query(
        title
    )

    params = {
        "q": f'"{query}"',
        "hl": (
            "si"
            if language == "si"
            else "en"
        ),
        "gl": "LK",
        "ceid": (
            "LK:si"
            if language == "si"
            else "LK:en"
        )
    }

    url = (
        "https://news.google.com/rss/search?"
        + urlencode(params)
    )

    try:
        response = requests.get(
            url,
            timeout=(4, 8),
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        if len(response.content) > 2000000:
            raise ValueError(
                "RSS response too large"
            )

        root = ElementTree.fromstring(
            response.content
        )

        results = []

        items = root.findall(
            "./channel/item"
        )

        for item in items[:max_results * 3]:
            heading = item.findtext(
                "title"
            ) or ""

            link = item.findtext(
                "link"
            ) or ""

            publisher = item.find(
                "source"
            )

            publisher_name = (
                publisher.text
                if publisher is not None
                else None
            )

            similarity = (
                headline_similarity(
                    query,
                    heading
                )
                if heading.strip()
                else 0
            )

            if link.startswith("https://"):
                results.append({
                    "platform": "news_website_index",
                    "headline": heading,
                    "source_name": publisher_name,
                    "search_result_url": link,
                    "published": item.findtext(
                        "pubDate"
                    ),
                    "headline_similarity": similarity,
                    "evidence_status": (
                        "indexed_result_not_independently_verified"
                    )
                })

            if len(results) >= max_results:
                break

        return {
            "status": "completed",
            "results": results
        }

    except (
        requests.RequestException,
        ValueError,
        ElementTree.ParseError
    ) as error:
        return {
            "status": "unavailable",
            "error_type": type(error).__name__,
            "results": []
        }


def search_youtube(title, max_results=5):
    key = os.getenv(
        "YOUTUBE_API_KEY",
        ""
    ).strip()

    if not key:
        return {
            "status": "api_key_required",
            "results": []
        }

    params = {
        "key": key,
        "part": "snippet",
        "type": "video",
        "maxResults": max_results,
        "q": clean_query(title)
    }

    try:
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params=params,
            timeout=(4, 8)
        )

        response.raise_for_status()

        payload = response.json()
        results = []

        for item in payload.get("items", []):
            video_id = item.get(
                "id",
                {}
            ).get("videoId")

            snippet = item.get(
                "snippet",
                {}
            )

            if not video_id:
                continue

            results.append({
                "platform": "youtube",
                "video_url": (
                    f"https://www.youtube.com/watch?v={video_id}"
                ),
                "headline": snippet.get("title"),
                "channel_title": snippet.get(
                    "channelTitle"
                ),
                "published": snippet.get(
                    "publishedAt"
                ),
                "headline_similarity": headline_similarity(
                    title,
                    snippet.get("title") or ""
                ),
                "evidence_status": (
                    "youtube_search_result_not_factual_confirmation"
                )
            })

        return {
            "status": "completed",
            "results": results
        }

    except (
        requests.RequestException,
        ValueError,
        TypeError
    ) as error:
        return {
            "status": "unavailable",
            "error_type": type(error).__name__,
            "results": []
        }


def discover_news_coverage(article):
    if not isinstance(article, dict):
        raise ValueError(
            "Article details are required"
        )

    title = article.get(
        "article_title"
    )

    if not title:
        raise ValueError(
            "Article title is missing"
        )

    return {
        "article_title": title,
        "other_websites": search_news_websites(
            title,
            article.get(
                "article_language",
                "si"
            )
        ),
        "youtube": search_youtube(
            title
        ),
        "facebook": {
            "status": (
                "not_checked_requires_authorized_platform_access"
            ),
            "results": []
        },
        "interpretation": (
            "Search matches are candidate coverage, "
            "not corroboration of truth or account ownership."
        )
    }
