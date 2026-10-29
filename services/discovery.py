
import os

from difflib import SequenceMatcher
from urllib.parse import urlencode, urlsplit
from xml.etree import ElementTree

import requests

from dotenv import load_dotenv

load_dotenv()


def similarity(left, right):
    return round(
        SequenceMatcher(
            None,
            str(left or "").casefold(),
            str(right or "").casefold()
        ).ratio(),
        3
    )


def news_index(title, language):
    query = " ".join(
        (title or "").split()
    )[:150]

    if not query:
        return {
            "status": "not_checked",
            "matches": []
        }

    params = {
        "q": query,
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
            timeout=(4, 10),
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        if len(response.content) > 2000000:
            raise ValueError("RSS too large")

        root = ElementTree.fromstring(
            response.content
        )

        results = []

        for item in root.findall(
            "./channel/item"
        )[:30]:
            headline = item.findtext(
                "title"
            ) or ""

            link = item.findtext(
                "link"
            ) or ""

            if urlsplit(link).scheme != "https":
                continue

            source = item.find(
                "source"
            )

            results.append({
                "platform": "news_index",
                "headline": headline,
                "publisher": (
                    source.text
                    if source is not None
                    else None
                ),
                "search_result_url": link,
                "publication_date": item.findtext(
                    "pubDate"
                ),
                "headline_similarity": similarity(
                    title,
                    headline
                ),
                "interpretation": "candidate_coverage_not_verified"
            })

            if len(results) >= 8:
                break

        return {
            "status": (
                "results_found"
                if results
                else "no_matches_found"
            ),
            "matches": results
        }

    except (
        requests.RequestException,
        ValueError,
        ElementTree.ParseError
    ) as exc:
        return {
            "status": "unavailable",
            "error": type(exc).__name__,
            "matches": []
        }


def youtube_index(title):
    key = os.getenv(
        "YOUTUBE_API_KEY",
        ""
    ).strip()

    if not key:
        return {
            "status": "api_key_not_configured",
            "matches": []
        }

    try:
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "key": key,
                "q": title[:150],
                "part": "snippet",
                "type": "video",
                "maxResults": 5
            },
            timeout=(4, 10)
        )

        response.raise_for_status()

        results = []

        for item in response.json().get(
            "items",
            []
        ):
            video = item.get(
                "id",
                {}
            ).get("videoId")

            snippet = item.get(
                "snippet",
                {}
            )

            if video:
                results.append({
                    "platform": "youtube",
                    "video_url": (
                        "https://www.youtube.com/watch?v="
                        + video
                    ),
                    "headline": snippet.get(
                        "title"
                    ),
                    "channel": snippet.get(
                        "channelTitle"
                    ),
                    "publication_date": snippet.get(
                        "publishedAt"
                    ),
                    "headline_similarity": similarity(
                        title,
                        snippet.get("title")
                    ),
                    "interpretation": "candidate_coverage_not_verified"
                })

        return {
            "status": (
                "results_found"
                if results
                else "no_matches_found"
            ),
            "matches": results
        }

    except (
        requests.RequestException,
        ValueError,
        TypeError
    ) as exc:
        return {
            "status": "unavailable",
            "error": type(exc).__name__,
            "matches": []
        }


def factcheck_index(title):
    key = os.getenv(
        "FACTCHECK_API_KEY",
        ""
    ).strip()

    if not key:
        return {
            "status": "api_key_not_configured",
            "reviews": []
        }

    try:
        response = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={
                "key": key,
                "query": title[:150],
                "pageSize": 5
            },
            timeout=(4, 10)
        )

        response.raise_for_status()

        reviews = []

        for claim in response.json().get(
            "claims",
            []
        ):
            for review in claim.get(
                "claimReview",
                []
            ):
                reviews.append({
                    "claim": claim.get("text"),
                    "reviewer": review.get(
                        "publisher",
                        {}
                    ).get("name"),
                    "review_url": review.get(
                        "url"
                    ),
                    "textual_rating": review.get(
                        "textualRating"
                    ),
                    "claim_similarity": similarity(
                        title,
                        claim.get("text")
                    ),
                    "match_status": "candidate_requires_claim_comparison"
                })

        return {
            "status": (
                "results_found"
                if reviews
                else "no_matches_found"
            ),
            "reviews": reviews[:10]
        }

    except (
        requests.RequestException,
        ValueError,
        TypeError
    ) as exc:
        return {
            "status": "unavailable",
            "error": type(exc).__name__,
            "reviews": []
        }


def find_related_coverage(article):
    title = article.get(
        "headline"
    )

    if not title:
        return {
            "status": "no_headline",
            "article_verdict": "not_verified"
        }

    return {
        "article_verdict": "not_verified",
        "website_coverage": news_index(
            title,
            article.get("language")
        ),
        "youtube_coverage": youtube_index(
            title
        ),
        "fact_check_reviews": factcheck_index(
            title
        ),
        "facebook_coverage": {
            "status": "authorized_access_not_configured",
            "matches": []
        },
        "explanation_en": (
            "Search results suggest possible coverage, "
            "not factual confirmation. Check event, "
            "date, location and original evidence before "
            "assigning a verdict."
        ),
        "explanation_si": (
            "සෙවුම් ප්‍රතිඵල අනුව සමාන පුවත් තිබිය හැකිය. "
            "එයින් සත්‍යතාව තහවුරු නොවේ. තීරණයකට පෙර "
            "සිදුවීම, දිනය, ස්ථානය සහ මුල් සාක්ෂි "
            "පරීක්ෂා කළ යුතුය."
        )
    }
