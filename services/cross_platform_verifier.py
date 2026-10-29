
import re
from urllib.parse import urlsplit


def normalize_name(value):
    return re.sub(
        r"[^\w\u0d80-\u0dff]",
        "",
        str(value or "").casefold()
    )


def verify_cross_platform(website_result):
    if not isinstance(website_result, dict):
        raise ValueError(
            "Website analysis is required"
        )

    source_name = normalize_name(
        website_result.get("source_name")
    )

    results = []

    definitions = (
        ("facebook", "facebook_page_urls"),
        ("youtube", "youtube_channel_urls")
    )

    for platform, field in definitions:
        for url in website_result.get(field, []):
            path = urlsplit(url).path.rstrip("/")

            handle = path.rsplit("/", 1)[-1].lstrip("@")

            candidate = normalize_name(
                handle
            )

            similarity = (
                "similar"
                if (
                    source_name
                    and candidate
                    and (
                        source_name in candidate
                        or candidate in source_name
                    )
                )
                else "not_established"
            )

            results.append({
                "platform": platform,
                "url": url,
                "association_evidence": "link_on_publisher_article_page",
                "handle_name_consistency": similarity,
                "ownership_verified": False,
                "identity_status": "website_associated_only"
            })

    return {
        "accounts": results,
        "associated_platforms": sorted(set(
            item["platform"]
            for item in results
        )),
        "identity_verification": (
            "not_independently_verified"
            if results
            else "no_accounts_discovered"
        ),
        "explanation_en": (
            "Website links establish an association, "
            "not account ownership or story publication."
        ),
        "explanation_si": (
            "වෙබ් සබැඳි හරහා සම්බන්ධතාවයක් පෙන්වයි; "
            "ගිණුමේ හිමිකාරීත්වය හෝ එම පුවත පළ කළ බව තහවුරු නොවේ."
        )
    }
