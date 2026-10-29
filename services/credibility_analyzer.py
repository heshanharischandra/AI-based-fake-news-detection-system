
from services.website_analyzer import clean_text


INDICATOR_WEIGHTS = {
    "https_used": 5,
    "publisher_identity": 15,
    "author_information": 5,
    "publication_date": 5,
    "contact_information": 15,
    "about_information": 15,
    "editorial_policy": 15,
    "corrections_policy": 10,
    "ownership_disclosure": 10,
    "social_presence": 5
}


def indicator(
    status,
    explanation_en,
    explanation_si,
    evidence_url=None
):
    return {
        "status": status,
        "evidence_url": evidence_url,
        "explanation_en": explanation_en,
        "explanation_si": explanation_si
    }


def evaluate_credibility(
    fetch_result,
    website_result,
    page_result
):
    if not all(
        isinstance(item, dict)
        for item in (
            fetch_result,
            website_result,
            page_result
        )
    ):
        raise ValueError(
            "Invalid assessment data"
        )

    url = website_result.get("final_url")

    if not url:
        raise ValueError(
            "Article URL is missing"
        )

    checks = page_result.get(
        "page_checks",
        {}
    )

    about = checks.get(
        "about",
        {}
    )

    source_name = clean_text(
        website_result.get("source_name")
    )

    about_excerpt = clean_text(
        about.get("content_excerpt")
    ).casefold()

    named_in_about = bool(
        source_name
        and source_name.casefold().replace(" ", "")
        in about_excerpt.replace(" ", "")
    )

    publisher_declared = (
        website_result.get("publisher_identity_status")
        == "declared"
    )

    publisher_supported = (
        publisher_declared
        or (
            about.get("content_status") == "substantive"
            and named_in_about
        )
    )

    indicators = {}

    https = url.lower().startswith(
        "https://"
    )

    indicators["https_used"] = indicator(
        "found" if https else "not_found",
        (
            "Article retrieved over HTTPS"
            if https
            else "Article retrieved over HTTP"
        ),
        (
            "ලිපිය HTTPS හරහා ලබාගන්නා ලදී"
            if https
            else "ලිපිය HTTP හරහා ලබාගන්නා ලදී"
        ),
        url
    )

    indicators["publisher_identity"] = indicator(
        "found" if publisher_supported else "unknown",
        (
            "Publisher name explicitly declared by page evidence"
            if publisher_supported
            else "Publisher name is only inferred or not independently documented"
        ),
        (
            "ප්‍රකාශක නාමය පිටුවේ සාක්ෂි මගින් සඳහන් වේ"
            if publisher_supported
            else "ප්‍රකාශක නාමය තහවුරු කරන පැහැදිලි සාක්ෂි නොමැත"
        ),
        (
            url
            if publisher_declared
            else (
                about.get("evidence_url")
                if publisher_supported
                else None
            )
        )
    )

    author = website_result.get(
        "author_name"
    )

    indicators["author_information"] = indicator(
        "found" if author else "unknown",
        (
            "Article credits an author"
            if author
            else "No author credit was identified"
        ),
        (
            "ලිපියේ කර්තෘ සඳහන් වේ"
            if author
            else "කර්තෘ නාමයක් හමු නොවීය"
        ),
        url if author else None
    )

    date = website_result.get(
        "publication_date"
    )

    indicators["publication_date"] = indicator(
        "found" if date else "unknown",
        (
            "Article publication date identified"
            if date
            else "Article publication date could not be determined"
        ),
        (
            "ලිපියේ ප්‍රකාශන දිනය හමු විය"
            if date
            else "ලිපියේ ප්‍රකාශන දිනය හඳුනාගත නොහැකි විය"
        ),
        url if date else None
    )

    definitions = (
        (
            "contact",
            "contact_information",
            "contact",
            "සම්බන්ධතා"
        ),
        (
            "about",
            "about_information",
            "about",
            "අප ගැන"
        ),
        (
            "editorial_policy",
            "editorial_policy",
            "editorial policy",
            "කර්තෘ ප්‍රතිපත්තිය"
        ),
        (
            "corrections_policy",
            "corrections_policy",
            "corrections policy",
            "නිවැරදි කිරීම් ප්‍රතිපත්තිය"
        )
    )

    for category, name, english, sinhala in definitions:
        page = checks.get(
            category,
            {}
        )

        found = (
            page.get("status") == "reachable"
            and page.get("content_status") == "substantive"
        )

        indicators[name] = indicator(
            "found" if found else "unknown",
            (
                f"Substantive {english} information was detected"
                if found
                else f"Substantive {english} information has not been confirmed"
            ),
            (
                f"{sinhala} පිළිබඳ තොරතුරු හමු විය"
                if found
                else f"{sinhala} පිළිබඳ අර්ථවත් තොරතුරු තහවුරු කර නොමැත"
            ),
            page.get("evidence_url")
        )

    ownership = (
        about.get("content_status") == "substantive"
        and about.get(
            "content_features",
            {}
        ).get(
            "ownership_statement_present",
            False
        )
    )

    indicators["ownership_disclosure"] = indicator(
        "found" if ownership else "unknown",
        (
            "Ownership-related disclosure language detected; ownership not independently verified"
            if ownership
            else "No substantive ownership disclosure was confirmed"
        ),
        (
            "හිමිකාරීත්වය පිළිබඳ සඳහනක් හමු විය; ස්වාධීනව තහවුරු කර නැත"
            if ownership
            else "හිමිකාරීත්වය පිළිබඳ තොරතුරු තහවුරු කර නොමැත"
        ),
        (
            about.get("evidence_url")
            if ownership
            else None
        )
    )

    social = bool(
        website_result.get("facebook_page_urls")
        or website_result.get("youtube_channel_urls")
    )

    indicators["social_presence"] = indicator(
        "found" if social else "unknown",
        (
            "Website declares social account links; ownership has not been independently checked"
            if social
            else "No publisher-declared social links found"
        ),
        (
            "වෙබ් අඩවිය සමාජ මාධ්‍ය සබැඳි දක්වයි; හිමිකාරීත්වය තහවුරු කර නැත"
            if social
            else "සමාජ මාධ්‍ය සබැඳි හමු නොවීය"
        ),
        (
            website_result.get("source_url")
            if social
            else None
        )
    )

    return {
        "indicators": indicators,
        "summary": {
            "total_indicators": len(indicators),
            "supported": sum(
                item["status"] == "found"
                for item in indicators.values()
            ),
            "unknown": sum(
                item["status"] == "unknown"
                for item in indicators.values()
            ),
            "negative": sum(
                item["status"] == "not_found"
                for item in indicators.values()
            )
        },
        "analysis_status": "credibility_indicators_collected"
    }
