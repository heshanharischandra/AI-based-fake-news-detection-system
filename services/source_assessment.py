
import re
from urllib.parse import urlsplit


WEIGHTS = {
    "secure_connection": 5,
    "publisher_identification": 15,
    "contact_disclosure": 15,
    "organization_information": 15,
    "ownership_declaration": 10,
    "editorial_policy": 15,
    "corrections_policy": 15,
    "publisher_social_links": 10,
}

METHOD_VERSION = "source_disclosure_v4"

LABELS = {
    "secure_connection": (
        "HTTPS connection",
        "HTTPS සම්බන්ධතාව"
    ),
    "publisher_identification": (
        "Publisher identification",
        "ප්‍රකාශක හඳුනාගැනීම"
    ),
    "contact_disclosure": (
        "Contact information",
        "සම්බන්ධතා තොරතුරු"
    ),
    "organization_information": (
        "Organization information",
        "ආයතනික තොරතුරු"
    ),
    "ownership_declaration": (
        "Ownership disclosure",
        "හිමිකාරීත්වය පිළිබඳ ප්‍රකාශය"
    ),
    "editorial_policy": (
        "Editorial standards",
        "කර්තෘ ප්‍රතිපත්තිය"
    ),
    "corrections_policy": (
        "Corrections policy",
        "නිවැරදි කිරීම් ප්‍රතිපත්තිය"
    ),
    "publisher_social_links": (
        "Declared social accounts",
        "ප්‍රකාශිත සමාජ මාධ්‍ය ගිණුම්"
    ),
}


def _normalized(value):
    return re.sub(
        r"[^\w\u0d80-\u0dff]",
        "",
        str(value or "").casefold()
    )


def _valid_url(value):
    if not isinstance(value, str):
        return None

    parsed = urlsplit(value)

    if (
        parsed.scheme in {"https", "http"}
        and parsed.hostname
    ):
        return value

    return None


def _evidence_registry(article, publisher, pages):
    catalog = {}
    seen = {}

    def add(url, kind):
        url = _valid_url(url)

        if not url:
            return None

        if url in seen:
            return seen[url]

        identifier = f"E{len(catalog) + 1}"

        catalog[identifier] = {
            "url": url,
            "type": kind
        }

        seen[url] = identifier

        return identifier

    references = {
        "article": add(
            article.get("url"),
            "article"
        ),
        "publisher": add(
            publisher.get("website"),
            "publisher_website"
        ),
    }

    for key in (
        "contact",
        "about",
        "editorial",
        "corrections"
    ):
        references[key] = add(
            pages.get(key, {}).get("url"),
            key
        )

    return catalog, references


def _history(current_indicators, current_score, history):
    comparable = None

    for record in history or []:
        old = record.get(
            "transparency",
            {}
        )

        if old.get("method") == METHOD_VERSION:
            comparable = record
            break

    if comparable is None:
        return {
            "status": "baseline_created",
            "note_en": (
                "No previous comparable assessment was found."
            ),
            "note_si": (
                "සසඳා බැලිය හැකි පෙර ඇගයීමක් හමු නොවීය."
            ),
            "changed_indicators": [],
        }

    previous = comparable["transparency"]

    previous_indicators = previous.get(
        "indicators",
        {}
    )

    changes = []

    for key in WEIGHTS:
        before = previous_indicators.get(
            key,
            {}
        ).get("status")

        after = current_indicators[key]["status"]

        if before != after:
            changes.append({
                "indicator": key,
                "previous": before,
                "current": after
            })

    before_score = previous.get(
        "observed_disclosure_points"
    )

    delta = (
        current_score - before_score
        if isinstance(before_score, (int, float))
        and not isinstance(before_score, bool)
        else None
    )

    return {
        "status": (
            "observed_change"
            if changes
            else "no_observed_change"
        ),
        "changed_indicators": changes,
        "point_change": delta,
        "interpretation": (
            "Changes in extraction or site availability "
            "can also affect the comparison."
        ),
    }


def build_source_assessment(
    raw,
    language="si",
    previous_assessments=None
):
    if not isinstance(raw, dict):
        raise ValueError(
            "Analysis result must be an object"
        )

    publisher = raw.get("publisher") or {}
    article = raw.get("article") or {}

    pages = (
        (raw.get("evidence") or {})
        .get("page_inspection")
        or {}
    )

    social = raw.get("social_accounts") or []

    previous_flags = (
        (raw.get("transparency") or {})
        .get("indicators")
        or {}
    )

    name = publisher.get("name")
    domain = publisher.get("domain")

    if not name or not domain or not article.get("url"):
        raise ValueError(
            "Source and article identifiers are missing"
        )

    evidence, reference = _evidence_registry(
        article,
        publisher,
        pages
    )

    def supported_page(key):
        page = pages.get(key) or {}

        return (
            page.get("status") == "inspected"
            and page.get("content_status") == "supported"
        )

    name_declared = (
        publisher.get("name_evidence")
        == "explicit_site_declaration"
    )

    about_name_match = (
        supported_page("about")
        and (
            previous_flags.get(
                "publisher_identified"
            ) or {}
        ).get("status") == "observed"
    )

    owner_declared = (
        supported_page("about")
        and (
            pages.get("about") or {}
        ).get("ownership_statement_status")
        == "declared_not_verified"
    )

    observed = {
        "secure_connection": article["url"].startswith(
            "https://"
        ),
        "publisher_identification": (
            name_declared or about_name_match
        ),
        "contact_disclosure": supported_page(
            "contact"
        ),
        "organization_information": supported_page(
            "about"
        ),
        "ownership_declaration": owner_declared,
        "editorial_policy": supported_page(
            "editorial"
        ),
        "corrections_policy": supported_page(
            "corrections"
        ),
        "publisher_social_links": bool(social),
    }

    method = {
        "secure_connection": "transport_observation",
        "publisher_identification": (
            "website_metadata"
            if name_declared
            else (
                "about_page_name_match"
                if about_name_match
                else "domain_inference_only"
            )
        ),
        "contact_disclosure": (
            "retrieved_page_contact_details"
        ),
        "organization_information": (
            "retrieved_about_page"
        ),
        "ownership_declaration": (
            "publisher_self_declaration_not_verified"
        ),
        "editorial_policy": (
            "retrieved_policy_page"
        ),
        "corrections_policy": (
            "retrieved_policy_page"
        ),
        "publisher_social_links": (
            "publisher_website_links_not_account_verification"
        ),
    }

    source_key = {
        "secure_connection": "article",
        "publisher_identification": (
            "article"
            if name_declared
            else "about"
        ),
        "contact_disclosure": "contact",
        "organization_information": "about",
        "ownership_declaration": "about",
        "editorial_policy": "editorial",
        "corrections_policy": "corrections",
        "publisher_social_links": "publisher",
    }

    indicators = {}

    for key, weight in WEIGHTS.items():
        if observed[key]:
            status = (
                "publisher_declared"
                if key in {
                    "ownership_declaration",
                    "publisher_social_links"
                }
                else "observed"
            )
        else:
            status = "not_established"

        evidence_id = (
            reference.get(source_key[key])
            if observed[key]
            else None
        )

        indicators[key] = {
            "label_en": LABELS[key][0],
            "label_si": LABELS[key][1],
            "status": status,
            "weight": weight,
            "observed_points": (
                weight
                if observed[key]
                else 0
            ),
            "method": method[key],
            "evidence_ids": (
                [evidence_id]
                if evidence_id
                else []
            ),
        }

    points = sum(
        item["observed_points"]
        for item in indicators.values()
    )

    platforms = sorted({
        item.get("platform")
        for item in social
        if item.get("platform") in (
            "facebook",
            "youtube"
        )
    })

    accounts = []

    root_name = _normalized(name)

    for item in social:
        url = _valid_url(
            item.get("url")
        )

        if not url:
            continue

        handle = _normalized(
            urlsplit(url)
            .path.strip("/")
            .split("/")[-1]
            .lstrip("@")
        )

        lexical_match = (
            "similar_text"
            if (
                root_name
                and handle
                and (
                    root_name in handle
                    or handle in root_name
                )
            )
            else "not_established"
        )

        accounts.append({
            "platform": item.get("platform"),
            "url": url,
            "association": (
                "linked_from_publisher_website"
            ),
            "handle_similarity": lexical_match,
            "account_ownership": (
                "not_independently_checked"
            ),
        })

    language_index = (
        1 if language == "si" else 0
    )

    present = [
        LABELS[key][language_index]
        for key in WEIGHTS
        if observed[key]
    ]

    missing = [
        LABELS[key][language_index]
        for key in WEIGHTS
        if not observed[key]
    ]

    if language == "si":
        explanation = (
            f"{name} ({domain}) මූලාශ්‍රය සඳහා "
            "මෙම පරීක්ෂාවේදී හමු වූ ප්‍රකාශක තොරතුරු: "
            f"{', '.join(present) if present else 'තහවුරු කළ තොරතුරු නොමැත'}. "
            "මෙවර සොයාගත නොහැකි තොරතුරු: "
            f"{', '.join(missing) if missing else 'නැත'}. "
            "වෙබ් අඩවියෙන් දක්වා ඇති Facebook/YouTube "
            "සම්බන්ධතා ගිණුම් හිමිකාරීත්වය ස්වාධීනව "
            "තහවුරු නොකරයි. "
            "මෙය මූලාශ්‍රයේ ප්‍රකාශිත විනිවිදභාවය "
            "පිළිබඳ ඇගයීමක් වන අතර පුවතේ සත්‍යතාව "
            "හෝ මූලාශ්‍රයේ සමස්ත විශ්වසනීයත්වය "
            "සඳහා තීන්දුවක් නොවේ."
        )

    else:
        explanation = (
            f"For {name} ({domain}), this scan found: "
            f"{', '.join(present) if present else 'no supported disclosures'}. "
            "Not established in this scan: "
            f"{', '.join(missing) if missing else 'none'}. "
            "Website-linked Facebook/YouTube accounts "
            "are associations, not independently verified ownership. "
            "This assesses disclosed source transparency, "
            "not the truth of this article or the publisher's "
            "overall factual reliability."
        )

    article_out = {
        key: article.get(key)
        for key in (
            "url",
            "headline",
            "language",
            "publication_date",
            "author",
            "summary",
            "text",
            "image_url"
        )
        if article.get(key) is not None
    }

    history = _history(
        indicators,
        points,
        previous_assessments
    )

    return {
        "assessment_type": "publisher_source_profile",
        "publisher": publisher,
        "article": article_out,
        "transparency": {
            "method": METHOD_VERSION,
            "observed_disclosure_points": points,
            "possible_points": sum(
                WEIGHTS.values()
            ),
            "score_meaning": (
                "disclosure_presence_not_source_reliability_or_truth_probability"
            ),
            "publisher_reliability_verdict": (
                "not_evaluated"
            ),
            "indicators": indicators,
        },
        "social_accounts": accounts,
        "source_identity": {
            "website_associated_platforms": platforms,
            "cross_platform_ownership_status": (
                "not_independently_checked"
                if platforms
                else "no_accounts_discovered"
            ),
        },
        "source_history": history,
        "article_verification": {
            "status": "outside_source_assessment",
            "article_truth_verdict": "not_assessed",
        },
        "evidence": {
            "catalog": evidence,
            "extraction_methods": {
                "publisher_name": publisher.get(
                    "name_evidence"
                ),
                "publication_date": (
                    (raw.get("evidence") or {})
                    .get("extraction", {})
                    .get("publication_date_source")
                ),
            },
            "unreviewed_article_links_omitted": True,
        },
        "explanation": explanation,
    }
