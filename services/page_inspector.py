
import re
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from services.website_fetcher import fetch_website
from services.website_analyzer import clean_text, get_registered_domain


PAGE_KEYWORDS = {
    "contact": (
        "contact",
        "contact-us",
        "contactus",
        "reach-us",
        "සම්බන්ධ",
        "අමතන්න"
    ),
    "about": (
        "about",
        "about-us",
        "aboutus",
        "who-we-are",
        "අප ගැන"
    ),
    "editorial_policy": (
        "editorial",
        "editorial-policy",
        "editorial-standards",
        "ethics",
        "code-of-conduct",
        "මාධ්‍ය ආචාරධර්ම",
        "කර්තෘ ප්‍රතිපත්තිය"
    ),
    "corrections_policy": (
        "corrections",
        "corrections-policy",
        "correction-policy",
        "correcting-errors",
        "නිවැරදි කිරීම්"
    )
}

POLICY_TERMS = {
    "editorial_policy": (
        "editorial policy",
        "editorial standards",
        "editorial guidelines",
        "code of ethics",
        "code of conduct",
        "journalistic standards",
        "ethics policy",
        "මාධ්‍ය ආචාරධර්ම",
        "කර්තෘ ප්‍රතිපත්තිය"
    ),
    "corrections_policy": (
        "corrections policy",
        "correction policy",
        "correct errors",
        "request a correction",
        "report an error",
        "corrections and clarifications",
        "නිවැරදි කිරීම්",
        "දෝෂ නිවැරදි"
    )
}

OWNERSHIP_TERMS = (
    "owned by",
    "ownership",
    "our company",
    "part of the",
    "registered company",
    "හිමිකාර",
    "අයිතිය",
    "සමාගම"
)


def get_homepage_url(url):
    parsed = urlsplit(url)

    return urlunsplit((
        parsed.scheme,
        parsed.netloc,
        "/",
        "",
        ""
    ))


def classify_page_link(url, text):
    path = unquote(
        urlsplit(url).path
    ).lower().rstrip("/").replace("_", "-")

    label = clean_text(text).lower()
    slug = path.rsplit("/", 1)[-1]
    matches = {}

    for category, keywords in PAGE_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            key = keyword.replace(" ", "-")

            if slug == key or label == keyword:
                score = max(score, 5)

            elif (
                slug.startswith(key + "-")
                or slug.endswith("-" + key)
            ):
                score = max(score, 4)

            elif len(keyword) > 6 and keyword in label:
                score = max(score, 2)

        if score:
            matches[category] = score

    return matches


def collect_links(html, page_url, domain):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    results = {
        category: {}
        for category in PAGE_KEYWORDS
    }

    for anchor in soup.find_all("a", href=True):
        absolute = urljoin(
            page_url,
            anchor["href"].strip()
        )

        parsed = urlsplit(absolute)

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
        ):
            continue

        if (
            get_registered_domain(parsed.hostname) != domain
            or len(absolute) > 2048
        ):
            continue

        normalized = urlunsplit((
            parsed.scheme,
            parsed.netloc,
            parsed.path or "/",
            "",
            ""
        ))

        matches = classify_page_link(
            normalized,
            anchor.get_text(" ", strip=True)
        )

        for category, score in matches.items():
            results[category][normalized] = max(
                results[category].get(normalized, 0),
                score
            )

    return results


def visible_content(html):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    for tag in soup.select(
        "script, style, nav, footer, header, noscript, svg, form"
    ):
        tag.decompose()

    root = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    return clean_text(
        root.get_text(" ", strip=True)
    )


def inspect_page(url, domain, category):
    try:
        response = fetch_website(url)
        final_url = response["final_url"]

        if (
            get_registered_domain(
                urlsplit(final_url).hostname
            ) != domain
        ):
            return {
                "status": "redirected_offsite",
                "evidence_url": final_url,
                "content_status": "unknown"
            }

        text = visible_content(
            response["html"]
        )

        lower = text.casefold()

        emails = sorted(set(re.findall(
            r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
            text
        )))[:5]

        phones = sorted(set(re.findall(
            r"(?<!\d)(?:\+94|0)[\d\s()-]{8,15}\d(?!\d)",
            text
        )))[:5]

        markers = {
            "email_present": bool(emails),
            "phone_present": bool(phones),
            "ownership_statement_present": any(
                term in lower
                for term in OWNERSHIP_TERMS
            ),
            "policy_language_present": any(
                term in lower
                for term in POLICY_TERMS.get(category, ())
            )
        }

        if category == "contact":
            substantive = (
                markers["email_present"]
                or markers["phone_present"]
            )

        elif category == "about":
            substantive = len(text) >= 120

        else:
            substantive = (
                len(text) >= 150
                and markers["policy_language_present"]
            )

        return {
            "status": "reachable",
            "http_status": response["http_status"],
            "evidence_url": final_url,
            "content_status": (
                "substantive"
                if substantive
                else "unconfirmed"
            ),
            "content_features": markers,
            "content_excerpt": (
                text[:1200]
                if text
                else None
            )
        }

    except (ValueError, RuntimeError, OSError) as error:
        return {
            "status": "verification_failed",
            "content_status": "unknown",
            "evidence_url": url,
            "error_type": type(error).__name__
        }


def inspect_publisher_pages(fetch_result, max_checks=3):
    if (
        not isinstance(fetch_result, dict)
        or not fetch_result.get("html")
        or not fetch_result.get("final_url")
    ):
        raise ValueError(
            "A completed website fetch result is required"
        )

    if (
        isinstance(max_checks, bool)
        or not isinstance(max_checks, int)
        or not 0 <= max_checks <= 4
    ):
        raise ValueError(
            "max_checks must be between 0 and 4"
        )

    article_url = fetch_result["final_url"]

    domain = get_registered_domain(
        urlsplit(article_url).hostname
    )

    homepage = get_homepage_url(
        article_url
    )

    candidates = collect_links(
        fetch_result["html"],
        article_url,
        domain
    )

    homepage_status = "already_loaded"

    if homepage.rstrip("/") != article_url.rstrip("/"):
        try:
            home = fetch_website(homepage)

            if (
                get_registered_domain(
                    urlsplit(home["final_url"]).hostname
                ) == domain
            ):
                other = collect_links(
                    home["html"],
                    home["final_url"],
                    domain
                )

                for category, links in other.items():
                    for link, score in links.items():
                        candidates[category][link] = max(
                            candidates[category].get(link, 0),
                            score
                        )

                homepage_status = "retrieved"

            else:
                homepage_status = "redirected_offsite"

        except (ValueError, RuntimeError, OSError):
            homepage_status = "fetch_failed"

    page_checks = {}
    discovered = {}
    checked = 0

    for category, links in candidates.items():
        ranked = sorted(
            links,
            key=lambda link: (
                -links[link],
                link
            )
        )

        discovered[category] = ranked[:5]

        if not ranked:
            page_checks[category] = {
                "status": "not_discovered",
                "content_status": "unknown",
                "evidence_url": None
            }

        elif checked >= max_checks:
            page_checks[category] = {
                "status": "discovered_not_checked",
                "content_status": "unknown",
                "evidence_url": ranked[0]
            }

        else:
            page_checks[category] = inspect_page(
                ranked[0],
                domain,
                category
            )

            checked += 1

    return {
        "publisher_domain": domain,
        "homepage_url": homepage,
        "homepage_status": homepage_status,
        "contact_page_urls": discovered["contact"],
        "about_page_urls": discovered["about"],
        "editorial_policy_urls": discovered["editorial_policy"],
        "corrections_policy_urls": discovered["corrections_policy"],
        "page_checks": page_checks,
        "verification_requests": checked,
        "analysis_status": "page_inspection_completed"
    }
