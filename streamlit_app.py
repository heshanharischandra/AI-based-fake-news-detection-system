
import csv
import hashlib
import html
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit, parse_qs
from xml.etree import ElementTree

import requests
import streamlit as st
import tldextract
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

API = os.getenv(
    "SOURCE_API_URL",
    "http://127.0.0.1:8002"
).rstrip("/")

EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=()
)

FB_HOSTS = {
    "facebook.com",
    "www.facebook.com",
    "m.facebook.com",
    "web.facebook.com",
    "mbasic.facebook.com",
    "fb.com",
    "www.fb.com",
}

YT_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
}

INDICATOR_NAMES = {
    "secure_connection": "HTTPS connection",
    "publisher_identification": "Publisher identity",
    "contact_disclosure": "Contact details",
    "organization_information": "About the organization",
    "ownership_declaration": "Ownership disclosure",
    "editorial_policy": "Editorial policy",
    "corrections_policy": "Corrections policy",
    "publisher_social_links": "Social account links",
}

STOPWORDS = {
    "the", "and", "for", "with",
    "from", "that", "this", "news",
    "latest", "breaking", "said",
    "says", "after", "officially",
    "report", "reports", "සහ",
    "සඳහා", "මෙම", "බව",
}

st.set_page_config(
    page_title="SourceLens | Cross-Platform Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
<style>
.stApp {
    background: radial-gradient(
        ellipse at 0% 0%,
        #183a55 0%,
        #0d2237 47%,
        #071522 100%
    );
    color: #eaf6fd;
}
[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    max-width: 1080px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}
.hero {
    border-radius: 24px;
    padding: 35px 40px;
    background: linear-gradient(
        115deg,
        #193d64,
        #156561
    );
    border: 1px solid rgba(169,246,222,.24);
    box-shadow: 0 20px 38px rgba(0,0,0,.12);
}
.eyebrow {
    color: #9affdd;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .15em;
}
.hero h1 {
    color: white;
    font-size: clamp(2.1rem,4vw,3.25rem);
    line-height: 1.12;
    margin: .55rem 0 1rem;
    font-weight: 850;
}
.hero p {
    font-size: 1rem;
    line-height: 1.6;
    color: #d7e9ee;
    max-width: 700px;
    margin: 0;
}
.feature {
    background: rgba(21,51,74,.85);
    border: 1px solid #355d71;
    border-radius: 16px;
    padding: 20px;
    min-height: 135px;
    margin: 14px 0;
}
.feature .symbol {
    font-size: 1.35rem;
    margin-bottom: 8px;
}
.feature strong {
    color: #edfaff;
}
.feature p {
    color: #abc7d6;
    font-size: .88rem;
    line-height: 1.55;
    margin: 7px 0 0;
}
.section-caption {
    color: #a8c2d0;
    font-size: .9rem;
    margin: 5px 0 15px;
}
[data-testid="stForm"] {
    background: #122d44;
    border: 1px solid #37576c;
    border-radius: 18px;
    padding: 20px;
}
[data-testid="stMetric"] {
    background: #16344b;
    border: 1px solid #365c74;
    border-radius: 15px;
    padding: 16px;
}
[data-testid="stMetricValue"] {
    color: #a1f8da;
}
.stButton button,
.stLinkButton a {
    border-radius: 11px !important;
}
.stFormSubmitButton button[kind="primary"] {
    background: linear-gradient(
        100deg,
        #8ff0cc,
        #78d9e3
    );
    color: #062f35;
    font-weight: 800;
    border: none;
}
[data-testid="stExpander"] {
    border-color: #37566e !important;
}
.label {
    color: #93dfcc;
    font-weight: 750;
    letter-spacing: .10em;
    font-size: .75rem;
}
</style>
""",
    unsafe_allow_html=True
)


def clean(value):
    return (
        value.strip()
        if isinstance(value, str)
        else ""
    )


def url_input(value):
    value = clean(value).strip(
        " \t\r\n<>\"'"
    ).rstrip(
        " ,，。.;:!)]}'\""
    )

    if value and "://" not in value:
        value = "https://" + value

    return value


def valid_url(value):
    try:
        parsed = urlsplit(
            clean(value)
        )

        if (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
        ):
            return parsed.geturl()

    except ValueError:
        pass

    return ""


def link(label, value):
    destination = valid_url(value)

    if destination:
        st.link_button(
            label,
            destination
        )


def registered_domain(value):
    try:
        parsed = urlsplit(
            url_input(value)
        )

        host = (
            parsed.hostname or ""
        ).casefold()

        ext = EXTRACT(host)

        return (
            f"{ext.domain}.{ext.suffix}"
            if ext.domain and ext.suffix
            else host
        )

    except ValueError:
        return ""


def classify(raw):
    value = url_input(raw)

    try:
        p = urlsplit(value)

        host = (
            p.hostname or ""
        ).casefold().rstrip(".")

        if (
            p.scheme not in {"http", "https"}
            or not host
            or len(value) > 2048
            or p.username is not None
            or p.password is not None
            or p.port not in {None, 80, 443}
            or re.search(
                r"[\s\\\x00-\x1f]",
                value
            )
        ):
            return "invalid", value

        if host == "fb.watch":
            return "facebook_post", value

        parts = [
            piece.lower()
            for piece in p.path.split("/")
            if piece
        ]

        if host in FB_HOSTS:
            if not parts:
                return "invalid", value

            if (
                parts[0] in {
                    "watch",
                    "reel",
                    "reels",
                    "share",
                    "story.php",
                    "permalink.php",
                    "photo.php",
                }
                or (
                    len(parts) >= 3
                    and parts[1] in {
                        "posts",
                        "photos",
                        "videos",
                        "reels",
                    }
                )
                or (
                    len(parts) >= 4
                    and parts[0] == "groups"
                    and parts[2] == "posts"
                )
            ):
                return "facebook_post", value

            if parts[0] in {
                "groups",
                "events",
                "marketplace",
                "login",
                "help",
            }:
                return "invalid", value

            return "facebook_page", value

        if host in YT_HOSTS:
            if (
                (
                    len(parts) == 1
                    and parts[0].startswith("@")
                )
                or (
                    len(parts) == 2
                    and parts[0] in {
                        "channel",
                        "user",
                        "c",
                    }
                )
            ):
                return "youtube_channel", value

            return "invalid", value

        if (
            host in {"localhost", "youtu.be"}
            or host.endswith(
                (
                    ".local",
                    ".internal",
                    ".test",
                    ".invalid",
                    ".onion",
                )
            )
        ):
            return "invalid", value

        return "website", value

    except ValueError:
        return "invalid", value


def page_from_post(url):
    try:
        p = urlsplit(url)

        parts = [
            part
            for part in p.path.split("/")
            if part
        ]

        if (
            (p.hostname or "").lower() in FB_HOSTS
            and len(parts) >= 3
            and parts[1].lower() in {
                "posts",
                "videos",
                "photos",
                "reels",
            }
            and re.fullmatch(
                r"[A-Za-z0-9._-]{2,110}",
                parts[0]
            )
        ):
            return (
                "https://www.facebook.com/"
                + parts[0]
            )

    except ValueError:
        pass

    return ""


def account_key(url):
    try:
        p = urlsplit(
            url_input(url)
        )

        host = (
            p.hostname or ""
        ).lower()

        parts = [
            part
            for part in p.path.split("/")
            if part
        ]

        if host in FB_HOSTS and parts:
            first = parts[0].casefold()

            if first in {
                "watch",
                "share",
                "reel",
                "reels",
                "story.php",
                "photo.php",
                "permalink.php",
                "groups",
            }:
                return "", ""

            if (
                len(parts) >= 2
                and parts[1].casefold() in {
                    "posts",
                    "photos",
                    "reels",
                    "videos",
                }
            ):
                return "", ""

            if first == "profile.php":
                identity = parse_qs(
                    p.query
                ).get("id", [""])[0]

                return (
                    ("facebook", identity)
                    if identity.isdecimal()
                    else ("", "")
                )

            if (
                first in {"people", "pages"}
                and len(parts) >= 3
            ):
                return (
                    "facebook",
                    parts[-1].casefold()
                )

            return (
                "facebook",
                first
            )

        if host in YT_HOSTS and parts:
            if (
                len(parts) == 1
                and parts[0].startswith("@")
            ):
                return (
                    "youtube",
                    parts[0].casefold()
                )

            if (
                len(parts) == 2
                and parts[0].casefold() in {
                    "channel",
                    "user",
                    "c",
                }
            ):
                return (
                    "youtube",
                    parts[1].casefold()
                )

    except ValueError:
        pass

    return "", ""


def api(
    endpoint,
    payload=None,
    quiet=False,
    timeout=60
):
    try:
        r = requests.request(
            (
                "POST"
                if payload is not None
                else "GET"
            ),
            API + endpoint,
            json=payload,
            timeout=(4, timeout)
        )

        if not r.ok:
            if not quiet:
                st.error(
                    f"Backend returned HTTP "
                    f"{r.status_code}. "
                    "See the Uvicorn terminal for details."
                )

            return None

        result = r.json()

        return (
            result
            if isinstance(result, dict)
            else None
        )

    except (
        requests.RequestException,
        ValueError
    ):
        if not quiet:
            st.error(
                "Cannot contact FastAPI on port 8002. "
                "Start unified_api:app first."
            )

        return None


@st.cache_data(
    ttl=120,
    show_spinner=False
)
def reviewed_pairs():
    path = (
        ROOT
        / "data"
        / "identity_pairs_reviewed.csv"
    )

    if not path.is_file():
        return []

    try:
        with path.open(
            encoding="utf-8-sig",
            newline=""
        ) as stream:
            return list(
                csv.DictReader(stream)
            )

    except (
        OSError,
        csv.Error
    ):
        return []


def relationships(
    website="",
    page=""
):
    wanted_domain = (
        registered_domain(website)
        if website
        else ""
    )

    wanted_key = (
        account_key(page)
        if page
        else ("", "")
    )

    rows = []

    for row in reviewed_pairs():
        if (
            clean(
                row.get("review_status")
            ).casefold() != "reviewed"
            or clean(
                row.get("identity_label")
            ).casefold() != "same_publisher"
            or not valid_url(
                row.get("review_evidence_url")
                or row.get("evidence_url")
            )
        ):
            continue

        rows.append(row)

    selected_ids = {
        clean(
            r.get("publisher_id")
        )
        for r in rows
        if (
            wanted_domain
            and registered_domain(
                r.get("website_url")
            ) == wanted_domain
        )
        or (
            wanted_key[1]
            and account_key(
                r.get("account_url")
            ) == wanted_key
        )
    }

    related = [
        r
        for r in rows
        if clean(
            r.get("publisher_id")
        ) in selected_ids
    ]

    accounts = []
    websites = []

    for row in related:
        p, handle = account_key(
            row.get("account_url")
        )

        if handle:
            accounts.append({
                "platform": p,
                "url": row["account_url"],
                "basis": "Reviewed account association",
                "reviewed": True,
            })

        site = valid_url(
            row.get("website_url")
        )

        if site and site not in websites:
            websites.append(site)

    return accounts, websites, related


def merge_accounts(*groups):
    data = {}

    for group in groups:
        for account in group:
            key = account_key(
                account.get("url")
            )

            if (
                key[1]
                and (
                    key not in data
                    or account.get("reviewed")
                )
            ):
                data[key] = account

    return list(
        data.values()
    )


def website_accounts(report):
    publisher = (
        report.get("publisher")
        or {}
    )

    declared = []

    for account in (
        report.get("social_accounts")
        or []
    ):
        if isinstance(account, dict):
            p, handle = account_key(
                account.get("url")
            )

            if handle:
                declared.append({
                    "platform": p,
                    "url": account["url"],
                    "basis": (
                        "Linked by publisher website"
                    ),
                    "reviewed": False,
                })

    reviewed, _, _ = relationships(
        website=(
            publisher.get("website")
            or publisher.get("source_url")
            or publisher.get("domain")
            or ""
        )
    )

    return merge_accounts(
        declared,
        reviewed
    )


def website_score(report):
    t = (
        report.get("transparency")
        or {}
    )

    checks = (
        t.get("indicators")
        or {}
    )

    if (
        set(checks) != set(INDICATOR_NAMES)
        or t.get("possible_points") != 100
    ):
        return None

    rows = []

    for key, name in INDICATOR_NAMES.items():
        item = (
            checks.get(key)
            or {}
        )

        state = item.get("status")
        maximum = item.get("weight")
        awarded = item.get(
            "observed_points"
        )

        if (
            state not in {
                "observed",
                "publisher_declared",
                "not_established",
            }
            or not isinstance(
                maximum,
                int
            )
        ):
            return None

        expected = (
            maximum
            if state in {
                "observed",
                "publisher_declared",
            }
            else 0
        )

        if awarded != expected:
            return None

        rows.append((
            name,
            awarded,
            maximum,
            state
        ))

    if (
        sum(x[2] for x in rows) != 100
        or sum(x[1] for x in rows)
        != t.get("observed_disclosure_points")
    ):
        return None

    return {
        "score": sum(
            x[1]
            for x in rows
        ),
        "supported": sum(
            x[3] != "not_established"
            for x in rows
        ),
        "rows": rows,
    }


def show_website_assessment(
    report,
    heading="Publisher website evidence"
):
    st.subheader(heading)

    result = website_score(
        report
    )

    if not result:
        st.warning(
            "The website indicator data did not "
            "pass the consistency check."
        )
        return

    first, second, third = st.columns(3)

    first.metric(
        "Documented disclosure points",
        f"{result['score']}/100"
    )

    second.metric(
        "Indicators documented",
        f"{result['supported']}/8"
    )

    third.metric(
        "Pending indicators",
        f"{8-result['supported']}/8"
    )

    st.progress(
        result["score"] / 100
    )

    if (
        result["supported"] < 5
        or result["score"] < 60
    ):
        st.info(
            "This scan has limited evidence. "
            "Unobserved disclosures are not proof "
            "that the publisher is unreliable."
        )

    else:
        st.caption(
            "Publisher disclosure profile, not a "
            "factual-trust or fake-news verdict."
        )

    with st.expander(
        "See score explanation and evidence"
    ):
        for title, gained, maximum, state in result["rows"]:
            st.write(
                f"**{title}** — "
                f"{gained}/{maximum} · "
                f"{state.replace('_', ' ')}"
            )

        explanation = clean(
            report.get("explanation")
        )

        if explanation:
            st.write(
                explanation
            )


def format_time(value):
    try:
        if clean(value):
            return datetime.fromisoformat(
                value.replace(
                    "Z",
                    "+00:00"
                )
            ).strftime(
                "%d %b %Y · %H:%M %Z"
            ).strip()

        return ""

    except (
        ValueError,
        AttributeError
    ):
        return clean(value)


def meaningful_text(
    record,
    title=""
):
    options = [
        clean(
            record.get(field)
        )
        for field in (
            "text",
            "message",
            "body",
            "content",
            "article_text",
            "description",
            "summary",
            "snippet",
        )
    ]

    candidates = [
        v
        for v in options
        if (
            len(v) > 35
            and v.casefold()
            != clean(title).casefold()
        )
    ]

    return (
        max(
            candidates,
            key=len
        )[:2000]
        if candidates
        else ""
    )


def article_panel(
    record,
    source_url,
    publisher="",
    facebook=False
):
    original_title = clean(
        record.get("headline")
        or record.get("title")
        or record.get("article_title")
    )

    title = original_title

    if facebook and (
        title.casefold() == publisher.casefold()
        or re.search(
            r"(?i)\.lk\s+(?:english|sinhala)$",
            title
        )
    ):
        title = ""

    description = meaningful_text(
        record,
        original_title
    )

    stamp = format_time(
        record.get("created_time")
        or record.get("published_at")
        or record.get("publication_date")
    )

    left, right = st.columns(
        [1, 1.5],
        gap="large"
    )

    with left:
        image = valid_url(
            record.get("image_url")
            or record.get("thumbnail_url")
        )

        if image:
            st.image(
                image,
                use_container_width=True
            )

    with right:
        if title:
            st.markdown(
                "#### "
                + html.escape(title)
            )

        if publisher:
            st.caption(publisher)

        st.caption(
            (
                "Posted on Facebook: "
                if facebook
                else "Published: "
            )
            + (
                stamp
                or "Not available from the source"
            )
        )

        if description:
            st.write(
                description
            )

            if description.endswith(
                ("…", "...")
            ):
                st.caption(
                    "Only a public excerpt is accessible."
                )

        else:
            st.caption(
                "A separate article description was "
                "not available from the source."
            )

        link(
            (
                "Open original Facebook post"
                if facebook
                else "Read original article"
            ),
            source_url
        )

    return title, description


def show_connections(
    accounts,
    website="",
    confirmed=False
):
    st.subheader(
        "Connected publisher accounts"
    )

    if website:
        link(
            (
                "Publisher website"
                if confirmed
                else "Candidate publisher website"
            ),
            website
        )

    if not accounts:
        st.caption(
            "No Facebook or YouTube accounts with "
            "accessible association evidence were found."
        )

    for item in accounts:
        col1, col2 = st.columns(
            [3, 1]
        )

        with col1:
            st.write(
                "**"
                + (
                    "Facebook Page"
                    if item["platform"] == "facebook"
                    else "YouTube channel"
                )
                + "**"
            )

            st.caption(
                item.get("basis")
                or "Candidate account"
            )

        with col2:
            link(
                "Open account",
                item.get("url")
            )


def show_page_assessment(
    result,
    page_url
):
    st.subheader(
        "Facebook Page transparency"
    )

    if not result:
        st.info(
            "Page-assessment service is not available. "
            "Confirm /facebook-page-score is registered "
            "in FastAPI."
        )
        return

    score = result.get("score")
    points = result.get(
        "observed_points",
        0
    )

    checked = result.get(
        "checked_indicators",
        0
    )

    if isinstance(
        score,
        (float, int)
    ):
        a, b, c = st.columns(3)

        a.metric(
            "Page disclosure points",
            f"{score}/100"
        )

        b.metric(
            "Indicators reviewed",
            f"{checked}/8"
        )

        c.metric(
            "Evidence coverage",
            f"{result.get('assessed_weight', 0)}/100"
        )

        st.progress(
            max(
                0,
                min(
                    score / 100,
                    1
                )
            )
        )

        st.caption(
            "Provisional Page-profile evidence score, "
            "not a factual-reliability score."
        )

    else:
        st.info(
            "Page score pending: "
            f"{checked}/8 indicators checked; "
            f"{result.get('assessed_weight', 0)}/100 "
            "indicator weight assessed. "
            "Document at least four indicators "
            "covering 60 points."
        )

        if points:
            st.caption(
                f"{points} observed positive disclosure "
                "points, not yet a reportable score."
            )

    if result.get("page_name"):
        st.markdown(
            "**"
            + html.escape(
                str(
                    result["page_name"]
                )
            )
            + "**"
        )

    if result.get("about"):
        st.write(
            result["about"]
        )

    with st.expander(
        "Page score methodology and evidence"
    ):
        for key, item in (
            result.get("indicators")
            or {}
        ).items():
            st.write(
                f"**{item.get('label', key)}** — "
                f"{item.get('awarded_points', 0)}/"
                f"{item.get('weight', 0)} · "
                f"{item.get('status', 'not_checked').replace('_', ' ')}"
            )

            if item.get("evidence_url"):
                link(
                    "Review evidence: "
                    + key.replace(
                        "_",
                        " "
                    ),
                    item["evidence_url"]
                )

        st.caption(
            "Unchecked indicators are not assumed to "
            "be absent. Manual evidence requires a "
            "dated review and source URL."
        )


def history_record(site):
    name = registered_domain(
        site
    )

    if not name:
        return None

    data = api(
        "/history/"
        + quote(name, safe="")
        + "?limit=20",
        quiet=True,
        timeout=18
    )

    for record in (
        (data or {}).get(
            "assessments",
            []
        )
    ):
        if (
            isinstance(record, dict)
            and website_score(record)
        ):
            return record

    return None


def show_linked_publisher(
    site,
    confirmed
):
    if not site:
        return

    st.subheader(
        "Linked publisher assessment"
    )

    st.caption(
        "Human-reviewed account association"
        if confirmed
        else "Website suggested by Facebook preview; "
             "Page ownership is not verified"
    )

    state_key = (
        "site_assessment_"
        + hashlib.sha256(
            site.encode()
        ).hexdigest()[:12]
    )

    record = (
        st.session_state.get(
            state_key
        )
        or history_record(site)
    )

    if not record and st.button(
        "Analyze publisher website",
        key="fetch_" + state_key
    ):
        record = api(
            "/assess",
            {
                "url": site,
                "output_language": "en",
            },
            timeout=90
        )

        if record:
            st.session_state[state_key] = record

    if record:
        show_website_assessment(
            record,
            "Website disclosure score "
            "(separate from Facebook)"
        )

    else:
        st.caption(
            "No saved website assessment was found. "
            "Use Analyze publisher website "
            "to collect one."
        )


def run_identity(
    account,
    publisher,
    site
):
    if not publisher or not valid_url(site):
        return

    key = (
        "ml_"
        + hashlib.sha256(
            account["url"].encode()
        ).hexdigest()[:12]
    )

    if st.button(
        "Check account identity",
        key="go_" + key
    ):
        st.session_state[key] = api(
            "/predict-identity",
            {
                "publisher_name": publisher,
                "website_url": site,
                "account_url": account["url"],
            },
            timeout=30
        )

    result = st.session_state.get(
        key
    )

    if not result:
        return

    label = (
        result.get("prediction")
        or {}
    ).get(
        "identity_label"
    ) or "unknown"

    st.info(
        "Experimental prediction: "
        + label.replace(
            "_",
            " "
        ).title()
    )

    features = (
        result.get("feature_explanation")
        or {}
    )

    one, two = st.columns(2)

    for column, title, field in (
        (
            one,
            "Publisher/handle similarity",
            "name_handle_similarity",
        ),
        (
            two,
            "Domain/handle similarity",
            "domain_handle_similarity",
        )
    ):
        val = features.get(field)

        column.metric(
            title,
            (
                f"{val:.3f}"
                if isinstance(
                    val,
                    (int, float)
                )
                else "N/A"
            )
        )

    st.caption(
        "Similarity is not an ownership probability "
        "or an accuracy verdict. The ML pilot "
        "requires independent validation."
    )


def keywords(value):
    return [
        word
        for word in re.findall(
            r"[\w\u0d80-\u0dff\u200c\u200d]+",
            clean(value).casefold()
        )
        if (
            len(word) >= 3
            and word not in STOPWORDS
        )
    ]


def topic(
    title,
    description="",
    publisher=""
):
    first = keywords(title)

    generic = (
        len(first) < 4
        or (
            keywords(publisher)
            and set(first).issubset(
                set(
                    keywords(publisher)
                )
                | {
                    "english",
                    "sinhala",
                    "page",
                }
            )
        )
    )

    material = (
        description
        if (
            generic
            and len(
                keywords(description)
            ) >= 4
        )
        else title or description
    )

    return " ".join(
        keywords(material)[:12]
    )[:155]


@st.cache_data(
    ttl=600,
    show_spinner=False
)
def news_search(
    title,
    description="",
    publisher=""
):
    query = topic(
        title,
        description,
        publisher
    )

    reference = set(
        keywords(query)
    )

    if len(reference) < 2:
        return []

    variants = list(
        dict.fromkeys([
            query,
            " ".join(
                keywords(query)[:7]
            ),
            " ".join(
                keywords(query)[:5]
            ),
        ])
    )

    lang = (
        "si"
        if re.search(
            r"[\u0d80-\u0dff]",
            query
        )
        else "en"
    )

    found = {}

    for term in variants:
        if not term:
            continue

        try:
            response = requests.get(
                "https://news.google.com/rss/search",
                params={
                    "q": term,
                    "hl": lang,
                    "gl": "LK",
                    "ceid": "LK:" + lang,
                },
                headers={
                    "User-Agent": "Mozilla/5.0"
                },
                timeout=(4, 9)
            )

            response.raise_for_status()

            if len(
                response.content
            ) > 1500000:
                continue

            root = ElementTree.fromstring(
                response.content
            )

            for item in root.findall(
                "./channel/item"
            )[:40]:
                heading = clean(
                    item.findtext("title")
                )

                destination = valid_url(
                    item.findtext("link")
                )

                shared = len(
                    reference
                    & set(
                        keywords(heading)
                    )
                )

                if (
                    not destination
                    or shared < 2
                ):
                    continue

                similarity = round(
                    100
                    * shared
                    / len(reference)
                )

                source = item.find(
                    "source"
                )

                candidate = {
                    "title": heading,
                    "url": destination,
                    "publisher": (
                        clean(source.text)
                        if source is not None
                        else ""
                    ),
                    "overlap": similarity,
                }

                if (
                    destination not in found
                    or found[destination]["overlap"]
                    < similarity
                ):
                    found[destination] = candidate

        except (
            requests.RequestException,
            ElementTree.ParseError,
            ValueError
        ):
            continue

    return sorted(
        found.values(),
        key=lambda r: r["overlap"],
        reverse=True
    )[:8]


def show_related(
    title,
    description,
    accounts,
    publisher=""
):
    st.subheader(
        "Related reporting across platforms"
    )

    query = topic(
        title,
        description,
        publisher
    )

    if len(
        keywords(query)
    ) < 2:
        st.caption(
            "A specific news headline or substantive "
            "description is needed for story comparison."
        )

        return

    tab_web, tab_fb, tab_youtube = st.tabs([
        "News websites",
        "Facebook",
        "YouTube",
    ])

    with tab_web:
        matches = news_search(
            title,
            description,
            publisher
        )

        if not matches:
            st.info(
                "No candidate indexed news reports "
                "found for this query."
            )

        for item in matches:
            with st.container(
                border=True
            ):
                st.write(
                    "**"
                    + item["title"]
                    + "**"
                )

                st.caption(
                    f"{item['publisher']} · "
                    f"{item['overlap']}% keyword overlap "
                    "· Candidate, not fact-checked"
                )

                link(
                    "Read report",
                    item["url"]
                )

        link(
            "Search Google News",
            "https://news.google.com/search?"
            + urlencode({
                "q": query
            })
        )

    with tab_fb:
        pages = [
            a
            for a in accounts
            if a["platform"] == "facebook"
        ]

        if not pages:
            st.caption(
                "No Facebook Page association is available."
            )

        for page in pages:
            link(
                "Open publisher Page",
                page["url"]
            )

            handle = account_key(
                page["url"]
            )[1]

            search = (
                f"site:facebook.com/{handle} {query}"
                if handle and not handle.isdecimal()
                else f"site:facebook.com {query}"
            )

            link(
                "Search public indexed posts",
                "https://www.google.com/search?"
                + urlencode({
                    "q": search
                })
            )

    with tab_youtube:
        channels = [
            a
            for a in accounts
            if a["platform"] == "youtube"
        ]

        for channel in channels:
            link(
                "Open associated channel",
                channel["url"]
            )

            handle = account_key(
                channel["url"]
            )[1]

            link(
                "Search channel videos",
                "https://www.youtube.com/results?"
                + urlencode({
                    "search_query": (
                        handle + " " + query
                    )
                })
            )

        link(
            "Search YouTube",
            "https://www.youtube.com/results?"
            + urlencode({
                "search_query": query
            })
        )


def reset():
    st.session_state["source_url"] = ""

    for key in list(
        st.session_state
    ):
        if (
            key == "current"
            or key == "manual_headline"
            or key.startswith((
                "ml_",
                "site_assessment_",
            ))
        ):
            st.session_state.pop(
                key,
                None
            )


st.markdown(
    """
<section class="hero">
    <span class="eyebrow">
        EVIDENCE-BASED NEWS SOURCE INTELLIGENCE
    </span>
    <h1>
        Cross-Platform
        <br>
        <span style="color:#a6ffdf">
            Source Verification
        </span>
    </h1>
    <p>
        Understand who publishes the news.
        Compare website disclosures, linked Facebook
        and YouTube accounts, and reports from other
        publishers—all in one place.
    </p>
</section>
""",
    unsafe_allow_html=True
)

feat = st.columns(
    3,
    gap="medium"
)

for col, icon, name, description in zip(
    feat,
    [
        "◈",
        "◎",
        "≋",
    ],
    [
        "Source transparency",
        "Platform connections",
        "Related reporting",
    ],
    [
        "Documented publisher information and evidence",
        "Account links and experimental identity matching",
        "Find candidate reports about the same event",
    ],
):
    with col:
        st.markdown(
            '<div class="feature">'
            '<div class="symbol">'
            + icon
            + "</div><strong>"
            + name
            + "</strong><p>"
            + description
            + "</p></div>",
            unsafe_allow_html=True
        )

st.markdown(
    "#### Check a source"
)

st.markdown(
    '<p class="section-caption">'
    "Paste a news website article, Facebook Page "
    "or post, or YouTube channel URL."
    "</p>",
    unsafe_allow_html=True
)

with st.form(
    "input_form"
):
    st.text_input(
        "News source URL",
        key="source_url",
        placeholder=(
            "https://example-news-website.com/article"
        )
    )

    clicked = st.form_submit_button(
        "Verify source",
        type="primary",
        use_container_width=True
    )

st.button(
    "Clear / Check another URL",
    use_container_width=True,
    on_click=reset
)

if clicked:
    st.session_state.pop(
        "current",
        None
    )

    kind, value = classify(
        st.session_state.get(
            "source_url",
            ""
        )
    )

    if kind == "invalid":
        st.error(
            "Enter a valid public news website, "
            "Facebook Page/post or YouTube channel URL."
        )

    else:
        with st.spinner(
            "Collecting available evidence..."
        ):
            if kind == "website":
                data = api(
                    "/assess",
                    {
                        "url": value,
                        "output_language": "en",
                    },
                    timeout=95
                )

            elif kind == "facebook_post":
                data = api(
                    "/facebook-assess",
                    {
                        "url": value,
                        "output_language": "en",
                    },
                    timeout=45
                )

            else:
                data = (
                    api(
                        "/social-source",
                        {
                            "url": value
                        },
                        quiet=True,
                        timeout=35
                    )
                    or {}
                )

            page = (
                page_from_post(value)
                if kind == "facebook_post"
                else (
                    value
                    if kind == "facebook_page"
                    else ""
                )
            )

            fb_profile = (
                api(
                    "/facebook-page-score",
                    {
                        "url": page
                    },
                    quiet=True,
                    timeout=25
                )
                if page
                else None
            )

            metadata = (
                api(
                    "/facebook-post-metadata",
                    {
                        "url": value
                    },
                    quiet=True,
                    timeout=20
                )
                if kind == "facebook_post"
                else None
            )

        if data is not None:
            st.session_state["current"] = {
                "kind": kind,
                "url": value,
                "data": data,
                "page": page,
                "fb_profile": fb_profile,
                "metadata": metadata,
            }


current = st.session_state.get(
    "current"
)

if current:
    st.divider()

    kind = current["kind"]
    value = current["url"]
    data = current["data"]

    st.markdown(
        '<span class="label">'
        "SOURCE VERIFICATION REPORT"
        "</span>",
        unsafe_allow_html=True
    )

    if kind == "website":
        publisher = (
            data.get("publisher")
            or {}
        )

        article = (
            data.get("article")
            or {}
        )

        name = clean(
            publisher.get("name")
        )

        site = valid_url(
            publisher.get("website")
            or publisher.get("source_url")
        )

        st.subheader(
            "Article and publisher"
        )

        title, description = article_panel(
            article,
            article.get("url")
            or value,
            name
        )

        show_website_assessment(
            data
        )

        accounts = website_accounts(
            data
        )

        show_connections(
            accounts,
            site,
            confirmed=True
        )

        for item in accounts:
            with st.expander(
                (
                    "Facebook"
                    if item["platform"] == "facebook"
                    else "YouTube"
                )
                + " · Publisher identity matching"
            ):
                run_identity(
                    item,
                    name,
                    site or value
                )

        show_related(
            title,
            description,
            accounts,
            name
        )

    elif kind == "facebook_post":
        publisher = (
            data.get("publisher")
            or {}
        )

        post = dict(
            data.get("post")
            or {}
        )

        meta = (
            current.get("metadata")
            or {}
        )

        if meta.get("status") in {
            "retrieved",
            "metadata_retrieved",
        }:
            for field, key in (
                (
                    "created_time",
                    "created_time"
                ),
                (
                    "message",
                    "text"
                ),
            ):
                if clean(
                    meta.get(field)
                ):
                    post[key] = meta[field]

        name = clean(
            publisher.get("name")
        )

        st.subheader(
            "Facebook post overview"
        )

        title, description = article_panel(
            post,
            value,
            name,
            facebook=True
        )

        page = current.get(
            "page",
            ""
        )

        accounts, sites, matches = relationships(
            page=page,
            website=(
                publisher.get("website_url")
                or ""
            )
        )

        if page:
            accounts = merge_accounts(
                [{
                    "platform": "facebook",
                    "url": page,
                    "basis": (
                        "Page inferred from submitted post URL"
                    ),
                    "reviewed": False,
                }],
                accounts
            )

        associated_site = (
            sites[0]
            if sites
            else valid_url(
                publisher.get("website_url")
            )
        )

        show_connections(
            accounts,
            associated_site,
            confirmed=bool(matches)
        )

        if page:
            show_page_assessment(
                current.get(
                    "fb_profile"
                ),
                page
            )

        else:
            st.info(
                "The submitted post URL does not "
                "expose an identifiable Page handle."
            )

        if associated_site:
            show_linked_publisher(
                associated_site,
                bool(matches)
            )

        if len(
            keywords(description)
        ) < 3:
            st.text_input(
                "Headline to search across platforms",
                key="manual_headline",
                placeholder=(
                    "Enter actual news headline"
                )
            )

        show_related(
            title,
            (
                st.session_state.get(
                    "manual_headline"
                )
                or description
            ),
            accounts,
            name
        )

    else:
        page = (
            value
            if kind == "facebook_page"
            else ""
        )

        accounts, sites, matches = relationships(
            page=value
        )

        accounts = merge_accounts(
            [{
                "platform": (
                    "facebook"
                    if kind == "facebook_page"
                    else "youtube"
                ),
                "url": value,
                "basis": "Submitted account",
                "reviewed": False,
            }],
            accounts
        )

        st.subheader(
            "Account overview"
        )

        link(
            "Open submitted account",
            value
        )

        if page:
            show_page_assessment(
                current.get(
                    "fb_profile"
                ),
                page
            )

        site = (
            sites[0]
            if sites
            else ""
        )

        show_connections(
            accounts,
            site,
            confirmed=bool(matches)
        )

        if site:
            show_linked_publisher(
                site,
                bool(matches)
            )

        if matches and site:
            with st.expander(
                "Experimental publisher identity matching"
            ):
                run_identity(
                    accounts[0],
                    clean(
                        matches[0].get(
                            "publisher_name"
                        )
                    ),
                    site
                )

        st.text_input(
            "News headline to search across platforms",
            key="manual_headline",
            placeholder=(
                "Enter a specific story headline"
            )
        )

        if clean(
            st.session_state.get(
                "manual_headline"
            )
        ):
            show_related(
                st.session_state["manual_headline"],
                "",
                accounts
            )
