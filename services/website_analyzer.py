
import json
import re
from datetime import datetime
from difflib import SequenceMatcher
from urllib.parse import urljoin, urlsplit, urlunsplit

import tldextract
from bs4 import BeautifulSoup

from services.platform_detector import detect_platform


MONTH = (
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|Jan|Feb|Mar|Apr|"
    r"Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)

DATE_PATTERN = re.compile(
    rf"(?:\b\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}\b|"
    rf"\b{MONTH}\s+\d{{1,2}},?\s+\d{{4}}\b|"
    rf"\b\d{{1,2}}\s+{MONTH}\s+\d{{4}}\b|"
    r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b)"
    r"(?:\s*(?:at\s+)?\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?)?",
    re.I
)

DOMAIN_EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=()
)


def clean_text(value):
    if value is None or isinstance(value, (dict, list)):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def get_registered_domain(hostname):
    extracted = DOMAIN_EXTRACTOR(hostname or "")

    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"

    return (hostname or "").lower()


def get_meta(soup, *names):
    for name in names:
        for attribute in ("property", "name", "itemprop"):
            tag = soup.find(
                "meta",
                attrs={
                    attribute: re.compile(
                        rf"^{re.escape(name)}$",
                        re.I
                    )
                }
            )

            if tag:
                value = clean_text(
                    tag.get("content")
                )

                if value:
                    return value

    return ""


def get_schema_nodes(soup):
    nodes = []

    scripts = soup.find_all(
        "script",
        type=re.compile(
            r"application/ld\+json",
            re.I
        )
    )

    for script in scripts:
        try:
            payload = json.loads(
                script.string or script.get_text()
            )
        except (ValueError, TypeError):
            continue

        stack = [payload]

        while stack:
            item = stack.pop()

            if isinstance(item, list):
                stack.extend(reversed(item))

            elif isinstance(item, dict):
                nodes.append(item)

                graph = item.get("@graph")

                if isinstance(graph, (list, dict)):
                    stack.append(graph)

    return nodes


def get_entity_name(value):
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return ""

        return clean_text(value)

    if isinstance(value, dict):
        return get_entity_name(
            value.get("name")
            or value.get("alternateName")
        )

    if isinstance(value, list):
        for item in value:
            name = get_entity_name(item)

            if name:
                return name

    return ""


def schema_types(node):
    types = node.get("@type", [])

    if isinstance(types, str):
        types = [types]

    if isinstance(types, list):
        return {
            str(item).rsplit("/", 1)[-1]
            for item in types
        }

    return set()


def normalize_keywords(value):
    if isinstance(value, str):
        candidates = re.split(r"[,;|]", value)

    elif isinstance(value, list):
        candidates = [
            clean_text(item)
            for item in value
            if isinstance(item, str)
        ]

    else:
        return []

    seen = set()
    result = []

    for item in candidates:
        keyword = clean_text(item)
        key = keyword.casefold()

        if 2 <= len(keyword) <= 80 and key not in seen:
            seen.add(key)
            result.append(keyword)

        if len(result) >= 20:
            break

    return result


def schema_image_url(value):
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        return clean_text(
            value.get("url")
            or value.get("contentUrl")
        ) or None

    if isinstance(value, list):
        for item in value:
            url = schema_image_url(item)

            if url:
                return url

    return None


def get_schema_information(nodes):
    information = {
        "publisher_name": "",
        "publisher_source": "",
        "author_name": "",
        "publication_date": "",
        "modified_date": "",
        "headline": "",
        "article_section": "",
        "keywords": [],
        "article_body": "",
        "image": None
    }

    article_nodes = []
    organization_name = ""

    for node in nodes:
        kinds = schema_types(node)

        if any(
            "Article" in kind
            or kind in {"NewsPosting", "BlogPosting"}
            for kind in kinds
        ):
            article_nodes.append(node)

        if kinds.intersection({
            "Organization",
            "NewsMediaOrganization",
            "MediaOrganization",
            "NewsAgency",
            "WebSite"
        }):
            organization_name = (
                organization_name
                or get_entity_name(node.get("name"))
            )

    article_nodes.sort(
        key=lambda node: (
            0
            if "NewsArticle" in schema_types(node)
            else 1
        )
    )

    for node in article_nodes:
        publisher = get_entity_name(
            node.get("publisher")
        )

        if publisher and not information["publisher_name"]:
            information["publisher_name"] = publisher
            information["publisher_source"] = (
                "jsonld_article_publisher"
            )

        fields = (
            ("author_name", "author"),
            ("publication_date", "datePublished"),
            ("modified_date", "dateModified"),
            ("headline", "headline"),
            ("article_section", "articleSection"),
            ("article_body", "articleBody")
        )

        for key, field in fields:
            if information[key]:
                continue

            if key == "author_name":
                value = get_entity_name(
                    node.get(field)
                )
            else:
                value = clean_text(
                    node.get(field)
                )

            if value:
                information[key] = value

        if not information["keywords"]:
            information["keywords"] = normalize_keywords(
                node.get("keywords")
            )

        if not information["image"]:
            information["image"] = schema_image_url(
                node.get("image")
            )

    if not information["publisher_name"] and organization_name:
        information["publisher_name"] = organization_name
        information["publisher_source"] = (
            "jsonld_organization"
        )

    return information


def parse_published_date(value):
    raw = clean_text(value)

    if not raw:
        return None

    try:
        parsed = datetime.fromisoformat(
            raw.replace("Z", "+00:00")
        )

        if 2000 <= parsed.year <= datetime.now().year + 1:
            is_date_only = bool(
                re.fullmatch(
                    r"\d{4}-\d{2}-\d{2}",
                    raw
                )
            )

            return {
                "raw": raw,
                "iso": (
                    parsed.date().isoformat()
                    if is_date_only
                    else parsed.isoformat()
                )
            }

    except ValueError:
        pass

    match = DATE_PATTERN.search(raw)

    if not match:
        return None

    found = clean_text(
        match.group(0)
    )

    normalized = clean_text(
        re.sub(
            r"\bat\b",
            "",
            found,
            flags=re.I
        )
    )

    normalized = re.sub(
        r"\bSept\b",
        "Sep",
        normalized,
        flags=re.I
    )

    normalized = re.sub(
        r"\s+(am|pm)$",
        lambda match: (
            " " + match.group(1).upper()
        ),
        normalized,
        flags=re.I
    )

    normalized = re.sub(
        r"^(\d{4})[./](\d{1,2})[./](\d{1,2})",
        r"\1-\2-\3",
        normalized
    )

    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d",
        "%B %d, %Y %I:%M %p",
        "%B %d %Y %I:%M %p",
        "%b %d, %Y %I:%M %p",
        "%b %d %Y %I:%M %p",
        "%d %B %Y %I:%M %p",
        "%d %b %Y %I:%M %p",
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y"
    )

    for pattern in formats:
        try:
            parsed = datetime.strptime(
                normalized,
                pattern
            )

            if 2000 <= parsed.year <= datetime.now().year + 1:
                has_time = (
                    "%H" in pattern
                    or "%I" in pattern
                )

                return {
                    "raw": found,
                    "iso": (
                        parsed.isoformat(
                            timespec="seconds"
                        )
                        if has_time
                        else parsed.date().isoformat()
                    )
                }

        except ValueError:
            continue

    return None


def find_article_heading(soup, title):
    headings = soup.find_all(
        ["h1", "h2"],
        limit=30
    )

    if not headings:
        return None

    if title:
        wanted = clean_text(
            title
        ).casefold()

        best = max(
            headings,
            key=lambda item: SequenceMatcher(
                None,
                wanted,
                clean_text(
                    item.get_text(
                        " ",
                        strip=True
                    )
                ).casefold()
            ).ratio()
        )

        match = SequenceMatcher(
            None,
            wanted,
            clean_text(
                best.get_text(
                    " ",
                    strip=True
                )
            ).casefold()
        ).ratio()

        if match >= 0.65:
            return best

    return next(
        (
            item
            for item in headings
            if item.name == "h1"
        ),
        None
    )


def date_near_title(text, title):
    text = clean_text(text)
    title = clean_text(title)

    if not text or not title:
        return None

    position = text.casefold().find(
        title.casefold()
    )

    if position < 0:
        return None

    title_end = position + len(title)

    start = max(
        0,
        position - 180
    )

    end = min(
        len(text),
        title_end + 500
    )

    matches = []

    for match in DATE_PATTERN.finditer(
        text[start:end]
    ):
        absolute_position = (
            start + match.start()
        )

        if absolute_position < position:
            distance = position - (
                absolute_position
                + len(match.group(0))
            )

            if distance < 0 or distance > 180:
                continue

            ranking = distance + 140

        else:
            distance = (
                absolute_position - title_end
            )

            if distance < 0 or distance > 500:
                continue

            ranking = distance

        parsed = parse_published_date(
            match.group(0)
        )

        if parsed:
            matches.append((
                ranking,
                parsed
            ))

    if not matches:
        return None

    return min(
        matches,
        key=lambda item: item[0]
    )[1]


def extract_publication_date(
    soup,
    schema,
    heading,
    article_title=""
):
    candidate = parse_published_date(
        schema.get("publication_date")
    )

    if candidate:
        return (
            candidate,
            "jsonld:datePublished"
        )

    meta_fields = (
        "article:published_time",
        "datePublished",
        "datepublished",
        "pubdate",
        "publishdate",
        "publication_date",
        "citation_date",
        "dc.date",
        "DC.date.issued",
        "parsely-pub-date"
    )

    for field in meta_fields:
        candidate = parse_published_date(
            get_meta(soup, field)
        )

        if candidate:
            return (
                candidate,
                "meta:" + field
            )

    if heading:
        parent = heading

        for _ in range(4):
            parent = parent.parent

            if not parent or parent.name in {
                "body",
                "html",
                "[document]"
            }:
                break

            visible = parent.get_text(
                " ",
                strip=True
            )

            if len(visible) <= 9000:
                candidate = date_near_title(
                    visible,
                    heading.get_text(
                        " ",
                        strip=True
                    )
                )

                if candidate:
                    return (
                        candidate,
                        "visible_text_near_headline"
                    )

    if article_title and soup.body:
        candidate = date_near_title(
            soup.body.get_text(
                " ",
                strip=True
            ),
            article_title
        )

        if candidate:
            return (
                candidate,
                "visible_text_near_article_title"
            )

    if heading:
        parent = heading

        for _ in range(3):
            parent = parent.parent

            if not parent or parent.name in {
                "body",
                "html",
                "[document]"
            }:
                break

            elements = parent.select(
                "time[datetime], "
                "[itemprop='datePublished'], "
                "[class*='published'], "
                "[class*='article-date'], "
                "[class*='news-date']"
            )

            for tag in elements[:15]:
                value = (
                    tag.get("datetime")
                    or tag.get("content")
                    or tag.get("data-published")
                    or tag.get_text(
                        " ",
                        strip=True
                    )
                )

                candidate = parse_published_date(
                    value
                )

                if candidate:
                    return (
                        candidate,
                        "article_date_element"
                    )

    return None, "not_found"


def extract_modified_date(
    soup,
    schema,
    published
):
    fields = [
        (
            schema.get("modified_date"),
            "jsonld:dateModified"
        ),
        (
            get_meta(
                soup,
                "article:modified_time"
            ),
            "meta:article:modified_time"
        ),
        (
            get_meta(
                soup,
                "dateModified",
                "last-modified",
                "modified"
            ),
            "meta:modified"
        )
    ]

    for raw, source in fields:
        parsed = parse_published_date(
            raw
        )

        if parsed and (
            not published
            or parsed["iso"] != published["iso"]
        ):
            return parsed["iso"], source

    return None, "not_found"


def extract_author(
    soup,
    schema,
    heading
):
    if schema.get("author_name"):
        return (
            schema["author_name"],
            "jsonld:author"
        )

    value = get_meta(
        soup,
        "author",
        "article:author",
        "citation_author",
        "parsely-author",
        "dc.creator"
    )

    if value:
        return value, "meta:author"

    if heading:
        parent = heading

        for _ in range(4):
            parent = parent.parent

            if not parent or parent.name in {
                "body",
                "html",
                "[document]"
            }:
                break

            elements = parent.select(
                "[itemprop='author'], "
                "[rel='author'], "
                ".byline, "
                ".author-name, "
                ".post-author"
            )

            for tag in elements[:12]:
                name = clean_text(
                    tag.get("content")
                    or tag.get_text(
                        " ",
                        strip=True
                    )
                )

                name = re.sub(
                    r"^(?:by|written by|කර්තෘ)"
                    r"\s*[:：-]?\s*",
                    "",
                    name,
                    flags=re.I
                )

                if 2 <= len(name) <= 100:
                    return name, "article_byline"

    return None, "not_found"


def detect_article_language(
    soup,
    title,
    description,
    final_url
):
    sample = f"{title} {description}"

    sinhala_count = sum(
        "\u0d80" <= char <= "\u0dff"
        for char in sample
    )

    if sinhala_count >= 3:
        return "si"

    html_tag = soup.find("html")

    declared = clean_text(
        html_tag.get("lang")
        if html_tag
        else ""
    ).lower().split("-")[0]

    if (
        declared == "si"
        or "sinhala." in final_url.lower()
    ):
        return "si"

    if declared == "en":
        return "en"

    sample = soup.get_text(
        " ",
        strip=True
    )[:5000]

    sinhala = sum(
        "\u0d80" <= char <= "\u0dff"
        for char in sample
    )

    latin = sum(
        "a" <= char.lower() <= "z"
        for char in sample
    )

    if (
        sinhala >= 10
        and sinhala >= latin * 0.2
    ):
        return "si"

    return (
        "en"
        if latin >= 10
        else "unknown"
    )


def extract_article_text(
    soup,
    schema,
    heading
):
    schema_body = clean_text(
        schema.get("article_body")
    )

    if len(schema_body) >= 80:
        return (
            schema_body[:12000],
            "jsonld:articleBody"
        )

    selectors = (
        "[itemprop='articleBody']",
        "article",
        ".article-body",
        ".article-content",
        ".story-content",
        ".news-content",
        ".news-story",
        ".news-details",
        ".entry-content",
        ".post-content",
        ".detail-content"
    )

    candidates = []
    seen = set()

    for selector in selectors:
        for node in soup.select(
            selector
        )[:6]:
            if id(node) in seen:
                continue

            seen.add(id(node))

            paragraphs = [
                clean_text(
                    tag.get_text(
                        " ",
                        strip=True
                    )
                )
                for tag in node.find_all("p")
            ]

            meaningful = [
                paragraph
                for paragraph in paragraphs
                if len(paragraph) >= 35
            ]

            text = clean_text(
                " ".join(meaningful)
            )

            if len(text) < 80:
                continue

            link_length = sum(
                len(
                    anchor.get_text(
                        " ",
                        strip=True
                    )
                )
                for anchor in node.find_all("a")
            )

            penalty = min(
                link_length / max(1, len(text)),
                0.8
            )

            candidates.append((
                len(text) * (1 - penalty),
                text
            ))

    if candidates:
        best = max(
            candidates,
            key=lambda item: item[0]
        )[1]

        return (
            best[:12000],
            "article_body_html"
        )

    return None, "not_found"


def extract_image(
    soup,
    schema,
    final_url
):
    candidates = [
        (
            get_meta(
                soup,
                "og:image:secure_url",
                "og:image"
            ),
            "meta:og:image"
        ),
        (
            get_meta(
                soup,
                "twitter:image"
            ),
            "meta:twitter:image"
        ),
        (
            schema.get("image"),
            "jsonld:image"
        )
    ]

    for raw, source in candidates:
        if not raw:
            continue

        url = urljoin(
            final_url,
            raw
        )

        if (
            urlsplit(url).scheme in {
                "http",
                "https"
            }
            and len(url) <= 2048
        ):
            alt = get_meta(
                soup,
                "og:image:alt",
                "twitter:image:alt"
            )

            return (
                url,
                alt or None,
                source
            )

    return None, None, "not_found"


def discover_social_links(
    soup,
    nodes,
    final_url
):
    candidates = [
        (
            tag.get("href"),
            "html_link"
        )
        for tag in soup.find_all(
            "a",
            href=True
        )
    ]

    for node in nodes:
        kinds = schema_types(node)

        if not kinds.intersection({
            "Organization",
            "NewsMediaOrganization",
            "WebSite",
            "MediaOrganization"
        }):
            continue

        same_as = node.get(
            "sameAs",
            []
        )

        if isinstance(same_as, str):
            same_as = [same_as]

        if isinstance(same_as, list):
            candidates.extend(
                (
                    item,
                    "organization_sameAs"
                )
                for item in same_as
                if isinstance(item, str)
            )

    links = {
        "facebook": {},
        "youtube": {}
    }

    for raw, method in candidates:
        if (
            not isinstance(raw, str)
            or not raw.strip()
        ):
            continue

        url = urljoin(
            final_url,
            raw.strip()
        )

        try:
            platform = detect_platform(
                url
            )["platform"]

        except ValueError:
            continue

        if platform not in links:
            continue

        parsed = urlsplit(url)

        parts = [
            part
            for part in parsed.path.split("/")
            if part
        ]

        if not parts:
            continue

        if platform == "facebook":
            excluded = {
                "sharer",
                "sharer.php",
                "share",
                "dialog",
                "plugins",
                "groups",
                "events",
                "watch",
                "reel",
                "reels",
                "stories",
                "photos",
                "photo.php",
                "permalink.php",
                "login",
                "help"
            }

            if parts[0].lower() in excluded:
                continue

            selected = (
                parts[:3]
                if parts[0].lower() in {
                    "pages",
                    "people"
                }
                else parts[:1]
            )

            query = (
                parsed.query
                if parts[0].lower() == "profile.php"
                else ""
            )

        else:
            if parts[0].startswith("@"):
                selected = parts[:1]

            elif (
                parts[0].lower() in {
                    "channel",
                    "user",
                    "c"
                }
                and len(parts) > 1
            ):
                selected = parts[:2]

            else:
                continue

            query = ""

        normalized = urlunsplit((
            "https",
            parsed.hostname or "",
            "/" + "/".join(selected),
            query,
            ""
        ))

        links[platform].setdefault(
            normalized,
            set()
        ).add(method)

    details = []

    for platform in (
        "facebook",
        "youtube"
    ):
        items = list(
            links[platform].items()
        )[:10]

        for index, (_, methods) in enumerate(items):
            details.append({
                "platform": platform,
                "link_index": index,
                "discovery_methods": sorted(methods),
                "ownership_verified": False
            })

    return {
        "facebook_page_urls": list(
            links["facebook"]
        )[:10],
        "youtube_channel_urls": list(
            links["youtube"]
        )[:10],
        "social_link_details": details
    }


def analyze_website(fetch_result):
    if not isinstance(fetch_result, dict):
        raise ValueError(
            "Website fetch result must be a dictionary"
        )

    html = fetch_result.get("html")
    final_url = fetch_result.get("final_url")

    if not html or not final_url:
        raise ValueError(
            "Website HTML and final URL are required"
        )

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    parsed = urlsplit(final_url)

    domain = get_registered_domain(
        parsed.hostname
    )

    nodes = get_schema_nodes(soup)

    schema = get_schema_information(
        nodes
    )

    first_h1 = soup.find("h1")
    title_tag = soup.find("title")

    title = (
        get_meta(
            soup,
            "og:title",
            "twitter:title"
        )
        or schema["headline"]
        or clean_text(
            first_h1.get_text(
                " ",
                strip=True
            )
            if first_h1
            else ""
        )
        or clean_text(
            title_tag.get_text(
                " ",
                strip=True
            )
            if title_tag
            else ""
        )
    )

    heading = find_article_heading(
        soup,
        title
    )

    description = get_meta(
        soup,
        "og:description",
        "description",
        "twitter:description"
    )

    declared_name = ""
    publisher_source = "domain_fallback"

    for field in (
        "og:site_name",
        "application-name",
        "citation_publisher",
        "dc.publisher",
        "publisher"
    ):
        value = get_meta(
            soup,
            field
        )

        if (
            value
            and not value.startswith((
                "https://",
                "http://"
            ))
        ):
            declared_name = value
            publisher_source = "meta:" + field
            break

    if (
        not declared_name
        and schema["publisher_name"]
    ):
        declared_name = schema["publisher_name"]
        publisher_source = schema[
            "publisher_source"
        ]

    source_name = (
        declared_name
        or domain.split(".")[0].replace(
            "-",
            " "
        ).title()
    )

    published, publication_source = (
        extract_publication_date(
            soup,
            schema,
            heading,
            title
        )
    )

    modified, modified_source = (
        extract_modified_date(
            soup,
            schema,
            published
        )
    )

    author, author_source = extract_author(
        soup,
        schema,
        heading
    )

    body, body_source = extract_article_text(
        soup,
        schema,
        heading
    )

    lead_image, image_alt, image_source = (
        extract_image(
            soup,
            schema,
            final_url
        )
    )

    section = (
        schema["article_section"]
        or get_meta(
            soup,
            "article:section",
            "category",
            "section"
        )
        or None
    )

    keywords = (
        schema["keywords"]
        or normalize_keywords(
            get_meta(
                soup,
                "news_keywords",
                "keywords",
                "article:tag"
            )
        )
    )

    canonical_url = None

    canonical_tag = soup.find(
        "link",
        rel="canonical"
    )

    if (
        canonical_tag
        and canonical_tag.get("href")
    ):
        candidate = urljoin(
            final_url,
            canonical_tag.get("href")
        )

        if (
            urlsplit(candidate).scheme in {
                "http",
                "https"
            }
            and get_registered_domain(
                urlsplit(candidate).hostname
            ) == domain
            and candidate.rstrip("/")
            != final_url.rstrip("/")
        ):
            canonical_url = candidate

    social = discover_social_links(
        soup,
        nodes,
        final_url
    )

    source_url = urlunsplit((
        parsed.scheme,
        parsed.netloc,
        "/",
        "",
        ""
    ))

    return {
        "final_url": final_url,
        "canonical_url": canonical_url,
        "platform": "website",
        "source_name": source_name,
        "source_url": source_url,
        "domain": domain,
        "hostname": parsed.hostname,
        "publisher_identity_status": (
            "declared"
            if declared_name
            else "inferred_from_domain"
        ),
        "publisher_identity_source": publisher_source,
        "article_title": title or None,
        "article_language": detect_article_language(
            soup,
            title,
            description,
            final_url
        ),
        "author_name": author,
        "author_source": author_source,
        "publication_date": (
            published["iso"]
            if published
            else None
        ),
        "publication_date_source": publication_source,
        "modified_date": modified,
        "modified_date_source": modified_source,
        "description": description or None,
        "article_section": section,
        "article_keywords": keywords,
        "article_text": body,
        "article_text_source": body_source,
        "article_word_count": (
            len(body.split())
            if body
            else None
        ),
        "lead_image_url": lead_image,
        "lead_image_alt": image_alt,
        "lead_image_source": image_source,
        "facebook_page_urls": social[
            "facebook_page_urls"
        ],
        "youtube_channel_urls": social[
            "youtube_channel_urls"
        ],
        "social_link_details": social[
            "social_link_details"
        ],
        "social_link_status": (
            "website_links_found_not_ownership_verified"
            if (
                social["facebook_page_urls"]
                or social["youtube_channel_urls"]
            )
            else "no_links_found"
        ),
        "http_status": fetch_result.get(
            "http_status"
        ),
        "analysis_status": "information_collected"
    }
