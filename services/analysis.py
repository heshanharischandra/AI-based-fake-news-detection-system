
import ipaddress
import json
import re
import socket

from datetime import datetime
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
import tldextract

from bs4 import BeautifulSoup


_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=()
)

_MONTH = (
    r"(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)

_DATE = re.compile(
    rf"(?:\b\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}\b|"
    rf"\b{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}\b|"
    rf"\b\d{{1,2}}\s+{_MONTH}\s+\d{{4}}\b|"
    rf"\b\d{{1,2}}[-/]\d{{1,2}}[-/]\d{{4}}\b)"
    rf"(?:\s*(?:at\s+)?\d{{1,2}}:\d{{2}}"
    rf"(?::\d{{2}})?\s*(?:am|pm)?)?",
    re.I
)

_WEIGHTS = {
    "https_used": 5,
    "publisher_identified": 15,
    "author_disclosed": 5,
    "publication_date": 5,
    "contact_details": 15,
    "about_page": 10,
    "editorial_policy": 15,
    "corrections_policy": 10,
    "ownership_disclosure": 10,
    "social_links": 5,
    "article_references": 5
}

_KEYWORDS = {
    "contact": (
        "contact-us",
        "contactus",
        "contact",
        "reach-us",
        "අමතන්න",
        "සම්බන්ධ"
    ),
    "about": (
        "about-us",
        "aboutus",
        "about",
        "who-we-are",
        "අප ගැන"
    ),
    "editorial": (
        "editorial-policy",
        "editorial",
        "code-of-ethics",
        "journalism-ethics",
        "මාධ්‍ය ආචාරධර්ම"
    ),
    "corrections": (
        "corrections-policy",
        "corrections",
        "correcting-errors",
        "නිවැරදි කිරීම්"
    )
}


def clean(value):
    return re.sub(
        r"\s+",
        " ",
        str(value or "")
    ).strip()


def domain_name(host):
    ext = _EXTRACT(host or "")

    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"

    return (host or "").lower()


def validate_url(url):
    if (
        not isinstance(url, str)
        or not 8 <= len(url.strip()) <= 2048
        or re.search(r"[\s\\\x00-\x1f]", url)
    ):
        raise ValueError("Provide a valid public news URL")

    value = (
        url.strip()
        if "://" in url
        else "https://" + url.strip()
    )

    parsed = urlsplit(value)

    if (
        parsed.scheme not in ("https", "http")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "Only public HTTP/HTTPS URLs are supported"
        )

    host = parsed.hostname.lower()

    if (
        host == "localhost"
        or host.endswith((
            ".local",
            ".localhost",
            ".internal",
            ".test",
            ".invalid",
            ".onion"
        ))
    ):
        raise ValueError("Local addresses are not supported")

    try:
        ipaddress.ip_address(host)
        raise ValueError("Direct IP addresses are not allowed")
    except ValueError as exc:
        if str(exc) == "Direct IP addresses are not allowed":
            raise

    if not re.fullmatch(
        r"[a-z0-9-]+(?:\.[a-z0-9-]+)+",
        host
    ):
        raise ValueError("Invalid domain name")

    if (
        parsed.port is not None
        and parsed.port not in (80, 443)
    ):
        raise ValueError("Unsupported port")

    port = parsed.port or (
        443 if parsed.scheme == "https" else 80
    )

    try:
        addresses = socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise ValueError(
            "Host DNS resolution failed"
        ) from exc

    if (
        not addresses
        or any(
            not ipaddress.ip_address(
                item[4][0]
            ).is_global
            for item in addresses
        )
    ):
        raise ValueError(
            "Host resolves to a restricted IP"
        )

    return urlunsplit((
        parsed.scheme,
        parsed.netloc,
        parsed.path or "/",
        parsed.query,
        ""
    ))


def fetch_html(url):
    current = url

    with requests.Session() as session:
        session.trust_env = False

        for _ in range(6):
            current = validate_url(current)

            try:
                response = session.get(
                    current,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 "
                            "(compatible; AcademicSourceResearch/1.0)"
                        ),
                        "Accept": (
                            "text/html,application/xhtml+xml"
                        ),
                        "Accept-Language": "si,en;q=0.8"
                    },
                    timeout=(5, 12),
                    allow_redirects=False,
                    stream=True
                )
            except requests.RequestException as exc:
                raise RuntimeError(
                    "Website request failed: "
                    + type(exc).__name__
                ) from exc

            with response:
                if response.status_code in (
                    301, 302, 303, 307, 308
                ):
                    target = response.headers.get(
                        "Location"
                    )

                    if not target:
                        raise RuntimeError(
                            "Missing redirect location"
                        )

                    current = urljoin(
                        current,
                        target
                    )

                    continue

                if response.status_code >= 400:
                    raise RuntimeError(
                        f"Website returned HTTP {response.status_code}"
                    )

                kind = response.headers.get(
                    "Content-Type",
                    ""
                ).lower()

                if kind and "html" not in kind:
                    raise RuntimeError(
                        "URL did not return HTML"
                    )

                chunks = []
                size = 0

                try:
                    for chunk in response.iter_content(65536):
                        size += len(chunk)

                        if size > 4000000:
                            raise RuntimeError(
                                "Website HTML exceeds 4 MB"
                            )

                        chunks.append(chunk)

                except requests.RequestException as exc:
                    raise RuntimeError(
                        "HTML download was interrupted"
                    ) from exc

                raw = b"".join(chunks)

                encoding = response.encoding

                if (
                    not encoding
                    or encoding.lower() in (
                        "iso-8859-1",
                        "latin-1"
                    )
                ):
                    encoding = "utf-8"

                return {
                    "url": current,
                    "html": raw.decode(
                        encoding,
                        errors="replace"
                    ),
                    "status": response.status_code
                }

    raise RuntimeError("Too many redirects")


def meta(soup, *names):
    for name in names:
        for attr in ("property", "name", "itemprop"):
            node = soup.find(
                "meta",
                attrs={
                    attr: re.compile(
                        "^" + re.escape(name) + "$",
                        re.I
                    )
                }
            )

            if node and clean(node.get("content")):
                return clean(node["content"])

    return None


def jsonld(soup):
    items = []

    scripts = soup.find_all(
        "script",
        type=re.compile(
            r"application/ld\+json",
            re.I
        )
    )

    for tag in scripts:
        try:
            stack = [
                json.loads(
                    tag.string or tag.get_text()
                )
            ]
        except (ValueError, TypeError):
            continue

        while stack:
            obj = stack.pop()

            if isinstance(obj, list):
                stack.extend(obj)

            elif isinstance(obj, dict):
                items.append(obj)

                if isinstance(
                    obj.get("@graph"),
                    (dict, list)
                ):
                    stack.append(obj["@graph"])

    return items


def entity_name(value):
    if isinstance(value, str):
        return (
            value.strip()
            if not value.startswith(("http:", "https:"))
            else None
        )

    if isinstance(value, dict):
        return entity_name(
            value.get("name")
            or value.get("alternateName")
        )

    if isinstance(value, list):
        for item in value:
            name = entity_name(item)

            if name:
                return name

    return None


def as_date(value):
    raw = clean(value)

    if not raw:
        return None

    try:
        parsed = datetime.fromisoformat(
            raw.replace("Z", "+00:00")
        )

        if 2000 <= parsed.year <= datetime.now().year + 1:
            if re.fullmatch(
                r"\d{4}-\d{2}-\d{2}",
                raw
            ):
                return parsed.date().isoformat()

            return parsed.isoformat()

    except ValueError:
        pass

    match = _DATE.search(raw)

    if not match:
        return None

    text = re.sub(
        r"\bat\b",
        "",
        match.group(0),
        flags=re.I
    )

    text = re.sub(
        r"\bSept\b",
        "Sep",
        text,
        flags=re.I
    )

    text = clean(text)

    text = re.sub(
        r"\s+(am|pm)$",
        lambda m: " " + m.group(1).upper(),
        text,
        flags=re.I
    )

    text = re.sub(
        r"^(\d{4})[./](\d{1,2})[./](\d{1,2})",
        r"\1-\2-\3",
        text
    )

    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %I:%M %p",
        "%Y-%m-%d",
        "%b %d, %Y %I:%M %p",
        "%b %d %Y %I:%M %p",
        "%B %d, %Y %I:%M %p",
        "%B %d %Y %I:%M %p",
        "%b %d, %Y",
        "%b %d %Y",
        "%B %d, %Y",
        "%B %d %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%d/%m/%Y",
        "%d-%m-%Y"
    )

    for fmt in formats:
        try:
            parsed = datetime.strptime(
                text,
                fmt
            )

            if 2000 <= parsed.year <= datetime.now().year + 1:
                if "%H" in fmt or "%I" in fmt:
                    return parsed.isoformat(
                        timespec="seconds"
                    )

                return parsed.date().isoformat()

        except ValueError:
            continue

    return None


def date_near_headline(soup, title):
    if not title or not soup.body:
        return None

    full = clean(
        soup.body.get_text(
            " ",
            strip=True
        )
    )

    start = full.casefold().find(
        title.casefold()
    )

    if start < 0:
        return None

    end = start + len(title)

    offset = max(
        0,
        start - 180
    )

    fragment = full[
        offset:end + 450
    ]

    candidates = []

    for match in _DATE.finditer(fragment):
        position = offset + match.start()

        if position >= end:
            distance = position - end
        else:
            distance = (
                start
                - (position + len(match.group(0)))
                + 150
            )

        if 0 <= distance < 500:
            parsed = as_date(
                match.group(0)
            )

            if parsed:
                candidates.append((
                    distance,
                    parsed
                ))

    return (
        min(candidates)[1]
        if candidates
        else None
    )


def social_profiles(soup, nodes, article_url):
    links = []

    sources = [
        (
            anchor.get("href"),
            "html_link"
        )
        for anchor in soup.find_all(
            "a",
            href=True
        )
    ]

    for node in nodes:
        kinds = node.get(
            "@type",
            []
        )

        if isinstance(kinds, str):
            kinds = [kinds]

        if (
            not isinstance(kinds, list)
            or not any(
                str(kind).split("/")[-1] in (
                    "Organization",
                    "NewsMediaOrganization",
                    "WebSite"
                )
                for kind in kinds
            )
        ):
            continue

        same_as = node.get(
            "sameAs",
            []
        )

        if isinstance(same_as, str):
            same_as = [same_as]

        if isinstance(same_as, list):
            sources.extend(
                (url, "organization_sameAs")
                for url in same_as
                if isinstance(url, str)
            )

    seen = set()

    for original, origin in sources:
        if not isinstance(original, str):
            continue

        parsed = urlsplit(
            urljoin(
                article_url,
                original
            )
        )

        host = (
            parsed.hostname or ""
        ).lower()

        path = [
            part
            for part in parsed.path.split("/")
            if part
        ]

        platform = None

        if host in (
            "facebook.com",
            "www.facebook.com",
            "m.facebook.com"
        ):
            platform = "facebook"

        elif host in (
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com"
        ):
            platform = "youtube"

        if not platform or not path:
            continue

        if platform == "facebook":
            excluded = (
                "share",
                "sharer",
                "sharer.php",
                "watch",
                "reel",
                "reels",
                "login",
                "groups",
                "events",
                "photo.php",
                "permalink.php"
            )

            if path[0].lower() in excluded:
                continue

            if (
                path[0].lower() in ("pages", "people")
                and len(path) >= 3
            ):
                keep = path[:3]
            else:
                keep = path[:1]

        else:
            if path[0].startswith("@"):
                keep = path[:1]

            elif (
                path[0].lower() in (
                    "channel",
                    "user",
                    "c"
                )
                and len(path) >= 2
            ):
                keep = path[:2]

            else:
                continue

        url = (
            f"https://{host}/"
            + "/".join(keep)
        )

        if url in seen:
            continue

        seen.add(url)

        links.append({
            "platform": platform,
            "url": url,
            "relationship": "publisher_website_link",
            "ownership_status": "not_independently_checked",
            "evidence_method": origin
        })

        if len(links) >= 12:
            break

    return links


def analyze_article(page):
    soup = BeautifulSoup(
        page["html"],
        "html.parser"
    )

    nodes = jsonld(soup)
    article_nodes = []

    for node in nodes:
        types = node.get(
            "@type",
            []
        )

        if isinstance(types, str):
            types = [types]

        if (
            isinstance(types, list)
            and any(
                "Article" in str(kind)
                or "NewsPosting" in str(kind)
                for kind in types
            )
        ):
            article_nodes.append(node)

    schema = (
        article_nodes[0]
        if article_nodes
        else {}
    )

    headline_node = soup.find("h1")

    title = (
        meta(
            soup,
            "og:title",
            "twitter:title"
        )
        or clean(schema.get("headline"))
        or (
            clean(
                headline_node.get_text(
                    " ",
                    strip=True
                )
            )
            if headline_node
            else None
        )
    )

    if not title:
        tag = soup.find("title")

        title = (
            clean(tag.get_text(" ", strip=True))
            if tag
            else None
        )

    description = meta(
        soup,
        "og:description",
        "description",
        "twitter:description"
    )

    organization_name = None

    for item in nodes:
        if "Organization" in str(
            item.get("@type")
        ):
            candidate = entity_name(
                item.get("name")
            )

            if candidate:
                organization_name = candidate
                break

    declared = (
        meta(
            soup,
            "og:site_name",
            "citation_publisher",
            "application-name",
            "publisher"
        )
        or entity_name(schema.get("publisher"))
        or organization_name
    )

    parsed = urlsplit(
        page["url"]
    )

    domain = domain_name(
        parsed.hostname
    )

    name = (
        declared
        or domain.split(".")[0].replace(
            "-",
            " "
        ).title()
    )

    author = (
        entity_name(schema.get("author"))
        or meta(
            soup,
            "author",
            "citation_author",
            "parsely-author"
        )
    )

    if not author and headline_node:
        for parent in list(
            headline_node.parents
        )[:4]:
            if parent.name in (
                "body",
                "html",
                "[document]"
            ):
                break

            tag = parent.select_one(
                "[itemprop='author'], "
                "[rel='author'], "
                ".byline, "
                ".author-name, "
                ".post-author"
            )

            if tag:
                candidate = re.sub(
                    r"^by\s+",
                    "",
                    clean(
                        tag.get_text(
                            " ",
                            strip=True
                        )
                    ),
                    flags=re.I
                )

                if 2 <= len(candidate) <= 100:
                    author = candidate
                    break

    date = as_date(
        schema.get("datePublished")
    )

    date_source = (
        "jsonld"
        if date
        else None
    )

    if not date:
        for key in (
            "article:published_time",
            "datePublished",
            "pubdate",
            "citation_date",
            "parsely-pub-date"
        ):
            date = as_date(
                meta(soup, key)
            )

            if date:
                date_source = "metadata"
                break

    if not date:
        date = date_near_headline(
            soup,
            title
        )

        if date:
            date_source = "article_headline_area"

    if not date and headline_node:
        date = date_near_headline(
            soup,
            clean(
                headline_node.get_text(
                    " ",
                    strip=True
                )
            )
        )

        if date:
            date_source = "article_headline_area"

    text = clean(
        schema.get("articleBody")
    )

    if len(text) < 80:
        candidates = []

        selectors = (
            "article, "
            "[itemprop='articleBody'], "
            ".article-content, "
            ".article-body, "
            ".news-content, "
            ".news-details, "
            ".entry-content, "
            ".post-content"
        )

        for area in soup.select(selectors):
            paragraphs = [
                clean(
                    paragraph.get_text(
                        " ",
                        strip=True
                    )
                )
                for paragraph in area.find_all("p")
            ]

            candidate = " ".join(
                paragraph
                for paragraph in paragraphs
                if len(paragraph) >= 25
            )

            if len(candidate) > 80:
                candidates.append(candidate)

        text = max(
            candidates,
            key=len,
            default=""
        )

    html_tag = soup.find("html")

    declared_lang = clean(
        html_tag.get("lang")
        if html_tag
        else ""
    ).split("-")[0].lower()

    sinhala_characters = sum(
        "\u0d80" <= character <= "\u0dff"
        for character in (
            (title or "")
            + (description or "")
        )
    )

    if sinhala_characters >= 3:
        language = "si"

    elif (
        declared_lang == "si"
        or "sinhala." in page["url"]
    ):
        language = "si"

    elif declared_lang == "en":
        language = "en"

    else:
        language = "unknown"

    image = meta(
        soup,
        "og:image:secure_url",
        "og:image",
        "twitter:image"
    )

    if not image:
        image_object = schema.get(
            "image"
        )

        if isinstance(image_object, str):
            image = image_object

        elif isinstance(image_object, dict):
            image = image_object.get(
                "url"
            )

    image = (
        urljoin(
            page["url"],
            image
        )
        if image
        else None
    )

    references = []

    article_tag = soup.find(
        "article"
    )

    if article_tag:
        for anchor in article_tag.find_all(
            "a",
            href=True
        ):
            target = urljoin(
                page["url"],
                anchor["href"]
            )

            host = urlsplit(
                target
            ).hostname

            if (
                host
                and domain_name(host) != domain
                and target.startswith("https://")
                and target not in references
            ):
                references.append(target)

            if len(references) >= 8:
                break

    accounts = social_profiles(
        soup,
        nodes,
        page["url"]
    )

    return {
        "article": {
            "url": page["url"],
            "headline": title,
            "language": language,
            "publication_date": date,
            "author": author,
            "summary": description,
            "text": text[:10000] or None,
            "image_url": image,
            "article_references": references
        },
        "publisher": {
            "name": name,
            "domain": domain,
            "website": urlunsplit((
                parsed.scheme,
                parsed.netloc,
                "/",
                "",
                ""
            )),
            "name_evidence": (
                "explicit_site_declaration"
                if declared
                else "domain_inference"
            )
        },
        "social_accounts": accounts,
        "extraction_evidence": {
            "publication_date_source": date_source,
            "author_status": (
                "disclosed"
                if author
                else "not_disclosed"
            ),
            "publisher_name_status": (
                "declared"
                if declared
                else "inferred"
            )
        }
    }


def page_links(html, url, domain):
    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    options = {
        kind: {}
        for kind in _KEYWORDS
    }

    for anchor in soup.find_all(
        "a",
        href=True
    ):
        href = urljoin(
            url,
            anchor.get("href", "")
        )

        parsed = urlsplit(
            href
        )

        if (
            parsed.scheme not in ("http", "https")
            or domain_name(parsed.hostname) != domain
        ):
            continue

        clean_url = urlunsplit((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            "",
            ""
        ))

        slug = (
            parsed.path.lower()
            .rstrip("/")
            .rsplit("/", 1)[-1]
        )

        label = clean(
            anchor.get_text(
                " ",
                strip=True
            )
        ).lower()

        for category, keywords in _KEYWORDS.items():
            for term in keywords:
                if slug == term or label == term:
                    options[category][clean_url] = 3

                elif (
                    slug.startswith(term + "-")
                    or slug.endswith("-" + term)
                ):
                    options[category][clean_url] = max(
                        options[category].get(
                            clean_url,
                            0
                        ),
                        2
                    )

    return options


def inspect_pages(page, publisher, limit=4):
    domain = publisher["domain"]
    homepage = publisher["website"]

    links = page_links(
        page["html"],
        page["url"],
        domain
    )

    if page["url"].rstrip("/") != homepage.rstrip("/"):
        try:
            home = fetch_html(
                homepage
            )

            other_links = page_links(
                home["html"],
                home["url"],
                domain
            )

            for category, urls in other_links.items():
                links[category].update(urls)

        except (ValueError, RuntimeError):
            pass

    result = {}
    inspected = 0

    for category, options in links.items():
        matches = sorted(
            options,
            key=lambda key: -options[key]
        )

        if not matches:
            result[category] = {
                "status": "not_discovered"
            }
            continue

        url = matches[0]

        if inspected >= limit:
            result[category] = {
                "status": "discovered_unchecked",
                "url": url
            }
            continue

        inspected += 1

        try:
            fetched = fetch_html(
                url
            )

            if domain_name(
                urlsplit(
                    fetched["url"]
                ).hostname
            ) != domain:
                result[category] = {
                    "status": "redirected_elsewhere",
                    "url": url
                }
                continue

            soup = BeautifulSoup(
                fetched["html"],
                "html.parser"
            )

            for tag in soup.select(
                "script, style, nav, footer, "
                "header, noscript, svg"
            ):
                tag.decompose()

            root = (
                soup.select_one(
                    "main, article, "
                    ".main-content, .content"
                )
                or soup.body
                or soup
            )

            text = clean(
                root.get_text(
                    " ",
                    strip=True
                )
            )

            email = bool(
                re.search(
                    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
                    text
                )
            )

            phone = bool(
                re.search(
                    r"(?<!\d)(?:\+94|0)"
                    r"[\d\s()\-]{8,15}\d(?!\d)",
                    text
                )
            )

            ownership = bool(
                re.search(
                    r"\b(?:owned by|operated by|"
                    r"published by|a company of|"
                    r"subsidiary of)\b|"
                    r"(?:හිමිකාරීත්වය|හිමිකාර|අයිතිය)",
                    text,
                    re.I
                )
            )

            policy = False

            if category == "editorial":
                policy = bool(
                    re.search(
                        r"editorial (?:policy|standards|"
                        r"guidelines)|code of (?:ethics|"
                        r"conduct)|මාධ්‍ය ආචාරධර්ම|"
                        r"කර්තෘ ප්‍රතිපත්තිය",
                        text,
                        re.I
                    )
                )

            elif category == "corrections":
                policy = bool(
                    re.search(
                        r"corrections? (?:policy|procedure)|"
                        r"report an error|request a correction|"
                        r"නිවැරදි කිරීම්",
                        text,
                        re.I
                    )
                )

            if category == "contact":
                substantive = email or phone

            elif category == "about":
                substantive = len(text) >= 120

            else:
                substantive = (
                    len(text) >= 150
                    and policy
                )

            result[category] = {
                "status": "inspected",
                "url": fetched["url"],
                "content_status": (
                    "supported"
                    if substantive
                    else "not_established"
                ),
                "ownership_statement_status": (
                    (
                        "declared_not_verified"
                        if ownership
                        else "not_detected"
                    )
                    if category == "about"
                    else None
                ),
                "excerpt": (
                    text[:700]
                    if category == "about"
                    else None
                )
            }

        except (ValueError, RuntimeError) as exc:
            result[category] = {
                "status": "could_not_check",
                "url": url,
                "error_type": type(exc).__name__
            }

    return result


def assess_transparency(
    article,
    publisher,
    pages,
    accounts
):
    about = pages.get(
        "about",
        {}
    )

    about_name_matches = (
        about.get("content_status") == "supported"
        and re.sub(
            r"[^\w]",
            "",
            publisher["name"].casefold()
        ) in re.sub(
            r"[^\w]",
            "",
            clean(
                about.get("excerpt")
            ).casefold()
        )
    )

    found = {
        "https_used": article["url"].startswith(
            "https://"
        ),
        "publisher_identified": (
            publisher["name_evidence"]
            == "explicit_site_declaration"
            or about_name_matches
        ),
        "author_disclosed": bool(
            article.get("author")
        ),
        "publication_date": bool(
            article.get("publication_date")
        ),
        "contact_details": (
            pages.get(
                "contact",
                {}
            ).get("content_status")
            == "supported"
        ),
        "about_page": (
            about.get("content_status")
            == "supported"
        ),
        "editorial_policy": (
            pages.get(
                "editorial",
                {}
            ).get("content_status")
            == "supported"
        ),
        "corrections_policy": (
            pages.get(
                "corrections",
                {}
            ).get("content_status")
            == "supported"
        ),
        "ownership_disclosure": (
            about.get("ownership_statement_status")
            == "declared_not_verified"
        ),
        "social_links": bool(accounts),
        "article_references": bool(
            article.get("article_references")
        )
    }

    support = sum(
        weight
        for key, weight in _WEIGHTS.items()
        if found[key]
    )

    indicators = {}

    page_mapping = {
        "contact_details": "contact",
        "about_page": "about",
        "editorial_policy": "editorial",
        "corrections_policy": "corrections",
        "ownership_disclosure": "about"
    }

    article_indicators = {
        "https_used",
        "author_disclosed",
        "publication_date",
        "article_references"
    }

    for name, weight in _WEIGHTS.items():
        page_name = page_mapping.get(name)

        if name == "ownership_disclosure" and found[name]:
            status = "declared_not_independently_verified"

        else:
            status = (
                "observed"
                if found[name]
                else "not_established"
            )

        if page_name:
            evidence_key = page_name

        elif name in article_indicators:
            evidence_key = "article"

        else:
            evidence_key = "publisher"

        indicators[name] = {
            "status": status,
            "points": (
                weight
                if found[name]
                else 0
            ),
            "weight": weight,
            "evidence_key": (
                evidence_key
                if found[name]
                else None
            )
        }

    return {
        "score_name": "observed_transparency_points",
        "observed_transparency_points": support,
        "maximum_points": sum(
            _WEIGHTS.values()
        ),
        "interpretation": (
            "Presence of publisher-disclosed "
            "transparency signals; not article truth "
            "probability or independently validated credibility"
        ),
        "rating_status": "research_baseline_not_validated",
        "unestablished_weight": 100 - support,
        "indicators": indicators
    }


def analyze_source(url):
    page = fetch_html(
        url
    )

    extracted = analyze_article(
        page
    )

    pages = inspect_pages(
        page,
        extracted["publisher"]
    )

    score = assess_transparency(
        extracted["article"],
        extracted["publisher"],
        pages,
        extracted["social_accounts"]
    )

    evidence = {
        "page_inspection": {
            key: {
                name: value
                for name, value in item.items()
                if name != "excerpt"
            }
            for key, item in pages.items()
        },
        "extraction": extracted["extraction_evidence"]
    }

    return {
        "publisher": extracted["publisher"],
        "article": extracted["article"],
        "social_accounts": extracted["social_accounts"],
        "transparency": score,
        "evidence": evidence
    }
