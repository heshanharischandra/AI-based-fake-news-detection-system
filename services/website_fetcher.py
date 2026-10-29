
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import requests

from services.platform_detector import detect_platform


MAX_RESPONSE_BYTES = 4000000
MAX_REDIRECTS = 5


def validate_public_website(url):
    details = detect_platform(url)

    if details["platform"] != "website":
        raise ValueError("A news website URL is required")

    normalized_url = details["normalized_url"]
    parsed = urlsplit(normalized_url)

    hostname = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        addresses = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM
        )
    except socket.gaierror:
        raise ValueError("Website hostname could not be resolved")

    if not addresses:
        raise ValueError("No IP addresses found for website")

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])

        if not ip.is_global:
            raise ValueError("Website resolves to a restricted IP address")

    return normalized_url


def fetch_website(url):
    current_url = url.strip()

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SourceCredibilityResearch/1.0)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "si,en;q=0.9"
    }

    with requests.Session() as session:
        session.trust_env = False

        for redirect_count in range(MAX_REDIRECTS + 1):
            current_url = validate_public_website(current_url)

            try:
                response = session.get(
                    current_url,
                    headers=headers,
                    timeout=(5, 12),
                    allow_redirects=False,
                    stream=True
                )
            except requests.RequestException as error:
                raise RuntimeError(
                    f"Website request failed: {type(error).__name__}"
                ) from error

            with response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")

                    if not location:
                        raise RuntimeError("Redirect location is missing")

                    current_url = urljoin(current_url, location)
                    continue

                try:
                    response.raise_for_status()
                except requests.HTTPError as error:
                    raise RuntimeError(
                        f"Website returned HTTP {response.status_code}"
                    ) from error

                content_type = response.headers.get(
                    "Content-Type", ""
                ).lower()

                if content_type and not (
                    "text/html" in content_type
                    or "application/xhtml+xml" in content_type
                ):
                    raise ValueError(
                        "The URL does not return an HTML page"
                    )

                declared_size = response.headers.get("Content-Length")

                if declared_size:
                    try:
                        if int(declared_size) > MAX_RESPONSE_BYTES:
                            raise ValueError("Website response is too large")
                    except ValueError as error:
                        if str(error) == "Website response is too large":
                            raise

                chunks = []
                total_size = 0

                try:
                    for chunk in response.iter_content(chunk_size=65536):
                        if not chunk:
                            continue

                        total_size += len(chunk)

                        if total_size > MAX_RESPONSE_BYTES:
                            raise ValueError("Website response is too large")

                        chunks.append(chunk)
                except requests.RequestException as error:
                    raise RuntimeError(
                        "Website content could not be downloaded"
                    ) from error

                raw_html = b"".join(chunks)

                encoding = response.encoding

                if not encoding or encoding.lower() in (
                    "iso-8859-1",
                    "latin-1"
                ):
                    encoding = "utf-8"

                try:
                    html = raw_html.decode(
                        encoding,
                        errors="replace"
                    )
                except LookupError:
                    encoding = "utf-8"
                    html = raw_html.decode(
                        encoding,
                        errors="replace"
                    )

                return {
                    "submitted_url": url,
                    "final_url": current_url,
                    "http_status": response.status_code,
                    "content_type": content_type,
                    "encoding": encoding,
                    "html": html,
                    "response_size_bytes": total_size
                }

    raise RuntimeError("Too many website redirects")
