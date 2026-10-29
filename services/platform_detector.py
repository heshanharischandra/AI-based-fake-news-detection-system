
import ipaddress
import re
from urllib.parse import urlsplit, urlunsplit, parse_qs

FACEBOOK_HOSTS = {
    "facebook.com",
    "www.facebook.com",
    "m.facebook.com",
    "web.facebook.com",
    "fb.com",
    "www.fb.com"
}

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com"
}

BLOCKED_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".test",
    ".invalid",
    ".onion"
)


def detect_platform(url: str) -> dict:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("A valid URL is required")

    submitted_url = url.strip()

    if len(submitted_url) > 2048:
        raise ValueError("URL is too long")

    if re.search(r"[\s\\\x00-\x1f\x7f]", submitted_url):
        raise ValueError("URL contains invalid characters")

    if "://" not in submitted_url:
        submitted_url = "https://" + submitted_url

    parsed = urlsplit(submitted_url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported")

    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URLs containing login credentials are not allowed")

    if not parsed.hostname:
        raise ValueError("Invalid URL hostname")

    try:
        hostname = parsed.hostname.encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise ValueError("Invalid domain name")

    try:
        ipaddress.ip_address(hostname)
        raise ValueError("Direct IP addresses are not supported")
    except ValueError as error:
        if str(error) == "Direct IP addresses are not supported":
            raise

    if hostname == "localhost":
        raise ValueError("Local addresses are not supported")

    if hostname.endswith(BLOCKED_SUFFIXES):
        raise ValueError("Restricted domain")

    labels = hostname.split(".")

    if len(labels) < 2 or len(hostname) > 253:
        raise ValueError("Invalid domain name")

    for label in labels:
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label):
            raise ValueError("Invalid domain name")

    if len(labels[-1]) < 2 or labels[-1].isdigit():
        raise ValueError("Invalid top-level domain")

    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid URL port")

    if port is not None and port not in {80, 443}:
        raise ValueError("Unsupported URL port")

    netloc = hostname + (f":{port}" if port else "")

    normalized_url = urlunsplit((
        parsed.scheme,
        netloc,
        parsed.path or "/",
        parsed.query,
        ""
    ))

    platform = "website"
    source_identifier = hostname

    parts = [part for part in parsed.path.split("/") if part]

    if hostname in FACEBOOK_HOSTS:
        platform = "facebook"

        if not parts:
            raise ValueError("A Facebook Page URL is required")

        first = parts[0].lower()

        if first == "profile.php":
            source_identifier = parse_qs(parsed.query).get("id", [None])[0]

            if not source_identifier:
                raise ValueError("Facebook profile ID is missing")

        elif first in {"pages", "people"} and len(parts) >= 3:
            source_identifier = parts[-1]

        elif first in {
            "watch", "reel", "reels", "share",
            "groups", "events", "stories",
            "marketplace", "login", "help"
        }:
            raise ValueError("Please provide a Facebook Page URL")

        else:
            source_identifier = parts[0]

    elif hostname in YOUTUBE_HOSTS:
        platform = "youtube"

        if not parts:
            raise ValueError("A YouTube channel URL is required")

        first = parts[0]

        if first.startswith("@"):
            source_identifier = first

        elif first.lower() in {"channel", "user", "c"} and len(parts) >= 2:
            source_identifier = parts[1]

        else:
            raise ValueError("Please provide a YouTube channel URL")

    elif hostname in {"youtu.be", "www.youtu.be", "fb.watch"}:
        raise ValueError("Please provide a channel or Page URL instead of a video link")

    return {
        "submitted_url": url,
        "normalized_url": normalized_url,
        "platform": platform,
        "domain": hostname,
        "source_identifier": source_identifier,
        "verification_status": "not_verified"
    }
