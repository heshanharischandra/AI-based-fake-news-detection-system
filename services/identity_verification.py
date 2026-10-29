
import csv
import json
import re
import unicodedata

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlsplit

import tldextract


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "identity_pairs.csv"

PUBLISHERS_PATH = BASE_DIR / "data" / "publishers.csv"

EXTRACTOR = tldextract.TLDExtract(
    suffix_list_urls=()
)

REQUIRED_COLUMNS = {
    "publisher_id",
    "publisher_name",
    "website_url",
    "platform",
    "account_url",
    "language",
    "association_evidence",
    "identity_label",
    "review_status",
    "evidence_url",
    "notes"
}

ALLOWED_PLATFORMS = {
    "facebook",
    "youtube"
}

ALLOWED_LABELS = {
    "unknown",
    "same_publisher",
    "different_publisher"
}

ALLOWED_REVIEW_STATUS = {
    "pending",
    "reviewed"
}


def normalize_name(value):
    text = unicodedata.normalize(
        "NFKC",
        str(value or "")
    ).casefold()

    return "".join(
        character
        for character in text
        if character.isalnum()
    )


def name_similarity(left, right):
    first = normalize_name(left)
    second = normalize_name(right)

    if not first or not second:
        return 0.0

    return round(
        SequenceMatcher(
            None,
            first,
            second
        ).ratio(),
        4
    )


def registered_domain(url):
    hostname = urlsplit(url).hostname or ""

    extracted = EXTRACTOR(
        hostname
    )

    if extracted.domain and extracted.suffix:
        return (
            f"{extracted.domain}.{extracted.suffix}"
        )

    return hostname.lower()


def validate_url(url):
    parsed = urlsplit(
        str(url or "").strip()
    )

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            f"Invalid URL: {url}"
        )

    return parsed.geturl()


def account_handle(url):
    parsed = urlsplit(
        url
    )

    parts = [
        part
        for part in parsed.path.split("/")
        if part
    ]

    if not parts:
        return None

    if parts[0].lower() == "profile.php":
        return None

    return parts[-1].lstrip("@")


def validate_platform(url, platform):
    hostname = (
        urlsplit(url).hostname or ""
    ).lower()

    if platform == "facebook":
        return hostname in {
            "facebook.com",
            "www.facebook.com",
            "m.facebook.com",
            "web.facebook.com"
        }

    if platform == "youtube":
        return hostname in {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com"
        }

    return False


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(
            file
        )

        if reader.fieldnames is None:
            raise ValueError(
                "CSV header is missing"
            )

        missing = REQUIRED_COLUMNS - set(
            reader.fieldnames
        )

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        return list(
            reader
        )


def load_publishers():
    if not PUBLISHERS_PATH.exists():
        return {}

    with PUBLISHERS_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        return {
            row["publisher_id"]: row
            for row in csv.DictReader(file)
        }


def analyze_identity_candidates(
    csv_path=DATA_PATH
):
    records = load_csv(
        Path(csv_path)
    )

    publishers = load_publishers()

    results = []
    seen = set()

    for number, record in enumerate(
        records,
        start=2
    ):
        publisher_id = record[
            "publisher_id"
        ].strip()

        publisher_name = record[
            "publisher_name"
        ].strip()

        platform = record[
            "platform"
        ].strip().lower()

        label = record[
            "identity_label"
        ].strip().lower()

        review = record[
            "review_status"
        ].strip().lower()

        if platform not in ALLOWED_PLATFORMS:
            raise ValueError(
                f"Invalid platform on row {number}"
            )

        if label not in ALLOWED_LABELS:
            raise ValueError(
                f"Invalid identity label on row {number}"
            )

        if review not in ALLOWED_REVIEW_STATUS:
            raise ValueError(
                f"Invalid review status on row {number}"
            )

        if (
            review == "pending"
            and label != "unknown"
        ):
            raise ValueError(
                f"Pending row {number} cannot have "
                "a confirmed training label"
            )

        website_url = validate_url(
            record["website_url"]
        )

        account_url = validate_url(
            record["account_url"]
        )

        evidence_url = validate_url(
            record["evidence_url"]
        )

        if not validate_platform(
            account_url,
            platform
        ):
            raise ValueError(
                f"Platform URL mismatch on row {number}"
            )

        pair_key = (
            publisher_id,
            platform,
            account_url.rstrip("/").lower()
        )

        if pair_key in seen:
            raise ValueError(
                f"Duplicate identity pair on row {number}"
            )

        seen.add(
            pair_key
        )

        website_domain = registered_domain(
            website_url
        )

        evidence_domain = registered_domain(
            evidence_url
        )

        handle = account_handle(
            account_url
        )

        similarity = name_similarity(
            publisher_name,
            handle
        )

        publisher_record = publishers.get(
            publisher_id
        )

        if publishers and not publisher_record:
            raise ValueError(
                f"Publisher {publisher_id} "
                "is missing from publishers.csv"
            )

        if publisher_record:
            registry_domain = registered_domain(
                validate_url(
                    publisher_record["website_url"]
                )
            )

            if registry_domain != website_domain:
                raise ValueError(
                    f"Publisher domain mismatch "
                    f"on row {number}"
                )

        results.append({
            "publisher_id": publisher_id,
            "publisher_name": publisher_name,
            "publisher_domain": website_domain,
            "platform": platform,
            "account_url": account_url,
            "account_handle": handle,
            "features": {
                "name_similarity": similarity,
                "evidence_url_on_publisher_domain": (
                    evidence_domain == website_domain
                ),
                "platform_url_valid": True
            },
            "evidence": {
                "url": evidence_url,
                "recorded_type": record[
                    "association_evidence"
                ],
                "independently_checked": False
            },
            "review": {
                "identity_label": label,
                "status": review
            },
            "verification_status": (
                "human_reviewed"
                if (
                    review == "reviewed"
                    and label != "unknown"
                )
                else "requires_manual_review"
            )
        })

    platforms = Counter(
        item["platform"]
        for item in results
    )

    labels = Counter(
        item["review"]["identity_label"]
        for item in results
    )

    reviewed = sum(
        item["review"]["status"] == "reviewed"
        and item["review"]["identity_label"] != "unknown"
        for item in results
    )

    summary = {
        "total_pairs": len(results),
        "unique_publishers": len({
            item["publisher_id"]
            for item in results
        }),
        "facebook_pairs": platforms["facebook"],
        "youtube_pairs": platforms["youtube"],
        "unknown_labels": labels["unknown"],
        "same_publisher_labels": labels["same_publisher"],
        "different_publisher_labels": labels[
            "different_publisher"
        ],
        "reviewed_labeled_pairs": reviewed,
        "pending_or_uncertain_pairs": (
            len(results) - reviewed
        ),
        "training_status": (
            "not_ready"
            if (
                labels["same_publisher"] == 0
                or labels["different_publisher"] == 0
            )
            else "requires_dataset_quality_review"
        )
    }

    return {
        "summary": summary,
        "pairs": results
    }


if __name__ == "__main__":
    result = analyze_identity_candidates()

    print(
        json.dumps(
            result["summary"],
            ensure_ascii=False,
            indent=2
        )
    )

    print(
        json.dumps(
            result["pairs"][:3],
            ensure_ascii=False,
            indent=2
        )
    )
