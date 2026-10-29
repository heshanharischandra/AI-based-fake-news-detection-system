
import csv
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent

FILE = (
    ROOT / "data" / "facebook_page_evidence.csv"
)

FIELDS = [
    "page_url",
    "indicator",
    "status",
    "evidence_url",
    "reviewer_id",
    "reviewed_at",
    "notes",
]

OPTIONS = {
    "1": "page_identity",
    "2": "about_information",
    "3": "contact_information",
    "4": "linked_website",
    "5": "page_category",
    "6": "location_information",
    "7": "page_transparency_information",
}


def http_url(value):
    try:
        p = urlsplit(value)

        return (
            p.scheme == "https"
            and bool(p.hostname)
            and not p.username
            and not p.password
        )

    except ValueError:
        return False


def run():
    print("Facebook Page evidence review")

    page = input(
        "Facebook Page URL: "
    ).strip()

    allowed_hosts = {
        "facebook.com",
        "www.facebook.com",
        "m.facebook.com",
        "web.facebook.com",
    }

    if (
        not http_url(page)
        or (
            urlsplit(page).hostname
            or ""
        ).lower() not in allowed_hosts
    ):
        raise ValueError(
            "Enter an HTTPS Facebook Page URL"
        )

    reviewer = input(
        "Reviewer ID: "
    ).strip()

    if not reviewer:
        raise ValueError(
            "Reviewer ID required"
        )

    FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not FILE.exists():
        with FILE.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as stream:
            csv.DictWriter(
                stream,
                fieldnames=FIELDS
            ).writeheader()

    while True:
        print(
            "\nSelect indicator (q = finish):"
        )

        for number, name in OPTIONS.items():
            print(
                number,
                name
            )

        selected = input(
            "Indicator: "
        ).strip().lower()

        if selected == "q":
            break

        if selected not in OPTIONS:
            print(
                "Unknown indicator"
            )
            continue

        status = input(
            "Observed status (present/absent): "
        ).strip().lower()

        if status not in {
            "present",
            "absent"
        }:
            print(
                "Status must be present or absent"
            )
            continue

        evidence = input(
            "Actual supporting evidence URL: "
        ).strip()

        if not http_url(evidence):
            print(
                "An HTTPS evidence URL is required"
            )
            continue

        notes = input(
            "What did you observe? (20+ characters): "
        ).strip()

        if len(notes) < 20:
            print(
                "Give a specific explanation, "
                "at least 20 characters"
            )
            continue

        row = {
            "page_url": page,
            "indicator": OPTIONS[selected],
            "status": status,
            "evidence_url": evidence,
            "reviewer_id": reviewer,
            "reviewed_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "notes": notes,
        }

        with FILE.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=FIELDS
            )

            writer.writerow(row)

        print(
            "Evidence row saved"
        )

    print(
        "File:",
        FILE
    )


if __name__ == "__main__":
    run()
