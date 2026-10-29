
import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data" / "identity_pairs.csv"

OUTPUT_PATH = BASE_DIR / "data" / "identity_pairs_reviewed.csv"

REVIEW_FIELDS = [
    "reviewer_id",
    "reviewed_at",
    "review_evidence_url",
    "review_notes"
]

LABEL_OPTIONS = {
    "1": "same_publisher",
    "2": "different_publisher",
    "3": "unknown"
}


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV header is missing")

        return (
            reader.fieldnames,
            list(reader)
        )


def pair_key(row):
    return (
        row["publisher_id"].strip(),
        row["platform"].strip().lower(),
        row["account_url"].strip().rstrip("/").lower()
    )


def valid_url(value):
    if not value:
        return False

    parsed = urlsplit(value)

    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.hostname)
    )


def save_reviews(rows, columns):
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = OUTPUT_PATH.with_suffix(".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()
        writer.writerows(rows)

    os.replace(
        temporary,
        OUTPUT_PATH
    )


def review_dataset():
    original_columns, original_rows = read_csv(
        INPUT_PATH
    )

    columns = list(
        dict.fromkeys(
            original_columns + REVIEW_FIELDS
        )
    )

    previous_reviews = {}

    if OUTPUT_PATH.exists():
        _, saved_rows = read_csv(
            OUTPUT_PATH
        )

        previous_reviews = {
            pair_key(row): row
            for row in saved_rows
        }

    rows = []

    for original in original_rows:
        previous = previous_reviews.get(
            pair_key(original)
        )

        if previous:
            row = {
                **original,
                **{
                    field: previous.get(field, "")
                    for field in (
                        "identity_label",
                        "review_status",
                        *REVIEW_FIELDS
                    )
                }
            }
        else:
            row = {
                **original,
                **{
                    field: ""
                    for field in REVIEW_FIELDS
                }
            }

        rows.append(row)

    reviewer = input(
        "Enter reviewer ID (example R01): "
    ).strip()

    if not reviewer:
        raise ValueError(
            "Reviewer ID is required"
        )

    print()
    print("=" * 65)
    print("CROSS-PLATFORM IDENTITY DATASET REVIEW")
    print("=" * 65)

    for index, row in enumerate(
        rows,
        start=1
    ):
        if row.get("review_status") == "reviewed":
            print(
                f"Skipping previously reviewed pair {index}"
            )
            continue

        print()
        print("=" * 65)
        print(
            f"PAIR {index} OF {len(rows)}"
        )
        print("=" * 65)

        print(
            "Publisher:",
            row["publisher_name"]
        )

        print(
            "Website:",
            row["website_url"]
        )

        print(
            "Platform:",
            row["platform"]
        )

        print(
            "Account:",
            row["account_url"]
        )

        print(
            "Recorded evidence:",
            row["evidence_url"]
        )

        print(
            "Evidence type:",
            row["association_evidence"]
        )

        print()
        print("1 - Supported same-publisher association")
        print("2 - Evidence of a different publisher")
        print("3 - Insufficient evidence")
        print("s - Skip")
        print("q - Save and quit")

        while True:
            choice = input(
                "Your review decision: "
            ).strip().lower()

            if choice in {
                "1", "2", "3", "s", "q"
            }:
                break

            print("Invalid option")

        if choice == "q":
            break

        if choice == "s":
            continue

        label = LABEL_OPTIONS[choice]

        print()
        print(
            "Open the publisher website and account"
        )
        print(
            "in your browser before recording evidence."
        )

        while True:
            review_url = input(
                "Evidence URL (blank if none): "
            ).strip()

            if (
                not review_url
                or valid_url(review_url)
            ):
                break

            print(
                "Provide a valid HTTP or HTTPS URL"
            )

        if (
            label != "unknown"
            and not review_url
        ):
            print(
                "A supported or contradicted association"
            )
            print(
                "requires a checked evidence URL."
            )
            print(
                "This pair remains pending."
            )
            continue

        notes = input(
            "Explain your evidence and decision: "
        ).strip()

        if (
            label != "unknown"
            and len(notes) < 15
        ):
            print(
                "A detailed review explanation is required."
            )
            print(
                "This pair remains pending."
            )
            continue

        row["identity_label"] = label
        row["review_status"] = "reviewed"
        row["reviewer_id"] = reviewer
        row["reviewed_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        row["review_evidence_url"] = review_url
        row["review_notes"] = notes

        save_reviews(
            rows,
            columns
        )

        print(
            "Review saved:",
            label
        )

    save_reviews(
        rows,
        columns
    )

    reviewed = [
        row
        for row in rows
        if row["review_status"] == "reviewed"
    ]

    positive = sum(
        row["identity_label"] == "same_publisher"
        for row in reviewed
    )

    negative = sum(
        row["identity_label"] == "different_publisher"
        for row in reviewed
    )

    uncertain = sum(
        row["identity_label"] == "unknown"
        for row in reviewed
    )

    print()
    print("=" * 65)
    print("REVIEW SUMMARY")
    print("=" * 65)

    print("Total pairs:", len(rows))
    print("Reviewed:", len(reviewed))
    print("Same publisher:", positive)
    print("Different publisher:", negative)
    print("Uncertain:", uncertain)
    print("Remaining:", len(rows) - len(reviewed))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    review_dataset()
