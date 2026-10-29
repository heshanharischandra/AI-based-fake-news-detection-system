
import csv
import os
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PUBLISHERS_CSV = ROOT / "data" / "publishers.csv"
PAIRS_CSV = ROOT / "data" / "identity_pairs.csv"



PLANNED_COMPARISONS = [
    ("PUB001", "PUB002", "youtube"),
    ("PUB001", "PUB024", "youtube"),
    ("PUB002", "PUB001", "youtube"),
    ("PUB002", "PUB024", "facebook"),
    ("PUB003", "PUB002", "facebook"),
    ("PUB003", "PUB008", "facebook"),
    ("PUB005", "PUB007", "facebook"),
    ("PUB005", "PUB024", "youtube"),
    ("PUB007", "PUB003", "youtube"),
    ("PUB007", "PUB001", "facebook"),
    ("PUB008", "PUB002", "youtube"),
    ("PUB008", "PUB024", "facebook"),
    ("PUB009", "PUB001", "facebook"),
    ("PUB009", "PUB003", "youtube"),
    ("PUB010", "PUB008", "facebook"),
    ("PUB010", "PUB002", "facebook"),
    ("PUB015", "PUB024", "youtube"),
    ("PUB015", "PUB007", "facebook"),
    ("PUB024", "PUB002", "facebook"),
    ("PUB024", "PUB001", "youtube"),
    ("PUB026", "PUB001", "facebook"),
    ("PUB026", "PUB003", "youtube"),
    ("PUB029", "PUB024", "youtube"),
    ("PUB029", "PUB003", "youtube"),
    ("PUB030", "PUB008", "facebook"),
    ("PUB030", "PUB001", "youtube"),
    ("PUB033", "PUB024", "youtube"),
    ("PUB033", "PUB008", "facebook"),
]


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def pair_key(record):
    return (
        (record.get("publisher_id") or "").strip().upper(),
        (record.get("platform") or "").strip().lower(),
        (record.get("account_url") or "")
        .strip()
        .rstrip("/")
        .lower(),
    )


def main():
    publisher_columns, publishers = read_csv(
        PUBLISHERS_CSV
    )

    pair_columns, existing = read_csv(
        PAIRS_CSV
    )

    required_publisher_columns = {
        "publisher_id",
        "publisher_name",
        "website_url",
    }

    required_pair_columns = {
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
        "notes",
    }

    if not required_publisher_columns.issubset(
        publisher_columns
    ):
        raise ValueError(
            "publishers.csv lacks required columns"
        )

    if not required_pair_columns.issubset(
        pair_columns
    ):
        raise ValueError(
            "identity_pairs.csv lacks required columns"
        )

    publisher_by_id = {
        (p.get("publisher_id") or "").strip().upper(): p
        for p in publishers
    }



    original_accounts = {}

    for record in existing:
        evidence_type = (
            record.get("association_evidence") or ""
        ).lower()

        if "cross_publisher" in evidence_type:
            continue

        key = (
            (record.get("publisher_id") or "")
            .strip()
            .upper(),
            (record.get("platform") or "")
            .strip()
            .lower(),
        )

        if (record.get("account_url") or "").strip():
            original_accounts.setdefault(
                key,
                record
            )

    # Check prerequisites before modifying the dataset.

    for target_id, source_id, platform in PLANNED_COMPARISONS:

        if target_id == source_id:
            raise ValueError(
                f"Invalid self-comparison: {target_id}"
            )

        if target_id not in publisher_by_id:
            raise ValueError(
                f"Publisher {target_id} not found"
            )

        if (source_id, platform) not in original_accounts:
            raise ValueError(
                f"No existing {platform} account "
                f"for {source_id}"
            )

    known_keys = {
        pair_key(row)
        for row in existing
    }

    additions = []

    # Create new pending candidate pairs.

    for target_id, source_id, platform in PLANNED_COMPARISONS:

        target = publisher_by_id[target_id]
        source = original_accounts[
            (source_id, platform)
        ]

        candidate = {
            "publisher_id": target_id,
            "publisher_name": target["publisher_name"],
            "website_url": target["website_url"],
            "platform": platform,
            "account_url": source["account_url"],
            "language": target.get("language") or "en",
            "association_evidence": (
                "cross_publisher_candidate"
            ),
            "identity_label": "unknown",
            "review_status": "pending",
            "evidence_url": "",
            "notes": (
                "Unverified cross-publisher candidate. "
                "This social account was originally "
                "listed under "
                f"{source.get('publisher_name', source_id)}. "
                "Check account ownership and both "
                "publishers before labeling."
            ),
        }

        key = pair_key(candidate)

        if key in known_keys:
            continue

        additions.append({
            column: candidate.get(column, "")
            for column in pair_columns
        })

        known_keys.add(key)

    if not additions:
        print(
            "No new pairs to add (already present)."
        )
        print(
            "Total identity pairs:",
            len(existing)
        )
        return

    

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    backup = PAIRS_CSV.with_name(
        f"identity_pairs_backup_{timestamp}.csv"
    )

    shutil.copy2(
        PAIRS_CSV,
        backup
    )

    temporary = PAIRS_CSV.with_suffix(
        ".csv.tmp"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as output:

            writer = csv.DictWriter(
                output,
                fieldnames=pair_columns
            )

            writer.writeheader()
            writer.writerows(
                existing + additions
            )

        os.replace(
            temporary,
            PAIRS_CSV
        )

    finally:
        if temporary.exists():
            temporary.unlink()

    print(
        "Added unlabeled candidate pairs:",
        len(additions)
    )

    print(
        "Total identity pairs:",
        len(existing) + len(additions)
    )

    print("Backup saved:", backup)

    print(
        "Previously saved review file: NOT MODIFIED"
    )

    print(
        "All new candidate labels: unknown / pending"
    )

    print(
        "Next: run review_identity_pairs.py "
        "and verify evidence manually"
    )


if __name__ == "__main__":
    main()
