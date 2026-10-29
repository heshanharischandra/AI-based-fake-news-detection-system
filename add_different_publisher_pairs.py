
import csv
import os
import shutil
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "identity_pairs.csv"


CANDIDATES = [
    {
        "publisher_id": "PUB001",
        "publisher_name": "Ada Derana",
        "website_url": "https://sinhala.adaderana.lk/",
        "platform": "facebook",
        "account_url": "https://www.facebook.com/hirunews",
        "language": "si",
        "association_evidence": "cross_publisher_candidate",
        "identity_label": "unknown",
        "review_status": "pending",
        "evidence_url": "https://t.me/s/hirunewsofficial",
        "notes": (
            "Potential different-publisher example. "
            "Verify that the Facebook account belongs "
            "to Hiru News, not Ada Derana."
        ),
    },
    {
        "publisher_id": "PUB002",
        "publisher_name": "Hiru News",
        "website_url": "https://www.hirunews.lk/",
        "platform": "facebook",
        "account_url": "https://www.facebook.com/sinhala.adaderana.lk",
        "language": "si",
        "association_evidence": "cross_publisher_candidate",
        "identity_label": "unknown",
        "review_status": "pending",
        "evidence_url": "https://sinhala.adaderana.lk/news/252937",
        "notes": (
            "Potential different-publisher example. "
            "Verify that the Facebook account belongs "
            "to Ada Derana, not Hiru News."
        ),
    },
]


def pair_key(row):
    return (
        (row.get("publisher_id") or "").strip(),
        (row.get("platform") or "").strip().lower(),
        (row.get("account_url") or "").strip().rstrip("/").lower(),
    )


def main():
    if not CSV_PATH.is_file():
        print("Dataset not found:", CSV_PATH)
        return

    with CSV_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required = {
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

    missing = required - set(fieldnames)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    known = {pair_key(row) for row in rows}
    added = []

    for candidate in CANDIDATES:
        key = pair_key(candidate)

        if key in known:
            continue

        rows.append({
            column: candidate.get(column, "")
            for column in fieldnames
        })

        known.add(key)
        added.append(candidate)

    if not added:
        print(
            "Both candidate pairs are already present. "
            "Nothing changed."
        )
        print("Total identity pairs:", len(rows))
        return

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    backup = CSV_PATH.with_name(
        f"identity_pairs_backup_{timestamp}.csv"
    )

    shutil.copy2(CSV_PATH, backup)

    temporary = CSV_PATH.with_suffix(".csv.tmp")

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as target:
            writer = csv.DictWriter(
                target,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(rows)

        os.replace(temporary, CSV_PATH)

    finally:
        if temporary.exists():
            temporary.unlink()

    print("Added candidate pairs:", len(added))
    print("Total identity pairs:", len(rows))
    print("Updated:", CSV_PATH)
    print("Backup:", backup)
    print(
        "New pairs remain unknown/pending "
        "until manually reviewed."
    )
    print(
        "Existing identity_pairs_reviewed.csv "
        "was NOT changed."
    )


if __name__ == "__main__":
    main()
