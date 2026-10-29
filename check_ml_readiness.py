
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def load_rows(filename):
    path = DATA / filename

    if not path.exists():
        print(f"MISSING: {path.relative_to(ROOT)}")
        return [], []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        rows = list(reader)

    print(
        f"FOUND: {path.relative_to(ROOT)} "
        f"({len(rows)} rows)"
    )

    return rows, columns


def check_identity():
    print()
    print("IDENTITY MATCHING DATA")
    print("=" * 54)

    rows, columns = load_rows(
        "identity_pairs_reviewed.csv"
    )

    if not rows:
        print(
            "Next: prepare identity pairs "
            "and review their evidence."
        )
        return

    required = {
        "publisher_id",
        "platform",
        "account_url",
        "identity_label",
        "review_status"
    }

    missing = required.difference(columns)

    if missing:
        print(
            "Missing columns:",
            ", ".join(sorted(missing))
        )
        return

    status_counts = Counter(
        (
            row.get("review_status") or ""
        ).strip().lower()
        for row in rows
    )

    labeled = [
        row
        for row in rows
        if (
            row.get("review_status") or ""
        ).strip().lower() == "reviewed"
        and (
            row.get("identity_label") or ""
        ).strip().lower()
        in {
            "same_publisher",
            "different_publisher"
        }
    ]

    labels = Counter(
        (
            row.get("identity_label") or ""
        ).strip().lower()
        for row in labeled
    )

    unique_publishers = len({
        (
            row.get("publisher_id") or ""
        ).strip()
        for row in labeled
    })

    keys = [
        (
            (
                row.get("publisher_id") or ""
            ).strip(),
            (
                row.get("platform") or ""
            ).strip().lower(),
            (
                row.get("account_url") or ""
            ).strip().rstrip("/").lower()
        )
        for row in rows
    ]

    print(
        "Status counts:",
        dict(status_counts)
    )

    print(
        "Independently reviewed pairs:",
        len(labeled)
    )

    print(
        "Same-publisher labels:",
        labels["same_publisher"]
    )

    print(
        "Different-publisher labels:",
        labels["different_publisher"]
    )

    print(
        "Publishers represented in labeled pairs:",
        unique_publishers
    )

    print(
        "Duplicate pair keys:",
        len(keys) - len(set(keys))
    )

    if (
        labels["same_publisher"] >= 3
        and labels["different_publisher"] >= 3
    ):
        print(
            "Baseline evaluation: POSSIBLE, "
            "but small-sample results may be unreliable."
        )

    else:
        print(
            "Baseline evaluation: NOT READY "
            "(needs at least 3 reviewed pairs "
            "of each class)."
        )

    if (
        len(labeled) >= 40
        and min(
            labels["same_publisher"],
            labels["different_publisher"]
        ) >= 15
        and unique_publishers >= 10
    ):
        print(
            "ML pilot: Enough labels to attempt "
            "grouped training and testing; "
            "evaluate carefully."
        )

    else:
        print(
            "ML pilot: NOT READY "
            "for meaningful supervised training."
        )

        print(
            "Suggested first milestone: "
            "40 reviewed pairs, at least 15 "
            "of each class, across 10 publishers."
        )


def check_publisher():
    print()
    print("PUBLISHER CLASSIFICATION DATA")
    print("=" * 54)

    publishers, _ = load_rows(
        "publishers.csv"
    )

    print(
        "Publisher registry size:",
        len(publishers)
    )

    source_labels, columns = load_rows(
        "source_reviewed_labels.csv"
    )

    if source_labels:
        labels = Counter(
            (
                row.get("label") or ""
            ).strip()
            for row in source_labels
        )

        print(
            "Source label counts:",
            dict(labels)
        )

    elif columns:
        print(
            "Source-reviewed-label template "
            "exists, but has no rows."
        )

    training, training_columns = load_rows(
        "reviewed_publishers.csv"
    )

    expected = {
        "domain",
        "reviewed_label"
    }

    if not training:
        print(
            "Publisher model: NOT READY. "
            "No reviewed publisher training "
            "records found."
        )

    elif not expected.issubset(
        training_columns
    ):
        print(
            "Publisher model: NOT READY. "
            "Expected domain and "
            "reviewed_label columns."
        )

    else:
        valid = [
            row
            for row in training
            if (
                row.get("domain")
                and row.get("reviewed_label")
                in {"0", "1"}
            )
        ]

        domain_count = len({
            row["domain"].strip().casefold()
            for row in valid
        })

        classes = Counter(
            row["reviewed_label"]
            for row in valid
        )

        print(
            "Labeled publishers:",
            domain_count
        )

        print(
            "Class counts:",
            dict(classes)
        )

        if (
            domain_count < 30
            or len(classes) < 2
        ):
            print(
                "Publisher model: NOT READY "
                "(requires 30 different "
                "publishers and two "
                "reviewed classes)."
            )

        else:
            print(
                "Publisher model: Initial "
                "row-count and class checks passed; "
                "feature schema must still "
                "be checked."
            )


if __name__ == "__main__":
    check_identity()
    check_publisher()
