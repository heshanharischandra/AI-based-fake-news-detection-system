
import argparse
import csv
import io
import re
import sys
import urllib.request

from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET_URL = (
    "https://raw.githubusercontent.com/sTechLab/"
    "local-news-dataset/main/"
    "local_news_outlets_dataset.csv"
)

CITATION_URL = (
    "https://github.com/sTechLab/local-news-dataset"
)

TARGET = 1000

OUTPUT_DIR = ROOT / "data" / "imported_1000"

US_STATES = set((
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA "
    "KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM "
    "NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA "
    "WV WI"
).split())


# ============================================================
# OUTPUT COLUMN DEFINITIONS
# ============================================================

PUB_COLS = [
    "publisher_id",
    "publisher_name",
    "website_url",
    "language",
    "media_type",
    "website_status",
    "country",
    "original_outlet_id",
    "source_dataset_url",
    "original_website_url",
]

PAIR_COLS = [
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
    "original_outlet_id",
    "original_facebook_id",
    "historical_facebook_retrieved_date",
    "source_dataset_url",
    "source_row_number",
    "original_website_url",
]

REVIEW_COLS = PAIR_COLS + [
    "reviewer_id",
    "reviewed_at",
    "review_evidence_url",
    "review_notes",
]


# ============================================================
# CLEAN DATA
# ============================================================

def clean(value):

    value = str(
        value if value is not None else ""
    ).strip()

    missing_values = {
        "na",
        "nan",
        "none",
        "null",
        "n/a",
    }

    if value.casefold() in missing_values:
        return ""

    return value


# ============================================================
# FIX WEBSITE URL VALIDATION
# ============================================================

def normalize_website(raw):

    raw = clean(raw)

    if not raw:
        return ""

    if re.search(r"\s", raw):
        return ""

    # Some source records omit the URL scheme.
    # Adding https:// is normalization,
    # not confirmation that HTTPS works.

    if raw.startswith("//"):
        raw = "https:" + raw

    if not re.match(
        r"^https?://",
        raw,
        re.I
    ):

        if "://" in raw:
            return ""

        raw = "https://" + raw

    try:

        parsed = urlsplit(raw)

        hostname = parsed.hostname or ""

        if parsed.scheme.lower() not in (
            "http",
            "https"
        ):
            return ""

        if not hostname:
            return ""

        if "." not in hostname:
            return ""

        if parsed.username or parsed.password:
            return ""

        if parsed.port is not None:
            return ""

        if any(
            character in hostname
            for character in " /@"
        ):
            return ""

        return raw

    except ValueError:
        return ""


# ============================================================
# VALIDATE HISTORICAL FACEBOOK ID
# ============================================================

def parse_facebook_id(value):

    value = clean(value)

    if re.fullmatch(
        r"[0-9]+\.0+",
        value
    ):
        value = value.split(".")[0]

    # Keep only exact numeric IDs.
    # Do not invent IDs from scientific notation.

    if re.fullmatch(
        r"[0-9]{5,25}",
        value
    ):
        return value

    return ""


# ============================================================
# READ ORIGINAL DATASET
# ============================================================

def read_source(path):

    if path:

        print(
            "Reading source:",
            path,
            flush=True
        )

        content = path.read_bytes()

    else:

        print(
            "Downloading published historical source CSV...",
            flush=True
        )

        request = urllib.request.Request(
            DATASET_URL,
            headers={
                "User-Agent":
                "ResearchHistoricalDatasetImporter/1.0"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            content = response.read()

    decoded = content.decode(
        "utf-8-sig"
    )

    reader = csv.DictReader(
        io.StringIO(
            decoded,
            newline=""
        )
    )

    required = {
        "OutletId",
        "Name",
        "WebsiteUrl",
        "FacebookId",
        "State",
    }

    missing = required - set(
        reader.fieldnames or []
    )

    if missing:

        raise ValueError(
            "Missing source CSV columns: "
            + ", ".join(sorted(missing))
        )

    rows = list(reader)

    print(
        "Historical source rows loaded:",
        len(rows),
        flush=True
    )

    print(
        "Source column names:",
        ", ".join(reader.fieldnames),
        flush=True
    )

    print(
        "Sample raw values:"
    )

    for row in rows[:3]:

        print(
            "State =",
            repr(row.get("State")),
            "WebsiteUrl =",
            repr(row.get("WebsiteUrl")),
            "FacebookId =",
            repr(row.get("FacebookId"))
        )

    return rows


# ============================================================
# SELECT 1000 UNIQUE HISTORICAL RECORDS
# ============================================================

def select_1000(rows):

    buckets = defaultdict(list)

    failures = Counter()

    for line, row in enumerate(
        rows,
        start=2
    ):

        name = clean(
            row.get("Name")
        )

        state = clean(
            row.get("State")
        ).upper()

        raw_url = clean(
            row.get("WebsiteUrl")
        )

        website = normalize_website(
            raw_url
        )

        fb_id = parse_facebook_id(
            row.get("FacebookId")
        )

        outlet_id = clean(
            row.get("OutletId")
        )

        if not name:
            failures["missing_name"] += 1

        if state not in US_STATES:
            failures["non_us_or_missing_state"] += 1

        if not website:
            failures["invalid_or_missing_website"] += 1

        if not fb_id:
            failures["invalid_or_missing_facebook_id"] += 1

        if not outlet_id:
            failures["missing_outlet_id"] += 1

        if not (
            name
            and state in US_STATES
            and website
            and fb_id
            and outlet_id
        ):
            continue

        buckets[state].append(
            (
                line,
                row,
                website,
                fb_id,
                raw_url,
            )
        )

    print(
        "Filter diagnostics:",
        dict(failures),
        flush=True
    )

    passing = sum(
        len(items)
        for items in buckets.values()
    )

    print(
        "Records passing required fields:",
        passing,
        flush=True
    )

    seen_names = set()
    seen_domains = set()
    seen_facebook_ids = set()
    seen_outlets = set()

    offsets = defaultdict(int)

    selected = []

    # Cycle through states to avoid selecting
    # everything from one state.

    while len(selected) < TARGET:

        changed = False

        for state in sorted(buckets):

            items = buckets[state]

            while offsets[state] < len(items):

                (
                    line,
                    row,
                    website,
                    fb_id,
                    raw_url
                ) = items[offsets[state]]

                offsets[state] += 1

                hostname = (
                    urlsplit(website).hostname or ""
                ).casefold()

                hostname = hostname.removeprefix(
                    "www."
                )

                key_name = clean(
                    row["Name"]
                ).casefold()

                id_key = clean(
                    row["OutletId"]
                )

                if (
                    hostname in seen_domains
                    or key_name in seen_names
                    or fb_id in seen_facebook_ids
                    or id_key in seen_outlets
                ):

                    failures[
                        "duplicate_name_site_account_or_outlet"
                    ] += 1

                    continue

                selected.append(
                    (
                        line,
                        row,
                        website,
                        fb_id,
                        raw_url
                    )
                )

                seen_domains.add(hostname)

                seen_names.add(key_name)

                seen_facebook_ids.add(fb_id)

                seen_outlets.add(id_key)

                changed = True

                break

            if len(selected) == TARGET:
                break

        if not changed:
            break

    print(
        "Unique candidates selected:",
        len(selected),
        flush=True
    )

    if len(selected) < TARGET:

        raise RuntimeError(
            f"Only {len(selected)} unique records "
            "passed the quality checks. "
            "No fake records were added. "
            "See diagnostics above."
        )

    return selected


# ============================================================
# BUILD THE THREE CSV DATASETS
# ============================================================

def build_tables(selected):

    publishers = []
    pairs = []
    reviewed = []

    for index, item in enumerate(
        selected,
        start=1
    ):

        (
            source_line,
            row,
            website,
            fb_id,
            original_website
        ) = item

        publisher_id = f"EXT{index:06d}"

        name = clean(
            row["Name"]
        )

        outlet_id = clean(
            row["OutletId"]
        )

        historical_date = clean(
            row.get("FacebookRetrievedDate")
        )

        common = {
            "publisher_id": publisher_id,
            "publisher_name": name,
            "website_url": website,
            "language": "unknown",
        }

        # DATASET 1: publishers.csv

        publisher = {
            **common,

            "media_type":
                "historical_us_local_news",

            "website_status":
                "not_live_checked",

            "country": "US",

            "original_outlet_id": outlet_id,

            "source_dataset_url": CITATION_URL,

            "original_website_url":
                original_website,
        }

        publishers.append(publisher)

        # DATASET 2: identity_pairs.csv

        pair = {
            **common,

            "platform": "facebook",

            "account_url":
                "https://www.facebook.com/" + fb_id,

            "association_evidence":
                "historical_dataset_candidate",

            "identity_label": "unknown",

            "review_status": "pending",

            "evidence_url": "",

            "notes": (
                "Historical dataset association only. "
                "Current ownership and account URL "
                "have not been verified. "
                "Website URL scheme may have "
                "been normalized."
            ),

            "original_outlet_id": outlet_id,

            "original_facebook_id": fb_id,

            "historical_facebook_retrieved_date":
                historical_date,

            "source_dataset_url": CITATION_URL,

            "source_row_number": source_line,

            "original_website_url":
                original_website,
        }

        pairs.append(pair)

        # DATASET 3: identity_pairs_reviewed.csv

        review_record = {
            **pair,

            "reviewer_id": "",

            "reviewed_at": "",

            "review_evidence_url": "",

            "review_notes": "",
        }

        reviewed.append(review_record)

    return (
        publishers,
        pairs,
        reviewed
    )


# ============================================================
# WRITE CSV
# ============================================================

def write_csv(path, columns, rows):

    with path.open(
        "x",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional local source CSV"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR
    )

    args = parser.parse_args()

    output_dir = args.output_dir

    output_paths = [
        output_dir / "publishers.csv",

        output_dir / "identity_pairs.csv",

        output_dir / "identity_pairs_reviewed.csv",
    ]

    # Never overwrite existing research files.

    existing = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing:

        raise FileExistsError(
            "Output files already exist. "
            "Refusing to overwrite: "
            + ", ".join(map(str, existing))
        )

    source_rows = read_source(
        args.source
    )

    selected = select_1000(
        source_rows
    )

    publishers, pairs, reviewed = (
        build_tables(selected)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    newly_created = []

    try:

        for path, columns, rows in zip(
            output_paths,
            (
                PUB_COLS,
                PAIR_COLS,
                REVIEW_COLS
            ),
            (
                publishers,
                pairs,
                reviewed
            )
        ):

            write_csv(
                path,
                columns,
                rows
            )

            newly_created.append(path)

    except Exception:

        for path in newly_created:
            path.unlink(
                missing_ok=True
            )

        raise

    # Verify output counts.

    for path in output_paths:

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            count = sum(
                1
                for _ in csv.DictReader(file)
            )

        print(
            f"CREATED: {path} - "
            f"{count} data rows"
        )

    print()
    print("=" * 60)
    print("HISTORICAL DATASET GENERATION COMPLETE")
    print("=" * 60)

    print(
        "Publisher records:",
        len(publishers)
    )

    print(
        "Identity candidate records:",
        len(pairs)
    )

    print(
        "Review candidate records:",
        len(reviewed)
    )

    print()
    print("All review labels: unknown / pending")
    print("Current Facebook ownership: not verified")
    print("Original project datasets: not overwritten")


# ============================================================
# EXECUTE
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except (
        OSError,
        ValueError,
        RuntimeError
    ) as error:

        print(
            "Stopped safely:",
            error,
            file=sys.stderr
        )

        sys.exit(1)
