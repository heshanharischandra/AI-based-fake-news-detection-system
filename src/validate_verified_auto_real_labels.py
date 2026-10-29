from pathlib import Path
import hashlib
import json
import unicodedata

import cv2
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

VERIFIED_CSV = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "auto_line_labels_verified.csv"
)

CHARSET_JSON = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)

EXISTING_REAL_LINES_DIR = (
    OCR_ROOT
    / "real_training_lines"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    text = str(value)

    text = text.replace(
        "\ufeff",
        ""
    )

    text = unicodedata.normalize(
        "NFC",
        text
    )

    return text.strip()


def sha256_file(path):

    sha = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            sha.update(
                block
            )

    return sha.hexdigest()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "VERIFIED AUTO REAL OCR DATASET VALIDATION"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # Load verified data
    # --------------------------------------------------------

    if not VERIFIED_CSV.exists():

        raise FileNotFoundError(
            f"Verified CSV not found:\n"
            f"{VERIFIED_CSV}"
        )

    df = pd.read_csv(
        VERIFIED_CSV,
        encoding="utf-8-sig"
    )

    df.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()

        for column in df.columns
    ]

    required_columns = {
        "sample_id",
        "image_path",
        "source_id",
        "ground_truth",
        "keep",
        "review_status"
    }

    missing_columns = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    print(
        f"\nCSV rows: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Charset
    # --------------------------------------------------------

    if not CHARSET_JSON.exists():

        raise FileNotFoundError(
            f"Charset file not found:\n"
            f"{CHARSET_JSON}"
        )

    with open(
        CHARSET_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        charset_data = json.load(
            file
        )

    known_characters = set(
        charset_data[
            "characters"
        ]
    )

    # --------------------------------------------------------
    # Validation containers
    # --------------------------------------------------------

    missing_images = []
    unreadable_images = []
    empty_labels = []
    nonaccepted_rows = []
    duplicate_sample_ids = []
    unseen_charset_records = []

    seen_sample_ids = set()

    current_hash_map = {}

    # ========================================================
    # VALIDATE EACH VERIFIED SAMPLE
    # ========================================================

    for index, row in df.iterrows():

        sample_id = normalize_text(
            row[
                "sample_id"
            ]
        )

        image_path_text = normalize_text(
            row[
                "image_path"
            ]
        )

        source_id = normalize_text(
            row[
                "source_id"
            ]
        )

        ground_truth = normalize_text(
            row[
                "ground_truth"
            ]
        )

        review_status = normalize_text(
            row[
                "review_status"
            ]
        ).lower()

        keep_value = normalize_text(
            row[
                "keep"
            ]
        )

        # ----------------------------------------------------
        # Sample ID uniqueness
        # ----------------------------------------------------

        if sample_id in seen_sample_ids:

            duplicate_sample_ids.append(
                sample_id
            )

        seen_sample_ids.add(
            sample_id
        )

        # ----------------------------------------------------
        # Review state
        # ----------------------------------------------------

        keep_is_true = (
            keep_value
            in {
                "1",
                "1.0",
                "true",
                "True"
            }
        )

        if (
            review_status != "accepted"
            or not keep_is_true
        ):

            nonaccepted_rows.append(
                sample_id
            )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        if not ground_truth:

            empty_labels.append(
                sample_id
            )

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image_path = (
            OCR_ROOT
            / image_path_text
        )

        if not image_path.exists():

            missing_images.append(
                sample_id
            )

            continue

        image = cv2.imread(
            str(
                image_path
            ),
            cv2.IMREAD_COLOR
        )

        if image is None:

            unreadable_images.append(
                sample_id
            )

            continue

        height, width = (
            image.shape[:2]
        )

        if (
            height <= 0
            or width <= 0
        ):

            unreadable_images.append(
                sample_id
            )

            continue

        # ----------------------------------------------------
        # Exact image hashes
        # ----------------------------------------------------

        file_hash = sha256_file(
            image_path
        )

        current_hash_map.setdefault(
            file_hash,
            []
        ).append(
            sample_id
        )

        # ----------------------------------------------------
        # Charset
        # ----------------------------------------------------

        unseen_characters = sorted(
            {
                character
                for character
                in ground_truth
                if character
                not in known_characters
            },
            key=lambda character:
                ord(character)
        )

        if unseen_characters:

            unseen_charset_records.append(
                {
                    "sample_id":
                        sample_id,

                    "source_id":
                        source_id,

                    "ground_truth":
                        ground_truth,

                    "characters":
                        unseen_characters,

                    "codepoints": [
                        f"U+{ord(character):04X}"
                        for character
                        in unseen_characters
                    ]
                }
            )

    # ========================================================
    # DUPLICATES WITHIN NEW VERIFIED SET
    # ========================================================

    duplicate_groups = []

    for file_hash, sample_ids in (
        current_hash_map.items()
    ):

        if len(sample_ids) > 1:

            duplicate_groups.append(
                sample_ids
            )

    # ========================================================
    # CHECK AGAINST EXISTING CLEAN REAL LINE IMAGES
    # ========================================================

    existing_hashes = {}

    if EXISTING_REAL_LINES_DIR.exists():

        for image_path in (
            EXISTING_REAL_LINES_DIR.glob(
                "*.png"
            )
        ):

            file_hash = sha256_file(
                image_path
            )

            existing_hashes.setdefault(
                file_hash,
                []
            ).append(
                image_path.stem
            )

    cross_dataset_duplicates = []

    for file_hash, new_sample_ids in (
        current_hash_map.items()
    ):

        if file_hash in existing_hashes:

            cross_dataset_duplicates.append(
                {
                    "new_samples":
                        new_sample_ids,

                    "existing_samples":
                        existing_hashes[
                            file_hash
                        ]
                }
            )

    # ========================================================
    # REPEATED TEXT - INFORMATION ONLY
    # ========================================================

    normalized_labels = (
        df[
            "ground_truth"
        ]
        .fillna("")
        .astype(str)
        .map(
            normalize_text
        )
    )

    repeated_text = (
        normalized_labels[
            normalized_labels.ne("")
        ]
        .value_counts()
    )

    repeated_text = (
        repeated_text[
            repeated_text > 1
        ]
    )

    # ========================================================
    # SOURCE DISTRIBUTION
    # ========================================================

    source_distribution = (
        df[
            "source_id"
        ]
        .astype(str)
        .value_counts()
        .sort_index()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "VALIDATION SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"\nVerified rows              : "
        f"{len(df)}"
    )

    print(
        f"Sources represented        : "
        f"{df['source_id'].nunique()}"
    )

    print(
        f"Missing images             : "
        f"{len(missing_images)}"
    )

    print(
        f"Unreadable images          : "
        f"{len(unreadable_images)}"
    )

    print(
        f"Empty labels               : "
        f"{len(empty_labels)}"
    )

    print(
        f"Non-accepted rows          : "
        f"{len(nonaccepted_rows)}"
    )

    print(
        f"Duplicate sample IDs       : "
        f"{len(duplicate_sample_ids)}"
    )

    print(
        f"Unseen charset records     : "
        f"{len(unseen_charset_records)}"
    )

    print(
        f"Exact duplicate groups     : "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Duplicates vs old dataset  : "
        f"{len(cross_dataset_duplicates)}"
    )

    # --------------------------------------------------------
    # Source distribution
    # --------------------------------------------------------

    print(
        "\nSource distribution:"
    )

    print(
        source_distribution.to_string()
    )

    # --------------------------------------------------------
    # Unseen charset details
    # --------------------------------------------------------

    if unseen_charset_records:

        print(
            "\nUNSEEN CHARSET DETAILS:"
        )

        for record in unseen_charset_records:

            print(
                "\nSample:",
                record[
                    "sample_id"
                ]
            )

            print(
                "Text:"
            )

            print(
                record[
                    "ground_truth"
                ]
            )

            print(
                "Characters:",
                record[
                    "characters"
                ]
            )

            print(
                "Codepoints:",
                record[
                    "codepoints"
                ]
            )

    # --------------------------------------------------------
    # Duplicate image details
    # --------------------------------------------------------

    if duplicate_groups:

        print(
            "\nEXACT DUPLICATES WITHIN NEW SET:"
        )

        for group in duplicate_groups:

            print(
                group
            )

    if cross_dataset_duplicates:

        print(
            "\nDUPLICATES AGAINST OLD CLEAN DATASET:"
        )

        for record in cross_dataset_duplicates:

            print(
                record
            )

    # --------------------------------------------------------
    # Repeated labels
    # --------------------------------------------------------

    if len(
        repeated_text
    ) > 0:

        print(
            "\nRepeated ground-truth labels "
            "(informational only):"
        )

        print(
            repeated_text.to_string()
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    critical_problem = (

        len(df) != 62

        or len(
            missing_images
        ) > 0

        or len(
            unreadable_images
        ) > 0

        or len(
            empty_labels
        ) > 0

        or len(
            nonaccepted_rows
        ) > 0

        or len(
            duplicate_sample_ids
        ) > 0

        or len(
            unseen_charset_records
        ) > 0

        or len(
            duplicate_groups
        ) > 0

        or len(
            cross_dataset_duplicates
        ) > 0
    )

    print(
        "\n" + "=" * 72
    )

    if critical_problem:

        print(
            "VERIFIED AUTO DATASET: NEEDS REVIEW"
        )

    else:

        print(
            "VERIFIED AUTO DATASET: PASSED"
        )

        print(
            "The 62 verified real OCR samples "
            "are safe to merge into the real dataset."
        )

    print(
        "=" * 72
    )