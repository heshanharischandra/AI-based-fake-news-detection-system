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

CSV_PATH = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "real_manual_line_labels.csv"
)

CHARSET_PATH = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.replace(
        "\ufeff",
        ""
    )

    text = unicodedata.normalize(
        "NFC",
        text
    )

    return text.strip()


# ============================================================
# FILE HASH
# ============================================================

def sha256_file(path):

    hasher = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "REAL LABELED SINHALA OCR DATASET VALIDATION"
    )

    print(
        "=" * 72
    )

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{CSV_PATH}"
        )

    if not CHARSET_PATH.exists():

        raise FileNotFoundError(
            f"Charset not found:\n{CHARSET_PATH}"
        )

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        CSV_PATH,
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
        "keep"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing columns: "
            f"{missing_columns}"
        )

    df["ground_truth"] = (
        df["ground_truth"]
        .apply(
            normalize_text
        )
    )

    # --------------------------------------------------------
    # Load charset
    # --------------------------------------------------------

    with open(
        CHARSET_PATH,
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
    # Validation
    # --------------------------------------------------------

    missing_images = []

    unreadable_images = []

    empty_labels = []

    unseen_character_records = []

    image_hashes = {}

    valid_rows = 0

    for _, row in df.iterrows():

        sample_id = normalize_text(
            row["sample_id"]
        )

        relative_path = normalize_text(
            row["image_path"]
        )

        ground_truth = normalize_text(
            row["ground_truth"]
        )

        image_path = (
            OCR_ROOT
            / relative_path
        )

        # ----------------------------
        # Image exists
        # ----------------------------

        if not image_path.exists():

            missing_images.append(
                sample_id
            )

            continue

        # ----------------------------
        # Image readable
        # ----------------------------

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:

            unreadable_images.append(
                sample_id
            )

            continue

        # ----------------------------
        # Ground truth exists
        # ----------------------------

        if not ground_truth:

            empty_labels.append(
                sample_id
            )

            continue

        # ----------------------------
        # Charset coverage
        # ----------------------------

        unseen = sorted(
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

        if unseen:

            unseen_character_records.append({
                "sample_id":
                    sample_id,

                "characters":
                    unseen
            })

        # ----------------------------
        # Exact image duplicate check
        # ----------------------------

        file_hash = sha256_file(
            image_path
        )

        image_hashes.setdefault(
            file_hash,
            []
        ).append(
            sample_id
        )

        valid_rows += 1

    # --------------------------------------------------------
    # Exact duplicate images
    # --------------------------------------------------------

    duplicate_image_groups = [
        sample_ids

        for sample_ids
        in image_hashes.values()

        if len(sample_ids) > 1
    ]

    # --------------------------------------------------------
    # Repeated labels
    # --------------------------------------------------------

    repeated_labels = (

        df[
            df["ground_truth"].ne("")
        ]
        .groupby(
            "ground_truth"
        )[
            "sample_id"
        ]
        .apply(list)
    )

    repeated_labels = {
        text: sample_ids

        for text, sample_ids
        in repeated_labels.items()

        if len(sample_ids) > 1
    }

    # --------------------------------------------------------
    # Source coverage
    # --------------------------------------------------------

    source_counts = (
        df.groupby(
            "source_id"
        )
        .size()
        .sort_index()
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        f"\nCSV rows         : "
        f"{len(df)}"
    )

    print(
        f"Validated rows   : "
        f"{valid_rows}"
    )

    print(
        f"Source images    : "
        f"{df['source_id'].nunique()}"
    )

    print(
        f"Missing images   : "
        f"{len(missing_images)}"
    )

    print(
        f"Unreadable images: "
        f"{len(unreadable_images)}"
    )

    print(
        f"Empty labels     : "
        f"{len(empty_labels)}"
    )

    print(
        f"Unseen charset   : "
        f"{len(unseen_character_records)}"
    )

    print(
        f"Duplicate images : "
        f"{len(duplicate_image_groups)}"
    )

    print(
        "\nSource distribution:"
    )

    print(
        source_counts.to_string()
    )

    # --------------------------------------------------------
    # Unseen characters
    # --------------------------------------------------------

    if unseen_character_records:

        print(
            "\nUNSEEN CHARACTERS:"
        )

        for record in unseen_character_records:

            print(
                record[
                    "sample_id"
                ],
                "->",
                record[
                    "characters"
                ]
            )

    # --------------------------------------------------------
    # Duplicate images
    # --------------------------------------------------------

    if duplicate_image_groups:

        print(
            "\nEXACT DUPLICATE IMAGE GROUPS:"
        )

        for group in duplicate_image_groups:

            print(
                "  ",
                group
            )

    # --------------------------------------------------------
    # Repeated labels
    # --------------------------------------------------------

    print(
        "\nRepeated ground-truth labels:"
    )

    if repeated_labels:

        for text, sample_ids in (
            repeated_labels.items()
        ):

            print(
                f"\n{text}"
            )

            print(
                "  ",
                sample_ids
            )

    else:

        print(
            "None"
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    critical_problem = (
        len(missing_images) > 0
        or len(unreadable_images) > 0
        or len(empty_labels) > 0
        or len(unseen_character_records) > 0
        or len(duplicate_image_groups) > 0
    )

    print(
        "\n" + "=" * 72
    )

    if critical_problem:

        print(
            "DATASET VALIDATION: NEEDS REVIEW"
        )

    else:

        print(
            "DATASET VALIDATION: PASSED"
        )

    print(
        "=" * 72
    )