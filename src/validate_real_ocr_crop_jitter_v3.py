from pathlib import Path
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

SPLIT_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "splits_v2"
)

V2_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "augmentation_v2"
)

V3_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "augmentation_v3"
)

ORIGINAL_TRAIN_CSV = (
    SPLIT_DIR
    / "real_finetune_train_v2.csv"
)

VALIDATION_CSV = (
    SPLIT_DIR
    / "real_finetune_validation_v2.csv"
)

V2_COMBINED_CSV = (
    V2_DIR
    / "real_finetune_train_augmented_v2.csv"
)

V3_JITTER_ONLY_CSV = (
    V3_DIR
    / "crop_jitter_only_v3.csv"
)

V3_COMBINED_CSV = (
    V3_DIR
    / "real_finetune_train_augmented_v3.csv"
)

V3_MANIFEST_JSON = (
    V3_DIR
    / "augmentation_v3_manifest.json"
)

CHARSET_JSON = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)

V3_IMAGE_DIR = (
    OCR_ROOT
    / "real_training_crop_jitter_v3"
)


# ============================================================
# EXPECTED COUNTS
# ============================================================

EXPECTED_ORIGINAL = 71
EXPECTED_VALIDATION = 22
EXPECTED_V2 = 923
EXPECTED_JITTER = 852
EXPECTED_V3 = 1775
EXPECTED_JITTER_PER_PARENT = 12


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

    text = text.replace(
        "\u200b",
        ""
    )

    text = unicodedata.normalize(
        "NFC",
        text
    )

    return text.strip()


def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig"
    )

    df.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()

        for column in df.columns
    ]

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 72)
    print("REAL OCR V3 CROP-JITTER DATASET VALIDATION")
    print("=" * 72)

    original_df = load_csv(
        ORIGINAL_TRAIN_CSV
    )

    validation_df = load_csv(
        VALIDATION_CSV
    )

    v2_df = load_csv(
        V2_COMBINED_CSV
    )

    jitter_df = load_csv(
        V3_JITTER_ONLY_CSV
    )

    v3_df = load_csv(
        V3_COMBINED_CSV
    )

    with open(
        CHARSET_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        charset_data = json.load(
            file
        )

    with open(
        V3_MANIFEST_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        manifest = json.load(
            file
        )

    known_characters = set(
        charset_data[
            "characters"
        ]
    )

    # ========================================================
    # BASIC COUNTS
    # ========================================================

    actual_jitter_images = list(
        V3_IMAGE_DIR.glob(
            "jv3_*.png"
        )
    )

    print(
        f"\nOriginal train rows : "
        f"{len(original_df)}"
    )

    print(
        f"V2 combined rows    : "
        f"{len(v2_df)}"
    )

    print(
        f"V3 jitter rows      : "
        f"{len(jitter_df)}"
    )

    print(
        f"V3 jitter images    : "
        f"{len(actual_jitter_images)}"
    )

    print(
        f"V3 combined rows    : "
        f"{len(v3_df)}"
    )

    print(
        f"Validation rows     : "
        f"{len(validation_df)}"
    )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = {
        "sample_id",
        "image_path",
        "source_id",
        "ground_truth"
    }

    for name, df in [
        ("original", original_df),
        ("validation", validation_df),
        ("V2 combined", v2_df),
        ("V3 jitter", jitter_df),
        ("V3 combined", v3_df)
    ]:

        missing = (
            required_columns
            - set(
                df.columns
            )
        )

        if missing:

            raise ValueError(
                f"{name} missing columns: "
                f"{missing}"
            )

    if (
        "parent_sample_id"
        not in jitter_df.columns
    ):

        raise ValueError(
            "V3 jitter CSV missing "
            "parent_sample_id."
        )

    # ========================================================
    # IDS
    # ========================================================

    original_ids = set(
        original_df[
            "sample_id"
        ]
        .astype(str)
    )

    validation_ids = set(
        validation_df[
            "sample_id"
        ]
        .astype(str)
    )

    v2_ids = set(
        v2_df[
            "sample_id"
        ]
        .astype(str)
    )

    jitter_ids = set(
        jitter_df[
            "sample_id"
        ]
        .astype(str)
    )

    v3_ids = set(
        v3_df[
            "sample_id"
        ]
        .astype(str)
    )

    duplicate_jitter_ids = int(
        jitter_df[
            "sample_id"
        ]
        .duplicated()
        .sum()
    )

    duplicate_v3_ids = int(
        v3_df[
            "sample_id"
        ]
        .duplicated()
        .sum()
    )

    # V2 must be fully preserved inside V3
    missing_v2_ids = (
        v2_ids
        - v3_ids
    )

    # All jitter records must be inside V3
    missing_jitter_ids = (
        jitter_ids
        - v3_ids
    )

    extra_v3_ids = (
        v3_ids
        - (
            v2_ids
            | jitter_ids
        )
    )

    train_validation_id_overlap = (
        v3_ids
        & validation_ids
    )

    # ========================================================
    # SOURCE LEAKAGE
    # ========================================================

    train_sources = set(
        v3_df[
            "source_id"
        ]
        .astype(str)
    )

    validation_sources = set(
        validation_df[
            "source_id"
        ]
        .astype(str)
    )

    source_overlap = (
        train_sources
        & validation_sources
    )

    # ========================================================
    # PARENT VALIDATION
    # ========================================================

    invalid_parent_ids = []

    parent_ids = (
        jitter_df[
            "parent_sample_id"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    for parent_id in parent_ids:

        if parent_id not in original_ids:

            invalid_parent_ids.append(
                parent_id
            )

    parent_counts = (
        jitter_df
        .groupby(
            "parent_sample_id"
        )
        .size()
    )

    wrong_parent_counts = {}

    for parent_id, count in (
        parent_counts.items()
    ):

        if (
            int(count)
            != EXPECTED_JITTER_PER_PARENT
        ):

            wrong_parent_counts[
                str(parent_id)
            ] = int(count)

    parents_without_jitter = (
        original_ids
        - set(
            parent_counts
            .index
            .astype(str)
        )
    )

    # ========================================================
    # IMAGE + LABEL VALIDATION
    # ========================================================

    missing_images = []
    unreadable_images = []
    empty_labels = []
    unseen_charset_records = []

    for _, row in v3_df.iterrows():

        sample_id = normalize_text(
            row[
                "sample_id"
            ]
        )

        relative_path = normalize_text(
            row[
                "image_path"
            ]
        )

        ground_truth = normalize_text(
            row[
                "ground_truth"
            ]
        )

        image_path = (
            OCR_ROOT
            / relative_path
        )

        if not image_path.exists():

            missing_images.append(
                sample_id
            )

            continue

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if (
            image is None
            or image.size == 0
        ):

            unreadable_images.append(
                sample_id
            )

            continue

        if not ground_truth:

            empty_labels.append(
                sample_id
            )

            continue

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

            unseen_charset_records.append(
                {
                    "sample_id":
                        sample_id,

                    "characters":
                        unseen
                }
            )

    # ========================================================
    # MANIFEST VALIDATION
    # ========================================================

    manifest_problem = not (

        manifest.get(
            "original_real_train_samples"
        ) == EXPECTED_ORIGINAL

        and manifest.get(
            "existing_v2_combined_samples"
        ) == EXPECTED_V2

        and manifest.get(
            "new_crop_jitter_samples"
        ) == EXPECTED_JITTER

        and manifest.get(
            "v3_combined_training_samples"
        ) == EXPECTED_V3

        and manifest.get(
            "validation_samples"
        ) == EXPECTED_VALIDATION

        and manifest.get(
            "source_leakage"
        ) is False

        and manifest.get(
            "a9_used"
        ) is False
    )

    # ========================================================
    # COUNT VALIDATION
    # ========================================================

    count_problem = not (

        len(original_df)
        == EXPECTED_ORIGINAL

        and len(validation_df)
        == EXPECTED_VALIDATION

        and len(v2_df)
        == EXPECTED_V2

        and len(jitter_df)
        == EXPECTED_JITTER

        and len(actual_jitter_images)
        == EXPECTED_JITTER

        and len(v3_df)
        == EXPECTED_V3
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("V3 VALIDATION SUMMARY")
    print("=" * 72)

    print(
        f"\nMissing images                 : "
        f"{len(missing_images)}"
    )

    print(
        f"Unreadable images              : "
        f"{len(unreadable_images)}"
    )

    print(
        f"Empty labels                   : "
        f"{len(empty_labels)}"
    )

    print(
        f"Unseen charset records         : "
        f"{len(unseen_charset_records)}"
    )

    print(
        f"Duplicate jitter IDs           : "
        f"{duplicate_jitter_ids}"
    )

    print(
        f"Duplicate V3 combined IDs      : "
        f"{duplicate_v3_ids}"
    )

    print(
        f"Invalid parent IDs             : "
        f"{len(invalid_parent_ids)}"
    )

    print(
        f"Wrong parent counts            : "
        f"{len(wrong_parent_counts)}"
    )

    print(
        f"Parents without jitter         : "
        f"{len(parents_without_jitter)}"
    )

    print(
        f"Missing V2 rows from V3        : "
        f"{len(missing_v2_ids)}"
    )

    print(
        f"Missing jitter rows from V3    : "
        f"{len(missing_jitter_ids)}"
    )

    print(
        f"Unexpected rows in V3          : "
        f"{len(extra_v3_ids)}"
    )

    print(
        f"Train-validation ID overlap    : "
        f"{len(train_validation_id_overlap)}"
    )

    print(
        f"Train-validation source overlap: "
        f"{len(source_overlap)}"
    )

    print(
        f"Manifest problem               : "
        f"{manifest_problem}"
    )

    print(
        f"Count problem                  : "
        f"{count_problem}"
    )

    # ========================================================
    # DETAILS ONLY IF PROBLEMS
    # ========================================================

    if unseen_charset_records:

        print(
            "\nUNSEEN CHARSET RECORDS:"
        )

        for record in (
            unseen_charset_records
        ):

            print(
                record
            )

    if wrong_parent_counts:

        print(
            "\nWRONG PARENT COUNTS:"
        )

        for parent, count in (
            wrong_parent_counts.items()
        ):

            print(
                parent,
                count
            )

    if source_overlap:

        print(
            "\nLEAKED VALIDATION SOURCES:"
        )

        for source in sorted(
            source_overlap
        ):

            print(
                source
            )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    critical_problem = (

        len(missing_images) > 0

        or len(unreadable_images) > 0

        or len(empty_labels) > 0

        or len(unseen_charset_records) > 0

        or duplicate_jitter_ids > 0

        or duplicate_v3_ids > 0

        or len(invalid_parent_ids) > 0

        or len(wrong_parent_counts) > 0

        or len(parents_without_jitter) > 0

        or len(missing_v2_ids) > 0

        or len(missing_jitter_ids) > 0

        or len(extra_v3_ids) > 0

        or len(train_validation_id_overlap) > 0

        or len(source_overlap) > 0

        or manifest_problem

        or count_problem
    )

    print()
    print("=" * 72)

    if critical_problem:

        print(
            "V3 DATASET VALIDATION: NEEDS REVIEW"
        )

    else:

        print(
            "V3 DATASET VALIDATION: PASSED"
        )

        print(
            "1775 training samples are ready "
            "for V3 robustness fine-tuning."
        )

        print(
            "22 validation samples remain "
            "source-isolated and unaugmented."
        )

        print(
            "A9 was not included in training."
        )

    print("=" * 72)