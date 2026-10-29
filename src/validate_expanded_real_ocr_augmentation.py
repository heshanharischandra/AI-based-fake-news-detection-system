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

AUGMENTATION_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "augmentation_v2"
)

TRAIN_CSV = (
    SPLIT_DIR
    / "real_finetune_train_v2.csv"
)

VALIDATION_CSV = (
    SPLIT_DIR
    / "real_finetune_validation_v2.csv"
)

AUGMENTED_ONLY_CSV = (
    AUGMENTATION_DIR
    / "real_augmented_only_v2.csv"
)

COMBINED_TRAIN_CSV = (
    AUGMENTATION_DIR
    / "real_finetune_train_augmented_v2.csv"
)

MANIFEST_JSON = (
    AUGMENTATION_DIR
    / "augmentation_v2_manifest.json"
)

CHARSET_JSON = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)

AUGMENTED_IMAGE_DIR = (
    OCR_ROOT
    / "real_training_augmented_v2"
)


# ============================================================
# EXPECTED COUNTS
# ============================================================

EXPECTED_TRAIN = 71
EXPECTED_VALIDATION = 22
EXPECTED_AUGMENTED = 852
EXPECTED_COMBINED = 923
EXPECTED_AUGMENTATIONS_PER_PARENT = 12


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


def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file:\n{path}"
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
    print("EXPANDED REAL OCR V2 AUGMENTATION VALIDATION")
    print("=" * 72)

    # --------------------------------------------------------
    # Load files
    # --------------------------------------------------------

    train_df = load_csv(
        TRAIN_CSV
    )

    validation_df = load_csv(
        VALIDATION_CSV
    )

    augmented_df = load_csv(
        AUGMENTED_ONLY_CSV
    )

    combined_df = load_csv(
        COMBINED_TRAIN_CSV
    )

    with open(
        MANIFEST_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        manifest = json.load(
            file
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
    # Basic counts
    # --------------------------------------------------------

    actual_image_files = list(
        AUGMENTED_IMAGE_DIR.glob(
            "augv2_*.png"
        )
    )

    print(
        f"\nOriginal V2 train rows : "
        f"{len(train_df)}"
    )

    print(
        f"Validation rows        : "
        f"{len(validation_df)}"
    )

    print(
        f"Augmented CSV rows     : "
        f"{len(augmented_df)}"
    )

    print(
        f"Augmented image files  : "
        f"{len(actual_image_files)}"
    )

    print(
        f"Combined train rows    : "
        f"{len(combined_df)}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = {
        "sample_id",
        "image_path",
        "source_id",
        "ground_truth"
    }

    for name, df in [
        ("train", train_df),
        ("validation", validation_df),
        ("augmented", augmented_df),
        ("combined", combined_df)
    ]:

        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:

            raise ValueError(
                f"{name} CSV missing columns: "
                f"{missing}"
            )

    # --------------------------------------------------------
    # Source leakage
    # --------------------------------------------------------

    train_sources = set(
        train_df[
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

    augmented_sources = set(
        augmented_df[
            "source_id"
        ]
        .astype(str)
    )

    combined_sources = set(
        combined_df[
            "source_id"
        ]
        .astype(str)
    )

    original_source_overlap = (
        train_sources
        & validation_sources
    )

    augmented_validation_overlap = (
        augmented_sources
        & validation_sources
    )

    combined_validation_overlap = (
        combined_sources
        & validation_sources
    )

    unexpected_augmented_sources = (
        augmented_sources
        - train_sources
    )

    # --------------------------------------------------------
    # Sample ID leakage
    # --------------------------------------------------------

    train_sample_ids = set(
        train_df[
            "sample_id"
        ]
        .astype(str)
    )

    validation_sample_ids = set(
        validation_df[
            "sample_id"
        ]
        .astype(str)
    )

    augmented_sample_ids = set(
        augmented_df[
            "sample_id"
        ]
        .astype(str)
    )

    sample_overlap_train_validation = (
        train_sample_ids
        & validation_sample_ids
    )

    sample_overlap_aug_validation = (
        augmented_sample_ids
        & validation_sample_ids
    )

    # --------------------------------------------------------
    # Duplicate sample IDs
    # --------------------------------------------------------

    duplicate_augmented_ids = int(
        augmented_df[
            "sample_id"
        ]
        .duplicated()
        .sum()
    )

    duplicate_combined_ids = int(
        combined_df[
            "sample_id"
        ]
        .duplicated()
        .sum()
    )

    # --------------------------------------------------------
    # Validate images / labels / charset
    # --------------------------------------------------------

    missing_images = []
    unreadable_images = []
    empty_labels = []
    unseen_charset_records = []

    for _, row in combined_df.iterrows():

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

        if image.size == 0:

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

    # --------------------------------------------------------
    # Parent validation
    # --------------------------------------------------------

    invalid_parent_ids = []

    incorrect_parent_counts = {}

    if (
        "parent_sample_id"
        not in augmented_df.columns
    ):

        raise ValueError(
            "parent_sample_id missing "
            "from augmented CSV."
        )

    parent_ids = (
        augmented_df[
            "parent_sample_id"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    for parent_id in parent_ids:

        if parent_id not in train_sample_ids:

            invalid_parent_ids.append(
                parent_id
            )

    parent_counts = (
        augmented_df
        .groupby(
            "parent_sample_id"
        )
        .size()
    )

    for parent_id, count in (
        parent_counts.items()
    ):

        if int(count) != (
            EXPECTED_AUGMENTATIONS_PER_PARENT
        ):

            incorrect_parent_counts[
                str(parent_id)
            ] = int(count)

    parents_without_augmentation = (
        train_sample_ids
        - set(
            parent_counts.index.astype(str)
        )
    )

    # --------------------------------------------------------
    # Check combined originals
    # --------------------------------------------------------

    combined_original_df = combined_df[
        combined_df[
            "is_augmented"
        ]
        .astype(str)
        .isin(
            [
                "0",
                "0.0",
                "False",
                "false"
            ]
        )
    ]

    combined_augmented_df = combined_df[
        combined_df[
            "is_augmented"
        ]
        .astype(str)
        .isin(
            [
                "1",
                "1.0",
                "True",
                "true"
            ]
        )
    ]

    # --------------------------------------------------------
    # Manifest checks
    # --------------------------------------------------------

    manifest_problem = not (

        manifest.get(
            "original_train_samples"
        ) == EXPECTED_TRAIN

        and manifest.get(
            "validation_samples"
        ) == EXPECTED_VALIDATION

        and manifest.get(
            "generated_augmented_samples"
        ) == EXPECTED_AUGMENTED

        and manifest.get(
            "combined_training_samples"
        ) == EXPECTED_COMBINED

        and manifest.get(
            "source_leakage"
        ) is False
    )

    # --------------------------------------------------------
    # Count checks
    # --------------------------------------------------------

    count_problem = not (

        len(train_df)
        == EXPECTED_TRAIN

        and len(validation_df)
        == EXPECTED_VALIDATION

        and len(augmented_df)
        == EXPECTED_AUGMENTED

        and len(actual_image_files)
        == EXPECTED_AUGMENTED

        and len(combined_df)
        == EXPECTED_COMBINED

        and len(combined_original_df)
        == EXPECTED_TRAIN

        and len(combined_augmented_df)
        == EXPECTED_AUGMENTED
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
        f"\nMissing images                  : "
        f"{len(missing_images)}"
    )

    print(
        f"Unreadable images               : "
        f"{len(unreadable_images)}"
    )

    print(
        f"Empty labels                    : "
        f"{len(empty_labels)}"
    )

    print(
        f"Unseen charset records          : "
        f"{len(unseen_charset_records)}"
    )

    print(
        f"Duplicate augmented sample IDs  : "
        f"{duplicate_augmented_ids}"
    )

    print(
        f"Duplicate combined sample IDs   : "
        f"{duplicate_combined_ids}"
    )

    print(
        f"Invalid parent IDs              : "
        f"{len(invalid_parent_ids)}"
    )

    print(
        f"Wrong augmentation counts       : "
        f"{len(incorrect_parent_counts)}"
    )

    print(
        f"Parents without augmentation    : "
        f"{len(parents_without_augmentation)}"
    )

    print(
        f"Original source leakage         : "
        f"{len(original_source_overlap)}"
    )

    print(
        f"Augmented-validation leakage    : "
        f"{len(augmented_validation_overlap)}"
    )

    print(
        f"Combined-validation leakage     : "
        f"{len(combined_validation_overlap)}"
    )

    print(
        f"Unexpected augmented sources    : "
        f"{len(unexpected_augmented_sources)}"
    )

    print(
        f"Train-validation sample overlap : "
        f"{len(sample_overlap_train_validation)}"
    )

    print(
        f"Aug-validation sample overlap   : "
        f"{len(sample_overlap_aug_validation)}"
    )

    print(
        f"Manifest problem                : "
        f"{manifest_problem}"
    )

    print(
        f"Count problem                   : "
        f"{count_problem}"
    )

    # --------------------------------------------------------
    # Details only when needed
    # --------------------------------------------------------

    if unseen_charset_records:

        print(
            "\nUNSEEN CHARACTERS:"
        )

        for record in unseen_charset_records:

            print(
                record
            )

    if incorrect_parent_counts:

        print(
            "\nWRONG PARENT COUNTS:"
        )

        for parent_id, count in (
            incorrect_parent_counts.items()
        ):

            print(
                f"{parent_id}: {count}"
            )

    if augmented_validation_overlap:

        print(
            "\nLEAKED VALIDATION SOURCES:"
        )

        for source_id in sorted(
            augmented_validation_overlap
        ):

            print(
                source_id
            )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    critical_problem = (

        count_problem

        or manifest_problem

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
            unseen_charset_records
        ) > 0

        or duplicate_augmented_ids > 0

        or duplicate_combined_ids > 0

        or len(
            invalid_parent_ids
        ) > 0

        or len(
            incorrect_parent_counts
        ) > 0

        or len(
            parents_without_augmentation
        ) > 0

        or len(
            original_source_overlap
        ) > 0

        or len(
            augmented_validation_overlap
        ) > 0

        or len(
            combined_validation_overlap
        ) > 0

        or len(
            unexpected_augmented_sources
        ) > 0

        or len(
            sample_overlap_train_validation
        ) > 0

        or len(
            sample_overlap_aug_validation
        ) > 0
    )

    print(
        "\n" + "=" * 72
    )

    if critical_problem:

        print(
            "V2 AUGMENTED DATASET VALIDATION: NEEDS REVIEW"
        )

    else:

        print(
            "V2 AUGMENTED DATASET VALIDATION: PASSED"
        )

        print(
            "923 training samples are ready "
            "for expanded real-domain fine-tuning."
        )

        print(
            "22 validation samples remain unseen."
        )

        print(
            "A9 remains an external holdout."
        )

    print(
        "=" * 72
    )