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

AUGMENTATION_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "augmentation"
)

AUGMENTED_ONLY_CSV = (
    AUGMENTATION_DIR
    / "real_augmented_only.csv"
)

COMBINED_TRAIN_CSV = (
    AUGMENTATION_DIR
    / "real_finetune_train_augmented.csv"
)

MANIFEST_JSON = (
    AUGMENTATION_DIR
    / "augmentation_manifest.json"
)

REAL_SPLIT_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "splits"
)

ORIGINAL_TRAIN_CSV = (
    REAL_SPLIT_DIR
    / "real_finetune_train.csv"
)

VALIDATION_CSV = (
    REAL_SPLIT_DIR
    / "real_finetune_validation.csv"
)

CHARSET_JSON = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)

AUGMENTED_IMAGE_DIR = (
    OCR_ROOT
    / "real_training_augmented"
)


# ============================================================
# NORMALIZE
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value)

    value = value.replace(
        "\ufeff",
        ""
    )

    value = unicodedata.normalize(
        "NFC",
        value
    )

    return value.strip()


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
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

    print(
        "=" * 72
    )

    print(
        "REAL OCR AUGMENTED DATASET VALIDATION"
    )

    print(
        "=" * 72
    )

    augmented_df = load_csv(
        AUGMENTED_ONLY_CSV
    )

    combined_df = load_csv(
        COMBINED_TRAIN_CSV
    )

    original_train_df = load_csv(
        ORIGINAL_TRAIN_CSV
    )

    validation_df = load_csv(
        VALIDATION_CSV
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    with open(
        MANIFEST_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        manifest = json.load(
            file
        )

    # --------------------------------------------------------
    # Charset
    # --------------------------------------------------------

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
    # Counts
    # --------------------------------------------------------

    actual_image_files = list(
        AUGMENTED_IMAGE_DIR.glob(
            "aug_*.png"
        )
    )

    print(
        f"\nOriginal train rows : "
        f"{len(original_train_df)}"
    )

    print(
        f"Augmented CSV rows  : "
        f"{len(augmented_df)}"
    )

    print(
        f"Augmented images    : "
        f"{len(actual_image_files)}"
    )

    print(
        f"Combined train rows : "
        f"{len(combined_df)}"
    )

    print(
        f"Validation rows     : "
        f"{len(validation_df)}"
    )

    print(
        f"Manifest augmented  : "
        f"{manifest.get('generated_augmented_samples')}"
    )

    print(
        f"Manifest combined   : "
        f"{manifest.get('combined_training_samples')}"
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

    missing_columns = (
        required_columns
        - set(
            augmented_df.columns
        )
    )

    if missing_columns:

        raise ValueError(
            f"Missing augmented columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Validate augmented samples
    # --------------------------------------------------------

    missing_images = []

    unreadable_images = []

    empty_labels = []

    unseen_records = []

    duplicate_sample_ids = []

    seen_sample_ids = set()

    for _, row in augmented_df.iterrows():

        sample_id = normalize_text(
            row[
                "sample_id"
            ]
        )

        image_path = (
            OCR_ROOT
            / normalize_text(
                row[
                    "image_path"
                ]
            )
        )

        ground_truth = normalize_text(
            row[
                "ground_truth"
            ]
        )

        # -------------------------
        # Duplicate sample IDs
        # -------------------------

        if sample_id in seen_sample_ids:

            duplicate_sample_ids.append(
                sample_id
            )

        seen_sample_ids.add(
            sample_id
        )

        # -------------------------
        # Image exists
        # -------------------------

        if not image_path.exists():

            missing_images.append(
                sample_id
            )

            continue

        # -------------------------
        # Image readable
        # -------------------------

        image = cv2.imread(
            str(image_path),
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

        if height <= 0 or width <= 0:

            unreadable_images.append(
                sample_id
            )

            continue

        # -------------------------
        # Ground truth
        # -------------------------

        if not ground_truth:

            empty_labels.append(
                sample_id
            )

            continue

        # -------------------------
        # Charset
        # -------------------------

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

            unseen_records.append({

                "sample_id":
                    sample_id,

                "characters":
                    unseen
            })

    # --------------------------------------------------------
    # Source leakage
    # --------------------------------------------------------

    original_train_sources = set(
        original_train_df[
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

    validation_sources = set(
        validation_df[
            "source_id"
        ]
        .astype(str)
    )

    augmented_validation_overlap = (
        augmented_sources
        & validation_sources
    )

    unexpected_augmented_sources = (
        augmented_sources
        - original_train_sources
    )

    # --------------------------------------------------------
    # Parent sample validation
    # --------------------------------------------------------

    original_sample_ids = set(
        original_train_df[
            "sample_id"
        ]
        .astype(str)
    )

    invalid_parent_ids = []

    if (
        "parent_sample_id"
        in augmented_df.columns
    ):

        for parent_id in augmented_df[
            "parent_sample_id"
        ].astype(str):

            if (
                parent_id
                not in original_sample_ids
            ):

                invalid_parent_ids.append(
                    parent_id
                )

    # --------------------------------------------------------
    # Expected augmentation count per parent
    # --------------------------------------------------------

    incorrect_parent_counts = {}

    if (
        "parent_sample_id"
        in augmented_df.columns
    ):

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

            if int(count) != 12:

                incorrect_parent_counts[
                    str(parent_id)
                ] = int(count)

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
        f"\nMissing images             : "
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
        f"Unseen charset records     : "
        f"{len(unseen_records)}"
    )

    print(
        f"Duplicate sample IDs       : "
        f"{len(duplicate_sample_ids)}"
    )

    print(
        f"Invalid parent IDs         : "
        f"{len(invalid_parent_ids)}"
    )

    print(
        f"Wrong augmentation counts  : "
        f"{len(incorrect_parent_counts)}"
    )

    print(
        f"Validation source leakage  : "
        f"{len(augmented_validation_overlap)}"
    )

    print(
        f"Unexpected train sources   : "
        f"{len(unexpected_augmented_sources)}"
    )

    # --------------------------------------------------------
    # Detailed errors
    # --------------------------------------------------------

    if augmented_validation_overlap:

        print(
            "\nLEAKED VALIDATION SOURCES:"
        )

        for source_id in sorted(
            augmented_validation_overlap
        ):

            print(
                f"  {source_id}"
            )

    if incorrect_parent_counts:

        print(
            "\nWRONG AUGMENTATION COUNTS:"
        )

        for parent_id, count in (
            incorrect_parent_counts.items()
        ):

            print(
                f"  {parent_id}: "
                f"{count}"
            )

    if unseen_records:

        print(
            "\nUNSEEN CHARACTERS:"
        )

        for record in unseen_records:

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
    # Count checks
    # --------------------------------------------------------

    count_problem = not (

        len(original_train_df)
        == 24

        and len(augmented_df)
        == 288

        and len(actual_image_files)
        == 288

        and len(combined_df)
        == 312

        and manifest.get(
            "generated_augmented_samples"
        )
        == 288

        and manifest.get(
            "combined_training_samples"
        )
        == 312
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    critical_problem = (

        count_problem

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
            unseen_records
        ) > 0

        or len(
            duplicate_sample_ids
        ) > 0

        or len(
            invalid_parent_ids
        ) > 0

        or len(
            incorrect_parent_counts
        ) > 0

        or len(
            augmented_validation_overlap
        ) > 0

        or len(
            unexpected_augmented_sources
        ) > 0
    )

    print(
        "\n" + "=" * 72
    )

    if critical_problem:

        print(
            "AUGMENTED DATASET VALIDATION: NEEDS REVIEW"
        )

    else:

        print(
            "AUGMENTED DATASET VALIDATION: PASSED"
        )

        print(
            "The augmented real OCR training "
            "dataset is ready for fine-tuning."
        )

    print(
        "=" * 72
    )