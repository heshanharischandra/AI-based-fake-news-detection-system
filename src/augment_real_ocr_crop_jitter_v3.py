from pathlib import Path
from datetime import datetime
import json
import shutil

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

V2_AUGMENTATION_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "augmentation_v2"
)

V3_METADATA_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "augmentation_v3"
)

V3_IMAGE_DIR = (
    OCR_ROOT
    / "real_training_crop_jitter_v3"
)


ORIGINAL_TRAIN_CSV = (
    SPLIT_DIR
    / "real_finetune_train_v2.csv"
)

VALIDATION_CSV = (
    SPLIT_DIR
    / "real_finetune_validation_v2.csv"
)

V2_COMBINED_TRAIN_CSV = (
    V2_AUGMENTATION_DIR
    / "real_finetune_train_augmented_v2.csv"
)


V3_JITTER_ONLY_CSV = (
    V3_METADATA_DIR
    / "crop_jitter_only_v3.csv"
)

V3_COMBINED_TRAIN_CSV = (
    V3_METADATA_DIR
    / "real_finetune_train_augmented_v3.csv"
)

V3_MANIFEST_JSON = (
    V3_METADATA_DIR
    / "augmentation_v3_manifest.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

JITTER_PIXELS = 3

EXPECTED_ORIGINAL_REAL_TRAIN = 71
EXPECTED_V2_COMBINED = 923
EXPECTED_JITTER_PER_SAMPLE = 12
EXPECTED_NEW_JITTER = 852
EXPECTED_V3_COMBINED = 1775
EXPECTED_VALIDATION = 22


# ============================================================
# HELPERS
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Required CSV not found:\n"
            f"{path}"
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


def load_image_from_row(row):

    relative_path = normalize_text(
        row["image_path"]
    )

    image_path = (
        OCR_ROOT
        / relative_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Training image not found:\n"
            f"{image_path}"
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    return (
        image,
        image_path
    )


# ============================================================
# BORDER / CROP AUGMENTATIONS
# ============================================================

def pad_left(image, pixels):

    return cv2.copyMakeBorder(
        image,
        0,
        0,
        pixels,
        0,
        cv2.BORDER_REPLICATE
    )


def pad_right(image, pixels):

    return cv2.copyMakeBorder(
        image,
        0,
        0,
        0,
        pixels,
        cv2.BORDER_REPLICATE
    )


def pad_top(image, pixels):

    return cv2.copyMakeBorder(
        image,
        pixels,
        0,
        0,
        0,
        cv2.BORDER_REPLICATE
    )


def pad_bottom(image, pixels):

    return cv2.copyMakeBorder(
        image,
        0,
        pixels,
        0,
        0,
        cv2.BORDER_REPLICATE
    )


def crop_left(image, pixels):

    if image.shape[1] <= pixels + 5:
        return image.copy()

    return image[
        :,
        pixels:
    ].copy()


def crop_right(image, pixels):

    if image.shape[1] <= pixels + 5:
        return image.copy()

    return image[
        :,
        :-pixels
    ].copy()


def crop_top(image, pixels):

    if image.shape[0] <= pixels + 5:
        return image.copy()

    return image[
        pixels:,
        :
    ].copy()


def crop_bottom(image, pixels):

    if image.shape[0] <= pixels + 5:
        return image.copy()

    return image[
        :-pixels,
        :
    ].copy()


# ============================================================
# TRANSLATION AUGMENTATION
# ============================================================

def shift_image(
    image,
    dx,
    dy
):

    height, width = (
        image.shape[:2]
    )

    matrix = (
        cv2.getRotationMatrix2D(
            (
                width / 2.0,
                height / 2.0
            ),
            0,
            1.0
        )
    )

    matrix[
        0,
        2
    ] += dx

    matrix[
        1,
        2
    ] += dy

    shifted = cv2.warpAffine(
        image,
        matrix,
        (
            width,
            height
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return shifted


# ============================================================
# CREATE 12 CROP-JITTER VARIANTS
# ============================================================

def create_jitter_variants(
    image
):

    p = JITTER_PIXELS

    variants = {

        "pad_left_3":
            pad_left(
                image,
                p
            ),

        "pad_right_3":
            pad_right(
                image,
                p
            ),

        "pad_top_3":
            pad_top(
                image,
                p
            ),

        "pad_bottom_3":
            pad_bottom(
                image,
                p
            ),

        "crop_left_3":
            crop_left(
                image,
                p
            ),

        "crop_right_3":
            crop_right(
                image,
                p
            ),

        "crop_top_3":
            crop_top(
                image,
                p
            ),

        "crop_bottom_3":
            crop_bottom(
                image,
                p
            ),

        "shift_left_3":
            shift_image(
                image,
                -p,
                0
            ),

        "shift_right_3":
            shift_image(
                image,
                p,
                0
            ),

        "shift_up_3":
            shift_image(
                image,
                0,
                -p
            ),

        "shift_down_3":
            shift_image(
                image,
                0,
                p
            )
    }

    return variants


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 72)
    print("REAL OCR V3 CROP-JITTER AUGMENTATION")
    print("=" * 72)

    # --------------------------------------------------------
    # Create V3 directories
    # --------------------------------------------------------

    V3_METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    V3_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Remove ONLY old V3 generated jitter images
    # --------------------------------------------------------

    removed = 0

    for path in V3_IMAGE_DIR.glob(
        "jv3_*.png"
    ):

        path.unlink()

        removed += 1

    print(
        f"\nOld V3 jitter images removed: "
        f"{removed}"
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    original_df = load_csv(
        ORIGINAL_TRAIN_CSV
    )

    validation_df = load_csv(
        VALIDATION_CSV
    )

    v2_combined_df = load_csv(
        V2_COMBINED_TRAIN_CSV
    )

    print(
        f"\nOriginal real train samples : "
        f"{len(original_df)}"
    )

    print(
        f"Existing V2 combined train  : "
        f"{len(v2_combined_df)}"
    )

    print(
        f"Validation samples          : "
        f"{len(validation_df)}"
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
        (
            "original train",
            original_df
        ),
        (
            "validation",
            validation_df
        ),
        (
            "V2 combined",
            v2_combined_df
        )
    ]:

        missing = (
            required_columns
            - set(
                df.columns
            )
        )

        if missing:

            raise ValueError(
                f"{name} CSV missing columns: "
                f"{missing}"
            )

    # --------------------------------------------------------
    # Leakage check BEFORE augmentation
    # --------------------------------------------------------

    train_sources = set(
        original_df[
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

    if source_overlap:

        raise ValueError(
            "SOURCE LEAKAGE DETECTED BEFORE V3:\n"
            + "\n".join(
                sorted(
                    source_overlap
                )
            )
        )

    print(
        "Pre-V3 source leakage       : NONE"
    )

    # ========================================================
    # GENERATE JITTER IMAGES
    # ========================================================

    jitter_records = []

    for index, row in original_df.iterrows():

        sample_id = normalize_text(
            row[
                "sample_id"
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

        image, source_image_path = (
            load_image_from_row(
                row
            )
        )

        variants = (
            create_jitter_variants(
                image
            )
        )

        print(
            f"\nProcessing "
            f"{index + 1}/"
            f"{len(original_df)}: "
            f"{sample_id}"
        )

        generated_for_sample = 0

        for (
            augmentation_name,
            augmented_image
        ) in variants.items():

            augmented_sample_id = (
                f"jv3_{sample_id}_"
                f"{augmentation_name}"
            )

            filename = (
                f"{augmented_sample_id}.png"
            )

            output_path = (
                V3_IMAGE_DIR
                / filename
            )

            success = cv2.imwrite(
                str(
                    output_path
                ),
                augmented_image
            )

            if not success:

                raise RuntimeError(
                    "Could not save augmented image:\n"
                    f"{output_path}"
                )

            record = (
                row.to_dict()
            )

            record[
                "sample_id"
            ] = augmented_sample_id

            record[
                "image_path"
            ] = (
                Path(
                    "real_training_crop_jitter_v3"
                )
                / filename
            ).as_posix()

            record[
                "source_id"
            ] = source_id

            record[
                "ground_truth"
            ] = ground_truth

            record[
                "is_augmented"
            ] = 1

            record[
                "parent_sample_id"
            ] = sample_id

            record[
                "augmentation_type"
            ] = (
                f"crop_jitter_v3_"
                f"{augmentation_name}"
            )

            jitter_records.append(
                record
            )

            generated_for_sample += 1

        print(
            f"Generated crop-jitter variants: "
            f"{generated_for_sample}"
        )

    # ========================================================
    # SAVE JITTER-ONLY CSV
    # ========================================================

    jitter_df = pd.DataFrame(
        jitter_records
    )

    jitter_df.to_csv(
        V3_JITTER_ONLY_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # COMBINE EXISTING V2 + NEW V3 JITTER
    # ========================================================

    all_columns = list(
        dict.fromkeys(
            list(
                v2_combined_df.columns
            )
            + list(
                jitter_df.columns
            )
        )
    )

    v2_for_merge = (
        v2_combined_df
        .reindex(
            columns=all_columns
        )
    )

    jitter_for_merge = (
        jitter_df
        .reindex(
            columns=all_columns
        )
    )

    combined_v3_df = pd.concat(
        [
            v2_for_merge,
            jitter_for_merge
        ],
        ignore_index=True
    )

    combined_v3_df.to_csv(
        V3_COMBINED_TRAIN_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # POST-AUGMENTATION VALIDATION
    # ========================================================

    jitter_sources = set(
        jitter_df[
            "source_id"
        ]
        .astype(str)
    )

    combined_sources = set(
        combined_v3_df[
            "source_id"
        ]
        .astype(str)
    )

    jitter_validation_overlap = (
        jitter_sources
        & validation_sources
    )

    combined_validation_overlap = (
        combined_sources
        & validation_sources
    )

    duplicate_jitter_ids = int(
        jitter_df[
            "sample_id"
        ]
        .duplicated()
        .sum()
    )

    duplicate_combined_ids = int(
        combined_v3_df[
            "sample_id"
        ]
        .duplicated()
        .sum()
    )

    actual_v3_images = list(
        V3_IMAGE_DIR.glob(
            "jv3_*.png"
        )
    )

    # --------------------------------------------------------
    # Parent counts
    # --------------------------------------------------------

    parent_counts = (
        jitter_df
        .groupby(
            "parent_sample_id"
        )
        .size()
    )

    wrong_parent_counts = {
        str(parent):
            int(count)

        for parent, count
        in parent_counts.items()

        if int(count)
        != EXPECTED_JITTER_PER_SAMPLE
    }

    # ========================================================
    # EXPECTED COUNT CHECKS
    # ========================================================

    count_problem = not (

        len(original_df)
        == EXPECTED_ORIGINAL_REAL_TRAIN

        and len(v2_combined_df)
        == EXPECTED_V2_COMBINED

        and len(jitter_df)
        == EXPECTED_NEW_JITTER

        and len(actual_v3_images)
        == EXPECTED_NEW_JITTER

        and len(combined_v3_df)
        == EXPECTED_V3_COMBINED

        and len(validation_df)
        == EXPECTED_VALIDATION
    )

    leakage_problem = bool(
        source_overlap
        or jitter_validation_overlap
        or combined_validation_overlap
    )

    duplicate_problem = (
        duplicate_jitter_ids > 0
        or duplicate_combined_ids > 0
    )

    parent_problem = (
        len(
            wrong_parent_counts
        )
        > 0
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {

        "created_at":
            datetime.now().isoformat(),

        "augmentation_version":
            "V3 crop-jitter robustness",

        "jitter_pixels":
            JITTER_PIXELS,

        "original_real_train_samples":
            len(
                original_df
            ),

        "existing_v2_combined_samples":
            len(
                v2_combined_df
            ),

        "jitter_variants_per_original":
            EXPECTED_JITTER_PER_SAMPLE,

        "new_crop_jitter_samples":
            len(
                jitter_df
            ),

        "v3_combined_training_samples":
            len(
                combined_v3_df
            ),

        "validation_samples":
            len(
                validation_df
            ),

        "train_sources":
            len(
                train_sources
            ),

        "validation_sources":
            len(
                validation_sources
            ),

        "source_leakage":
            leakage_problem,

        "duplicate_sample_ids":
            duplicate_problem,

        "wrong_parent_counts":
            len(
                wrong_parent_counts
            ),

        "a9_used":
            False,

        "augmentations": [
            "pad_left_3",
            "pad_right_3",
            "pad_top_3",
            "pad_bottom_3",
            "crop_left_3",
            "crop_right_3",
            "crop_top_3",
            "crop_bottom_3",
            "shift_left_3",
            "shift_right_3",
            "shift_up_3",
            "shift_down_3"
        ]
    }

    with open(
        V3_MANIFEST_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=4
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("V3 CROP-JITTER AUGMENTATION SUMMARY")
    print("=" * 72)

    print(
        f"\nOriginal real train samples : "
        f"{len(original_df)}"
    )

    print(
        f"Existing V2 combined       : "
        f"{len(v2_combined_df)}"
    )

    print(
        f"New crop-jitter samples    : "
        f"{len(jitter_df)}"
    )

    print(
        f"Actual jitter image files  : "
        f"{len(actual_v3_images)}"
    )

    print(
        f"V3 combined train samples  : "
        f"{len(combined_v3_df)}"
    )

    print(
        f"Validation samples         : "
        f"{len(validation_df)}"
    )

    print(
        f"Train sources              : "
        f"{len(train_sources)}"
    )

    print(
        f"Validation sources         : "
        f"{len(validation_sources)}"
    )

    print(
        f"\nJitter-validation leakage  : "
        f"{len(jitter_validation_overlap)}"
    )

    print(
        f"Combined-validation leakage: "
        f"{len(combined_validation_overlap)}"
    )

    print(
        f"Duplicate jitter IDs       : "
        f"{duplicate_jitter_ids}"
    )

    print(
        f"Duplicate combined IDs     : "
        f"{duplicate_combined_ids}"
    )

    print(
        f"Wrong parent counts        : "
        f"{len(wrong_parent_counts)}"
    )

    print(
        f"Count problem              : "
        f"{count_problem}"
    )

    print(
        f"Leakage problem            : "
        f"{leakage_problem}"
    )

    print(
        f"Duplicate problem          : "
        f"{duplicate_problem}"
    )

    print(
        f"Parent problem             : "
        f"{parent_problem}"
    )

    critical_problem = (
        count_problem
        or leakage_problem
        or duplicate_problem
        or parent_problem
    )

    print()
    print("=" * 72)

    if critical_problem:

        print(
            "V3 DATASET STATUS: NEEDS REVIEW"
        )

    else:

        print(
            "V3 DATASET STATUS: PASSED"
        )

        print(
            "Crop-jitter training dataset "
            "is ready for V3 fine-tuning."
        )

        print(
            "Validation set was not augmented."
        )

        print(
            "A9 holdout was not used."
        )

    print("=" * 72)

    print(
        "\nV3 jitter images:"
    )

    print(
        V3_IMAGE_DIR
    )

    print(
        "\nV3 combined training CSV:"
    )

    print(
        V3_COMBINED_TRAIN_CSV
    )

    print(
        "\nV3 manifest:"
    )

    print(
        V3_MANIFEST_JSON
    )