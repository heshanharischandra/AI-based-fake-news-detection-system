from pathlib import Path
from datetime import datetime

import json

import cv2
import numpy as np
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

TRAIN_CSV = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "splits_v2"
    / "real_finetune_train_v2.csv"
)

VALIDATION_CSV = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "splits_v2"
    / "real_finetune_validation_v2.csv"
)

AUGMENTED_IMAGE_DIR = (
    OCR_ROOT
    / "real_training_augmented_v2"
)

AUGMENTATION_META_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "augmentation_v2"
)

AUGMENTED_ONLY_CSV = (
    AUGMENTATION_META_DIR
    / "real_augmented_only_v2.csv"
)

COMBINED_TRAIN_CSV = (
    AUGMENTATION_META_DIR
    / "real_finetune_train_augmented_v2.csv"
)

MANIFEST_JSON = (
    AUGMENTATION_META_DIR
    / "augmentation_v2_manifest.json"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

AUGMENTATIONS_PER_SAMPLE = 12

np.random.seed(
    RANDOM_SEED
)


# ============================================================
# SETUP
# ============================================================

def setup():

    AUGMENTED_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    AUGMENTATION_META_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CLEAN PREVIOUS V2 OUTPUTS ONLY
# ============================================================

def clean_old_outputs():

    removed = 0

    for path in AUGMENTED_IMAGE_DIR.glob(
        "augv2_*.png"
    ):

        path.unlink()
        removed += 1

    for path in [
        AUGMENTED_ONLY_CSV,
        COMBINED_TRAIN_CSV,
        MANIFEST_JSON
    ]:

        if path.exists():
            path.unlink()

    print(
        f"Old V2 augmented images removed: "
        f"{removed}"
    )


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def ensure_uint8(image):

    return np.clip(
        image,
        0,
        255
    ).astype(
        np.uint8
    )


def adjust_brightness(
    image,
    value
):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    ).astype(
        np.int16
    )

    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2] + value,
        0,
        255
    )

    hsv = hsv.astype(
        np.uint8
    )

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )


def adjust_contrast(
    image,
    alpha
):

    return cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=0
    )


def gaussian_blur(
    image,
    kernel_size
):

    return cv2.GaussianBlur(
        image,
        (
            kernel_size,
            kernel_size
        ),
        0
    )


def add_gaussian_noise(
    image,
    sigma
):

    noise = np.random.normal(
        0,
        sigma,
        image.shape
    )

    noisy = (
        image.astype(
            np.float32
        )
        + noise
    )

    return ensure_uint8(
        noisy
    )


def rotate_image(
    image,
    angle
):

    height, width = (
        image.shape[:2]
    )

    center = (
        width / 2.0,
        height / 2.0
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    return cv2.warpAffine(
        image,
        matrix,
        (
            width,
            height
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )


def resize_degrade_restore(
    image,
    scale
):

    height, width = (
        image.shape[:2]
    )

    small_width = max(
        1,
        int(
            width * scale
        )
    )

    small_height = max(
        1,
        int(
            height * scale
        )
    )

    small = cv2.resize(
        image,
        (
            small_width,
            small_height
        ),
        interpolation=cv2.INTER_AREA
    )

    return cv2.resize(
        small,
        (
            width,
            height
        ),
        interpolation=cv2.INTER_LINEAR
    )


def sharpen_image(
    image
):

    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )

    return cv2.filter2D(
        image,
        -1,
        kernel
    )


def jpeg_compress(
    image,
    quality
):

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            int(
                cv2.IMWRITE_JPEG_QUALITY
            ),
            quality
        ]
    )

    if not success:

        raise RuntimeError(
            "JPEG encoding failed."
        )

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )

    if decoded is None:

        raise RuntimeError(
            "JPEG decoding failed."
        )

    return decoded


# ============================================================
# CREATE AUGMENTATIONS
# ============================================================

def create_augmentations(
    image
):

    return [

        (
            "brightness_minus_25",
            adjust_brightness(
                image,
                -25
            )
        ),

        (
            "brightness_plus_25",
            adjust_brightness(
                image,
                25
            )
        ),

        (
            "contrast_low",
            adjust_contrast(
                image,
                0.80
            )
        ),

        (
            "contrast_high",
            adjust_contrast(
                image,
                1.20
            )
        ),

        (
            "gaussian_blur_3",
            gaussian_blur(
                image,
                3
            )
        ),

        (
            "gaussian_blur_5",
            gaussian_blur(
                image,
                5
            )
        ),

        (
            "gaussian_noise",
            add_gaussian_noise(
                image,
                sigma=7.0
            )
        ),

        (
            "jpeg_quality_70",
            jpeg_compress(
                image,
                70
            )
        ),

        (
            "downscale_restore",
            resize_degrade_restore(
                image,
                0.70
            )
        ),

        (
            "rotation_minus_1",
            rotate_image(
                image,
                -1.0
            )
        ),

        (
            "rotation_plus_1",
            rotate_image(
                image,
                1.0
            )
        ),

        (
            "sharpen",
            sharpen_image(
                image
            )
        )
    ]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "EXPANDED REAL OCR V2 TRAINING AUGMENTATION"
    )

    print(
        "=" * 72
    )

    setup()

    if not TRAIN_CSV.exists():

        raise FileNotFoundError(
            f"V2 training CSV not found:\n"
            f"{TRAIN_CSV}"
        )

    if not VALIDATION_CSV.exists():

        raise FileNotFoundError(
            f"V2 validation CSV not found:\n"
            f"{VALIDATION_CSV}"
        )

    clean_old_outputs()

    # --------------------------------------------------------
    # Load V2 train and validation
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_CSV,
        encoding="utf-8-sig"
    )

    validation_df = pd.read_csv(
        VALIDATION_CSV,
        encoding="utf-8-sig"
    )

    train_df.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()
        for column in train_df.columns
    ]

    validation_df.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()
        for column in validation_df.columns
    ]

    print(
        f"\nV2 original train samples : "
        f"{len(train_df)}"
    )

    print(
        f"V2 validation samples     : "
        f"{len(validation_df)}"
    )

    # --------------------------------------------------------
    # Critical source leakage check before augmentation
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

    source_overlap = (
        train_sources
        & validation_sources
    )

    if source_overlap:

        raise RuntimeError(
            f"STOP: source leakage exists "
            f"before augmentation: "
            f"{source_overlap}"
        )

    print(
        "Pre-augmentation source leakage: "
        "NONE"
    )

    # --------------------------------------------------------
    # Prepare original rows
    # --------------------------------------------------------

    original_df = (
        train_df.copy()
    )

    original_df[
        "is_augmented"
    ] = 0

    original_df[
        "augmentation_type"
    ] = "original"

    original_df[
        "parent_sample_id"
    ] = original_df[
        "sample_id"
    ]

    # --------------------------------------------------------
    # Generate fresh V2 augmentation
    # --------------------------------------------------------

    augmented_rows = []

    for row_number, row in train_df.iterrows():

        sample_id = str(
            row[
                "sample_id"
            ]
        ).strip()

        relative_path = str(
            row[
                "image_path"
            ]
        ).strip()

        source_id = str(
            row[
                "source_id"
            ]
        ).strip()

        ground_truth = str(
            row[
                "ground_truth"
            ]
        ).strip()

        source_image_path = (
            OCR_ROOT
            / relative_path
        )

        if not source_image_path.exists():

            raise FileNotFoundError(
                f"Training image missing:\n"
                f"{source_image_path}"
            )

        image = cv2.imread(
            str(
                source_image_path
            ),
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise RuntimeError(
                f"Could not read:\n"
                f"{source_image_path}"
            )

        variants = create_augmentations(
            image
        )

        if (
            len(variants)
            != AUGMENTATIONS_PER_SAMPLE
        ):

            raise RuntimeError(
                "Unexpected augmentation count."
            )

        print(
            f"\nProcessing "
            f"{row_number + 1}/"
            f"{len(train_df)}: "
            f"{sample_id}"
        )

        for version_number, (
            augmentation_type,
            augmented_image
        ) in enumerate(
            variants,
            start=1
        ):

            augmented_sample_id = (
                f"augv2_"
                f"{sample_id}_"
                f"v{version_number:02d}"
            )

            filename = (
                f"{augmented_sample_id}.png"
            )

            output_path = (
                AUGMENTED_IMAGE_DIR
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
                    f"Could not save:\n"
                    f"{output_path}"
                )

            augmented_rows.append(
                {
                    "sample_id":
                        augmented_sample_id,

                    "image_path":
                        (
                            Path(
                                "real_training_augmented_v2"
                            )
                            / filename
                        ).as_posix(),

                    "source_id":
                        source_id,

                    "ground_truth":
                        ground_truth,

                    "language":
                        row.get(
                            "language",
                            "si"
                        ),

                    "is_augmented":
                        1,

                    "augmentation_type":
                        augmentation_type,

                    "parent_sample_id":
                        sample_id
                }
            )

        print(
            f"Generated variants: "
            f"{len(variants)}"
        )

    # --------------------------------------------------------
    # Build augmented dataframe
    # --------------------------------------------------------

    augmented_df = pd.DataFrame(
        augmented_rows
    )

    expected_augmented = (
        len(train_df)
        * AUGMENTATIONS_PER_SAMPLE
    )

    if len(
        augmented_df
    ) != expected_augmented:

        raise RuntimeError(
            "Generated augmentation count "
            "does not match expectation."
        )

    # --------------------------------------------------------
    # Common columns
    # --------------------------------------------------------

    columns = [
        "sample_id",
        "image_path",
        "source_id",
        "ground_truth",
        "language",
        "is_augmented",
        "augmentation_type",
        "parent_sample_id"
    ]

    for column in columns:

        if column not in original_df.columns:

            original_df[
                column
            ] = ""

        if column not in augmented_df.columns:

            augmented_df[
                column
            ] = ""

    # --------------------------------------------------------
    # Combined training data
    # --------------------------------------------------------

    combined_df = pd.concat(
        [
            original_df[
                columns
            ],
            augmented_df[
                columns
            ]
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Post-augmentation source leakage check
    # --------------------------------------------------------

    combined_sources = set(
        combined_df[
            "source_id"
        ]
        .astype(str)
    )

    post_overlap = (
        combined_sources
        & validation_sources
    )

    if post_overlap:

        raise RuntimeError(
            f"STOP: V2 validation leakage "
            f"detected after augmentation: "
            f"{post_overlap}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    augmented_df.to_csv(
        AUGMENTED_ONLY_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    combined_df.to_csv(
        COMBINED_TRAIN_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = {

        "created_at":
            datetime.now().isoformat(),

        "random_seed":
            RANDOM_SEED,

        "original_train_samples":
            len(
                train_df
            ),

        "original_train_sources":
            len(
                train_sources
            ),

        "validation_samples":
            len(
                validation_df
            ),

        "validation_sources":
            len(
                validation_sources
            ),

        "augmentations_per_sample":
            AUGMENTATIONS_PER_SAMPLE,

        "generated_augmented_samples":
            len(
                augmented_df
            ),

        "combined_training_samples":
            len(
                combined_df
            ),

        "validation_augmented":
            False,

        "a9_external_holdout_augmented":
            False,

        "source_leakage":
            False,

        "augmentation_types": [
            "brightness_minus_25",
            "brightness_plus_25",
            "contrast_low",
            "contrast_high",
            "gaussian_blur_3",
            "gaussian_blur_5",
            "gaussian_noise",
            "jpeg_quality_70",
            "downscale_restore",
            "rotation_minus_1",
            "rotation_plus_1",
            "sharpen"
        ]
    }

    with open(
        MANIFEST_JSON,
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
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "EXPANDED REAL OCR V2 AUGMENTATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nOriginal V2 train samples : "
        f"{len(train_df)}"
    )

    print(
        f"Train sources             : "
        f"{len(train_sources)}"
    )

    print(
        f"Augmentations per sample  : "
        f"{AUGMENTATIONS_PER_SAMPLE}"
    )

    print(
        f"Augmented samples         : "
        f"{len(augmented_df)}"
    )

    print(
        f"Combined training samples : "
        f"{len(combined_df)}"
    )

    print(
        f"Validation samples        : "
        f"{len(validation_df)}"
    )

    print(
        "\nValidation augmented      : NO"
    )

    print(
        "A9 holdout augmented       : NO"
    )

    print(
        "Source leakage             : NONE"
    )

    print(
        "\nAugmented image folder:"
    )

    print(
        AUGMENTED_IMAGE_DIR
    )

    print(
        "\nCombined V2 training CSV:"
    )

    print(
        COMBINED_TRAIN_CSV
    )

    print(
        "\nManifest:"
    )

    print(
        MANIFEST_JSON
    )