from pathlib import Path
from datetime import datetime

import json

import cv2
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

REAL_SPLIT_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "splits"
)

SOURCE_TRAIN_CSV = (
    REAL_SPLIT_DIR
    / "real_finetune_train.csv"
)

AUGMENTED_IMAGE_DIR = (
    OCR_ROOT
    / "real_training_augmented"
)

AUGMENTATION_META_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "augmentation"
)

AUGMENTED_ONLY_CSV = (
    AUGMENTATION_META_DIR
    / "real_augmented_only.csv"
)

COMBINED_TRAIN_CSV = (
    AUGMENTATION_META_DIR
    / "real_finetune_train_augmented.csv"
)

MANIFEST_JSON = (
    AUGMENTATION_META_DIR
    / "augmentation_manifest.json"
)


# ============================================================
# 2. SETTINGS
# ============================================================

JPEG_TEMP_PATH = (
    AUGMENTATION_META_DIR
    / "_temp_jpeg.jpg"
)

RANDOM_SEED = 42

np.random.seed(
    RANDOM_SEED
)


# ============================================================
# 3. SETUP
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
# 4. CLEAN OLD AUGMENTED FILES
# ============================================================

def clean_old_outputs():

    removed = 0

    for path in AUGMENTED_IMAGE_DIR.glob(
        "aug_*.png"
    ):

        path.unlink()

        removed += 1

    if AUGMENTED_ONLY_CSV.exists():
        AUGMENTED_ONLY_CSV.unlink()

    if COMBINED_TRAIN_CSV.exists():
        COMBINED_TRAIN_CSV.unlink()

    if MANIFEST_JSON.exists():
        MANIFEST_JSON.unlink()

    if JPEG_TEMP_PATH.exists():
        JPEG_TEMP_PATH.unlink()

    print(
        f"Old augmented images removed: "
        f"{removed}"
    )


# ============================================================
# 5. IMAGE HELPERS
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

    result = cv2.convertScaleAbs(
        image,
        alpha=alpha,
        beta=0
    )

    return result


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

    rotated = cv2.warpAffine(
        image,
        matrix,
        (
            width,
            height
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


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

    restored = cv2.resize(
        small,
        (
            width,
            height
        ),
        interpolation=cv2.INTER_LINEAR
    )

    return restored


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
# 6. CREATE 12 AUGMENTATIONS
# ============================================================

def create_augmentations(
    image
):

    augmentations = [

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

    return augmentations


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "REAL SINHALA OCR TRAINING DATA AUGMENTATION"
    )

    print(
        "=" * 72
    )

    setup()

    if not SOURCE_TRAIN_CSV.exists():

        raise FileNotFoundError(
            f"Training CSV not found:\n"
            f"{SOURCE_TRAIN_CSV}"
        )

    clean_old_outputs()

    # --------------------------------------------------------
    # Load ONLY real training split
    # --------------------------------------------------------

    train_df = pd.read_csv(
        SOURCE_TRAIN_CSV,
        encoding="utf-8-sig"
    )

    train_df.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()

        for column in train_df.columns
    ]

    required_columns = {
        "sample_id",
        "image_path",
        "source_id",
        "ground_truth"
    }

    missing = (
        required_columns
        - set(
            train_df.columns
        )
    )

    if missing:

        raise ValueError(
            f"Missing columns: "
            f"{missing}"
        )

    print(
        f"\nOriginal real train samples: "
        f"{len(train_df)}"
    )

    # --------------------------------------------------------
    # Mark original rows
    # --------------------------------------------------------

    combined_original_df = (
        train_df.copy()
    )

    combined_original_df[
        "is_augmented"
    ] = 0

    combined_original_df[
        "augmentation_type"
    ] = "original"

    combined_original_df[
        "parent_sample_id"
    ] = combined_original_df[
        "sample_id"
    ]

    # --------------------------------------------------------
    # Generate augmentations
    # --------------------------------------------------------

    augmented_rows = []

    generated_count = 0

    for row_index, row in train_df.iterrows():

        original_sample_id = str(
            row[
                "sample_id"
            ]
        )

        relative_image_path = str(
            row[
                "image_path"
            ]
        )

        source_id = str(
            row[
                "source_id"
            ]
        )

        ground_truth = str(
            row[
                "ground_truth"
            ]
        )

        image_path = (
            OCR_ROOT
            / relative_image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found:\n"
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

        augmentations = (
            create_augmentations(
                image
            )
        )

        print(
            f"\nProcessing "
            f"{row_index + 1}/"
            f"{len(train_df)}: "
            f"{original_sample_id}"
        )

        for version_number, (
            augmentation_name,
            augmented_image
        ) in enumerate(
            augmentations,
            start=1
        ):

            augmented_sample_id = (
                f"aug_"
                f"{original_sample_id}_"
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
                str(output_path),
                augmented_image
            )

            if not success:

                raise RuntimeError(
                    f"Could not save:\n"
                    f"{output_path}"
                )

            augmented_rows.append({

                "sample_id":
                    augmented_sample_id,

                "image_path":
                    (
                        Path(
                            "real_training_augmented"
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
                    augmentation_name,

                "parent_sample_id":
                    original_sample_id
            })

            generated_count += 1

        print(
            f"Generated variants: "
            f"{len(augmentations)}"
        )

    # --------------------------------------------------------
    # Augmented dataframe
    # --------------------------------------------------------

    augmented_df = pd.DataFrame(
        augmented_rows
    )

    # --------------------------------------------------------
    # Save augmented-only CSV
    # --------------------------------------------------------

    augmented_df.to_csv(
        AUGMENTED_ONLY_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Align columns before combining
    # --------------------------------------------------------

    required_combined_columns = [
        "sample_id",
        "image_path",
        "source_id",
        "ground_truth",
        "language",
        "is_augmented",
        "augmentation_type",
        "parent_sample_id"
    ]

    for column in required_combined_columns:

        if column not in combined_original_df.columns:

            combined_original_df[
                column
            ] = ""

        if column not in augmented_df.columns:

            augmented_df[
                column
            ] = ""

    combined_df = pd.concat(
        [
            combined_original_df[
                required_combined_columns
            ],
            augmented_df[
                required_combined_columns
            ]
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Save combined training CSV
    # --------------------------------------------------------

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

        "original_real_train_samples":
            len(
                train_df
            ),

        "augmentations_per_sample":
            12,

        "generated_augmented_samples":
            len(
                augmented_df
            ),

        "combined_training_samples":
            len(
                combined_df
            ),

        "validation_data_augmented":
            False,

        "external_a9_holdout_augmented":
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
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "REAL OCR AUGMENTATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nOriginal real train samples : "
        f"{len(train_df)}"
    )

    print(
        f"Augmentations per sample    : "
        f"12"
    )

    print(
        f"Augmented samples generated : "
        f"{len(augmented_df)}"
    )

    print(
        f"Combined train samples      : "
        f"{len(combined_df)}"
    )

    print(
        "\nValidation augmented        : NO"
    )

    print(
        "A9 holdout augmented         : NO"
    )

    print(
        "\nAugmented images:"
    )

    print(
        AUGMENTED_IMAGE_DIR
    )

    print(
        "\nCombined training CSV:"
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