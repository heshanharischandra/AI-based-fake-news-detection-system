from pathlib import Path
import csv
import json
import hashlib

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

CLEAN_ROOT = (
    OCR_ROOT
    / "cleaned_real_ocr_v1"
)

TRAIN_CSV = (
    CLEAN_ROOT
    / "metadata"
    / "splits"
    / "clean_real_train_v1.csv"
)

VALIDATION_CSV = (
    CLEAN_ROOT
    / "metadata"
    / "splits"
    / "clean_real_validation_v1.csv"
)

AUGMENT_ROOT = (
    CLEAN_ROOT
    / "augmentation_v4"
)

AUGMENT_IMAGES_DIR = (
    AUGMENT_ROOT
    / "images"
)

AUGMENT_METADATA_DIR = (
    AUGMENT_ROOT
    / "metadata"
)

AUGMENTED_ONLY_CSV = (
    AUGMENT_METADATA_DIR
    / "clean_real_train_augmented_only_v4.csv"
)

COMBINED_TRAIN_CSV = (
    AUGMENT_METADATA_DIR
    / "clean_real_train_combined_v4.csv"
)

SUMMARY_JSON = (
    AUGMENT_METADATA_DIR
    / "augmentation_v4_summary.json"
)


# ============================================================
# SETTINGS
# ============================================================

AUGMENTATIONS_PER_SAMPLE = 12

EXPECTED_ORIGINAL_TRAIN_COUNT = 76

EXPECTED_VALIDATION_COUNT = 23


# ============================================================
# OUTPUT COLUMNS
# ============================================================

FIELDNAMES = [
    "sample_id",
    "image_path",
    "source_id",
    "language",
    "ground_truth",
    "parent_sample_id",
    "augmentation",
    "augmentation_index",
    "is_augmented",
]


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as file:

        return list(
            csv.DictReader(file)
        )


# ============================================================
# RESOLVE IMAGE PATH
# ============================================================

def resolve_image_path(record):

    relative_path = str(
        record.get(
            "image_path",
            ""
        )
    ).strip()

    if not relative_path:

        return None

    return (
        OCR_ROOT
        / Path(relative_path)
    )


# ============================================================
# DETERMINISTIC RANDOM GENERATOR
#
# Same sample + variant always creates the same augmentation.
# ============================================================

def create_rng(
    sample_id,
    variant_index
):

    seed_text = (
        f"{sample_id}_{variant_index}"
    )

    digest = hashlib.sha256(
        seed_text.encode(
            "utf-8"
        )
    ).digest()

    seed = int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False
    )

    return np.random.default_rng(
        seed
    )


# ============================================================
# SAFE BORDER COLOUR
# ============================================================

def get_border_colour(image):

    height, width = image.shape[:2]

    border_pixels = []

    border_pixels.append(
        image[
            0:min(3, height),
            :
        ].reshape(
            -1,
            image.shape[2]
        )
    )

    border_pixels.append(
        image[
            max(0, height - 3):height,
            :
        ].reshape(
            -1,
            image.shape[2]
        )
    )

    border_pixels.append(
        image[
            :,
            0:min(3, width)
        ].reshape(
            -1,
            image.shape[2]
        )
    )

    border_pixels.append(
        image[
            :,
            max(0, width - 3):width
        ].reshape(
            -1,
            image.shape[2]
        )
    )

    pixels = np.concatenate(
        border_pixels,
        axis=0
    )

    median_colour = np.median(
        pixels,
        axis=0
    )

    return tuple(
        int(value)
        for value
        in median_colour
    )


# ============================================================
# BRIGHTNESS / CONTRAST
# ============================================================

def adjust_brightness_contrast(
    image,
    alpha=1.0,
    beta=0.0
):

    result = (
        image.astype(
            np.float32
        )
        * float(alpha)
        + float(beta)
    )

    result = np.clip(
        result,
        0,
        255
    )

    return result.astype(
        np.uint8
    )


# ============================================================
# GAUSSIAN NOISE
# ============================================================

def add_gaussian_noise(
    image,
    sigma,
    rng
):

    noise = rng.normal(
        0.0,
        sigma,
        image.shape
    )

    result = (
        image.astype(
            np.float32
        )
        + noise
    )

    result = np.clip(
        result,
        0,
        255
    )

    return result.astype(
        np.uint8
    )


# ============================================================
# BLUR
# ============================================================

def apply_blur(
    image,
    sigma=0.6
):

    return cv2.GaussianBlur(
        image,
        (3, 3),
        sigmaX=sigma,
        sigmaY=sigma
    )


# ============================================================
# JPEG COMPRESSION
# ============================================================

def apply_jpeg_compression(
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
            int(
                quality
            )
        ]
    )

    if not success:

        raise RuntimeError(
            "JPEG augmentation failed."
        )

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR
    )

    if decoded is None:

        raise RuntimeError(
            "Could not decode JPEG augmentation."
        )

    return decoded


# ============================================================
# AFFINE JITTER
# ============================================================

def apply_affine(
    image,
    rotation=0.0,
    scale=1.0,
    tx_ratio=0.0,
    ty_ratio=0.0
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
        rotation,
        scale
    )

    matrix[
        0,
        2
    ] += (
        width
        * tx_ratio
    )

    matrix[
        1,
        2
    ] += (
        height
        * ty_ratio
    )

    border_colour = (
        get_border_colour(
            image
        )
    )

    result = cv2.warpAffine(
        image,
        matrix,
        (
            width,
            height
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_colour
    )

    return result


# ============================================================
# HORIZONTAL SCALE
#
# Simulates different text width / crop geometry while
# preserving the overall output canvas size.
# ============================================================

def apply_horizontal_scale(
    image,
    scale_factor
):

    height, width = (
        image.shape[:2]
    )

    new_width = max(
        1,
        int(
            round(
                width
                * scale_factor
            )
        )
    )

    resized = cv2.resize(
        image,
        (
            new_width,
            height
        ),
        interpolation=cv2.INTER_CUBIC
    )

    border_colour = (
        get_border_colour(
            image
        )
    )

    # --------------------------------------------------------
    # NEW IMAGE NARROWER -> CENTER PAD
    # --------------------------------------------------------

    if new_width < width:

        missing = (
            width
            - new_width
        )

        left = (
            missing // 2
        )

        right = (
            missing - left
        )

        result = cv2.copyMakeBorder(
            resized,
            0,
            0,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=border_colour
        )

    # --------------------------------------------------------
    # NEW IMAGE WIDER -> CENTER CROP
    # --------------------------------------------------------

    elif new_width > width:

        extra = (
            new_width
            - width
        )

        start = (
            extra // 2
        )

        result = resized[
            :,
            start:start + width
        ]

    else:

        result = resized

    return result


# ============================================================
# SHARPEN
# ============================================================

def apply_sharpen(image):

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=1.0
    )

    sharpened = cv2.addWeighted(
        image,
        1.35,
        blurred,
        -0.35,
        0
    )

    return sharpened


# ============================================================
# CREATE ONE AUGMENTED VARIANT
#
# Mild transformations only.
# We do NOT use strong rotation or perspective distortion
# because Sinhala glyph marks can easily be damaged.
# ============================================================

def create_augmentation(
    image,
    sample_id,
    variant_index
):

    rng = create_rng(
        sample_id,
        variant_index
    )

    if variant_index == 1:

        result = adjust_brightness_contrast(
            image,
            alpha=0.92,
            beta=-7
        )

        name = "brightness_darker"

    elif variant_index == 2:

        result = adjust_brightness_contrast(
            image,
            alpha=1.08,
            beta=7
        )

        name = "brightness_lighter"

    elif variant_index == 3:

        result = adjust_brightness_contrast(
            image,
            alpha=0.88,
            beta=4
        )

        name = "contrast_lower"

    elif variant_index == 4:

        result = adjust_brightness_contrast(
            image,
            alpha=1.12,
            beta=-3
        )

        name = "contrast_higher"

    elif variant_index == 5:

        result = apply_blur(
            image,
            sigma=0.55
        )

        name = "mild_blur"

    elif variant_index == 6:

        result = add_gaussian_noise(
            image,
            sigma=3.0,
            rng=rng
        )

        name = "mild_noise"

    elif variant_index == 7:

        result = apply_jpeg_compression(
            image,
            quality=78
        )

        name = "jpeg_compression"

    elif variant_index == 8:

        result = apply_affine(
            image,
            rotation=-0.55,
            scale=0.995,
            tx_ratio=-0.012,
            ty_ratio=0.025
        )

        name = "affine_left"

    elif variant_index == 9:

        result = apply_affine(
            image,
            rotation=0.55,
            scale=1.005,
            tx_ratio=0.012,
            ty_ratio=-0.025
        )

        name = "affine_right"

    elif variant_index == 10:

        result = apply_horizontal_scale(
            image,
            scale_factor=0.97
        )

        name = "horizontal_narrow"

    elif variant_index == 11:

        result = apply_horizontal_scale(
            image,
            scale_factor=1.03
        )

        name = "horizontal_wide"

    elif variant_index == 12:

        rotation = float(
            rng.uniform(
                -0.40,
                0.40
            )
        )

        tx_ratio = float(
            rng.uniform(
                -0.008,
                0.008
            )
        )

        ty_ratio = float(
            rng.uniform(
                -0.018,
                0.018
            )
        )

        alpha = float(
            rng.uniform(
                0.96,
                1.04
            )
        )

        beta = float(
            rng.uniform(
                -4.0,
                4.0
            )
        )

        result = apply_affine(
            image,
            rotation=rotation,
            scale=1.0,
            tx_ratio=tx_ratio,
            ty_ratio=ty_ratio
        )

        result = (
            adjust_brightness_contrast(
                result,
                alpha=alpha,
                beta=beta
            )
        )

        result = (
            add_gaussian_noise(
                result,
                sigma=2.0,
                rng=rng
            )
        )

        name = "combined_mild_jitter"

    else:

        raise ValueError(
            f"Unsupported augmentation index: "
            f"{variant_index}"
        )

    return (
        result,
        name
    )


# ============================================================
# STANDARDIZE ORIGINAL RECORD
# ============================================================

def create_original_manifest_record(
    record
):

    return {
        "sample_id":
            str(
                record.get(
                    "sample_id",
                    ""
                )
            ).strip(),

        "image_path":
            str(
                record.get(
                    "image_path",
                    ""
                )
            ).strip(),

        "source_id":
            str(
                record.get(
                    "source_id",
                    ""
                )
            ).strip(),

        "language":
            str(
                record.get(
                    "language",
                    "si"
                )
            ).strip(),

        "ground_truth":
            str(
                record.get(
                    "ground_truth",
                    ""
                )
            ).strip(),

        "parent_sample_id":
            str(
                record.get(
                    "sample_id",
                    ""
                )
            ).strip(),

        "augmentation":
            "original",

        "augmentation_index":
            0,

        "is_augmented":
            0,
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    path,
    records
):

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()

        writer.writerows(
            records
        )


# ============================================================
# MAIN AUGMENTATION
# ============================================================

def main():

    AUGMENT_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    AUGMENT_METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 72)
    print("CLEAN REAL OCR AUGMENTATION V4")
    print("=" * 72)

    train_records = load_csv(
        TRAIN_CSV
    )

    validation_records = load_csv(
        VALIDATION_CSV
    )

    print()
    print(
        "Original clean training samples:",
        len(
            train_records
        )
    )

    print(
        "Untouched validation samples:",
        len(
            validation_records
        )
    )

    # ========================================================
    # BASIC COUNT CHECK
    # ========================================================

    if (
        len(train_records)
        != EXPECTED_ORIGINAL_TRAIN_COUNT
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            "Expected",
            EXPECTED_ORIGINAL_TRAIN_COUNT,
            "training samples but found",
            len(train_records)
        )

    if (
        len(validation_records)
        != EXPECTED_VALIDATION_COUNT
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            "Expected",
            EXPECTED_VALIDATION_COUNT,
            "validation samples but found",
            len(validation_records)
        )

    # ========================================================
    # SOURCE LEAKAGE CHECK BEFORE AUGMENTATION
    # ========================================================

    train_sources = {
        str(
            row.get(
                "source_id",
                ""
            )
        ).strip()
        for row in train_records
    }

    validation_sources = {
        str(
            row.get(
                "source_id",
                ""
            )
        ).strip()
        for row in validation_records
    }

    source_overlap = (
        train_sources
        & validation_sources
    )

    if source_overlap:

        print()
        print(
            "ERROR: Train/validation source leakage:"
        )

        for source in sorted(
            source_overlap
        ):

            print(
                " ",
                source
            )

        raise RuntimeError(
            "Source leakage detected."
        )

    print(
        "Source leakage before augmentation: 0"
    )

    # ========================================================
    # CHECK TRAIN RECORDS
    # ========================================================

    missing_images = []

    blank_labels = []

    for record in train_records:

        sample_id = str(
            record.get(
                "sample_id",
                ""
            )
        ).strip()

        ground_truth = str(
            record.get(
                "ground_truth",
                ""
            )
        ).strip()

        image_path = (
            resolve_image_path(
                record
            )
        )

        if not ground_truth:

            blank_labels.append(
                sample_id
            )

        if (
            image_path is None
            or not image_path.exists()
        ):

            missing_images.append(
                sample_id
            )

    if blank_labels:

        raise RuntimeError(
            "Blank training labels found:\n"
            + "\n".join(
                blank_labels
            )
        )

    if missing_images:

        raise RuntimeError(
            "Missing training images found:\n"
            + "\n".join(
                missing_images
            )
        )

    # ========================================================
    # GENERATE AUGMENTATIONS
    # ========================================================

    augmented_records = []

    combined_records = []

    # Keep original 76 records in combined training manifest.
    for record in train_records:

        combined_records.append(
            create_original_manifest_record(
                record
            )
        )

    print()
    print(
        "Generating augmentations..."
    )

    for sample_number, record in enumerate(
        train_records,
        start=1
    ):

        sample_id = str(
            record[
                "sample_id"
            ]
        ).strip()

        source_id = str(
            record[
                "source_id"
            ]
        ).strip()

        language = str(
            record.get(
                "language",
                "si"
            )
        ).strip()

        ground_truth = str(
            record[
                "ground_truth"
            ]
        ).strip()

        source_path = (
            resolve_image_path(
                record
            )
        )

        image = cv2.imread(
            str(
                source_path
            ),
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise RuntimeError(
                f"Could not read image:\n"
                f"{source_path}"
            )

        print()
        print(
            f"[{sample_number:02d}/"
            f"{len(train_records):02d}] "
            f"{sample_id}"
        )

        for variant_index in range(
            1,
            AUGMENTATIONS_PER_SAMPLE + 1
        ):

            (
                augmented_image,
                augmentation_name
            ) = create_augmentation(
                image,
                sample_id,
                variant_index
            )

            augmented_sample_id = (
                f"{sample_id}"
                f"_aug_v4_"
                f"{variant_index:02d}"
            )

            filename = (
                f"{augmented_sample_id}.png"
            )

            output_path = (
                AUGMENT_IMAGES_DIR
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

            relative_path = (
                Path(
                    "cleaned_real_ocr_v1"
                )
                / "augmentation_v4"
                / "images"
                / filename
            )

            augmented_record = {

                "sample_id":
                    augmented_sample_id,

                "image_path":
                    relative_path.as_posix(),

                "source_id":
                    source_id,

                "language":
                    language,

                "ground_truth":
                    ground_truth,

                "parent_sample_id":
                    sample_id,

                "augmentation":
                    augmentation_name,

                "augmentation_index":
                    variant_index,

                "is_augmented":
                    1,
            }

            augmented_records.append(
                augmented_record
            )

            combined_records.append(
                augmented_record
            )

        print(
            "  Created:",
            AUGMENTATIONS_PER_SAMPLE,
            "variants"
        )

    # ========================================================
    # EXPECTED COUNTS
    # ========================================================

    expected_augmented_count = (
        len(train_records)
        * AUGMENTATIONS_PER_SAMPLE
    )

    expected_combined_count = (
        len(train_records)
        + expected_augmented_count
    )

    if (
        len(augmented_records)
        != expected_augmented_count
    ):

        raise RuntimeError(
            "Augmented record count mismatch."
        )

    if (
        len(combined_records)
        != expected_combined_count
    ):

        raise RuntimeError(
            "Combined record count mismatch."
        )

    # ========================================================
    # VALIDATE AUGMENTED IMAGES
    # ========================================================

    generated_files = list(
        AUGMENT_IMAGES_DIR.glob(
            "*.png"
        )
    )

    if (
        len(generated_files)
        != expected_augmented_count
    ):

        raise RuntimeError(
            "Generated image file count does not "
            "match augmented metadata count."
        )

    # ========================================================
    # AUGMENTED SOURCE CHECK
    # ========================================================

    augmented_sources = {
        row[
            "source_id"
        ]
        for row in augmented_records
    }

    augmented_validation_overlap = (
        augmented_sources
        & validation_sources
    )

    if augmented_validation_overlap:

        raise RuntimeError(
            "VALIDATION SOURCE LEAKAGE FOUND "
            "AFTER AUGMENTATION."
        )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    save_csv(
        AUGMENTED_ONLY_CSV,
        augmented_records
    )

    save_csv(
        COMBINED_TRAIN_CSV,
        combined_records
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "version":
            "clean_real_ocr_augmentation_v4",

        "original_training_samples":
            len(train_records),

        "validation_samples_untouched":
            len(validation_records),

        "augmentations_per_sample":
            AUGMENTATIONS_PER_SAMPLE,

        "augmented_samples":
            len(augmented_records),

        "combined_training_samples":
            len(combined_records),

        "training_source_count":
            len(train_sources),

        "validation_source_count":
            len(validation_sources),

        "source_leakage_before_augmentation":
            len(source_overlap),

        "source_leakage_after_augmentation":
            len(
                augmented_validation_overlap
            ),

        "blank_training_labels":
            len(blank_labels),

        "missing_training_images":
            len(missing_images),

        "augmentation_types": [
            "brightness_darker",
            "brightness_lighter",
            "contrast_lower",
            "contrast_higher",
            "mild_blur",
            "mild_noise",
            "jpeg_compression",
            "affine_left",
            "affine_right",
            "horizontal_narrow",
            "horizontal_wide",
            "combined_mild_jitter",
        ]
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 72)
    print("AUGMENTATION V4 VALIDATION PASSED")
    print("=" * 72)

    print(
        "Original clean training samples:",
        len(train_records)
    )

    print(
        "Augmentations per sample:",
        AUGMENTATIONS_PER_SAMPLE
    )

    print(
        "New augmented samples:",
        len(augmented_records)
    )

    print(
        "Combined training samples:",
        len(combined_records)
    )

    print(
        "Untouched validation samples:",
        len(validation_records)
    )

    print()
    print(
        "Training sources:",
        len(train_sources)
    )

    print(
        "Validation sources:",
        len(validation_sources)
    )

    print(
        "Source leakage after augmentation:",
        len(
            augmented_validation_overlap
        )
    )

    print(
        "Blank labels:",
        len(blank_labels)
    )

    print(
        "Missing images:",
        len(missing_images)
    )

    print()
    print(
        "Augmented images:"
    )

    print(
        AUGMENT_IMAGES_DIR
    )

    print()
    print(
        "Augmented-only CSV:"
    )

    print(
        AUGMENTED_ONLY_CSV
    )

    print()
    print(
        "Combined training CSV:"
    )

    print(
        COMBINED_TRAIN_CSV
    )

    print()
    print(
        "Summary JSON:"
    )

    print(
        SUMMARY_JSON
    )

    print("=" * 72)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Validation data was NOT augmented."
    )

    print(
        "Do NOT start training until the generated "
        "augmentation images are visually checked."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print(
            "ERROR:"
        )

        print(
            error
        )