from pathlib import Path
import csv
import hashlib
import json

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

TRAIN_CSV = (
    OCR_ROOT
    / "real_training_expansion_v5"
    / "combined_v5"
    / "combined_real_train_v5.csv"
)

VALIDATION_CSV = (
    OCR_ROOT
    / "cleaned_real_ocr_v1"
    / "metadata"
    / "splits"
    / "clean_real_validation_v1.csv"
)

AUGMENT_ROOT = (
    OCR_ROOT
    / "real_training_expansion_v5"
    / "augmentation_v5"
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
    / "v5_augmented_only.csv"
)

COMBINED_TRAIN_CSV = (
    AUGMENT_METADATA_DIR
    / "v5_train_combined_augmented.csv"
)

SUMMARY_JSON = (
    AUGMENT_METADATA_DIR
    / "v5_augmentation_summary.json"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_ORIGINAL_TRAIN_COUNT = 121
EXPECTED_VALIDATION_COUNT = 23

AUGMENTATIONS_PER_SAMPLE = 12

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
# CSV
# ============================================================

def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        return list(
            csv.DictReader(file)
        )


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_image_path(record):

    image_path_text = str(
        record.get(
            "image_path",
            "",
        )
    ).strip()

    if not image_path_text:

        return None

    path = Path(
        image_path_text.replace(
            "\\",
            "/",
        )
    )

    if path.is_absolute():

        return path

    return (
        OCR_ROOT
        / path
    )


# ============================================================
# DETERMINISTIC RNG
# ============================================================

def create_rng(
    sample_id,
    variant_index,
):

    key = (
        f"{sample_id}|"
        f"{variant_index}|"
        "v5"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        key
    ).digest()

    seed = int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )

    return np.random.default_rng(
        seed
    )


# ============================================================
# IMAGE HELPERS
# ============================================================

def adjust_brightness_contrast(
    image,
    alpha,
    beta,
):

    return cv2.convertScaleAbs(
        image,
        alpha=float(alpha),
        beta=float(beta),
    )


def apply_blur(
    image,
    sigma,
):

    return cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    )


def add_gaussian_noise(
    image,
    sigma,
    rng,
):

    noise = rng.normal(
        0.0,
        float(sigma),
        image.shape,
    ).astype(
        np.float32
    )

    noisy = (
        image.astype(
            np.float32
        )
        + noise
    )

    noisy = np.clip(
        noisy,
        0,
        255,
    ).astype(
        np.uint8
    )

    return noisy


def apply_jpeg_compression(
    image,
    quality,
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
            ),
        ],
    )

    if not success:

        raise RuntimeError(
            "JPEG encoding failed."
        )

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    if decoded is None:

        raise RuntimeError(
            "JPEG decoding failed."
        )

    return decoded


def apply_affine(
    image,
    rotation,
    scale,
    tx_ratio,
    ty_ratio,
):

    height, width = image.shape[:2]

    center = (
        width / 2.0,
        height / 2.0,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        float(rotation),
        float(scale),
    )

    matrix[0, 2] += (
        float(tx_ratio)
        * width
    )

    matrix[1, 2] += (
        float(ty_ratio)
        * height
    )

    result = cv2.warpAffine(
        image,
        matrix,
        (
            width,
            height,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return result


def apply_horizontal_scale(
    image,
    scale_factor,
):

    height, width = image.shape[:2]

    new_width = max(
        1,
        int(
            round(
                width
                * float(
                    scale_factor
                )
            )
        ),
    )

    resized = cv2.resize(
        image,
        (
            new_width,
            height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    if new_width < width:

        extra = (
            width
            - new_width
        )

        left = (
            extra // 2
        )

        right = (
            extra
            - left
        )

        result = cv2.copyMakeBorder(
            resized,
            0,
            0,
            left,
            right,
            cv2.BORDER_REPLICATE,
        )

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
            start:start + width,
        ]

    else:

        result = resized

    return result


# ============================================================
# CREATE ONE AUGMENTED VARIANT
# ============================================================

def create_augmentation(
    image,
    sample_id,
    variant_index,
):

    rng = create_rng(
        sample_id,
        variant_index,
    )

    if variant_index == 1:

        result = adjust_brightness_contrast(
            image,
            alpha=0.92,
            beta=-7,
        )

        name = "brightness_darker"

    elif variant_index == 2:

        result = adjust_brightness_contrast(
            image,
            alpha=1.08,
            beta=7,
        )

        name = "brightness_lighter"

    elif variant_index == 3:

        result = adjust_brightness_contrast(
            image,
            alpha=0.88,
            beta=4,
        )

        name = "contrast_lower"

    elif variant_index == 4:

        result = adjust_brightness_contrast(
            image,
            alpha=1.12,
            beta=-3,
        )

        name = "contrast_higher"

    elif variant_index == 5:

        result = apply_blur(
            image,
            sigma=0.55,
        )

        name = "mild_blur"

    elif variant_index == 6:

        result = add_gaussian_noise(
            image,
            sigma=3.0,
            rng=rng,
        )

        name = "mild_noise"

    elif variant_index == 7:

        result = apply_jpeg_compression(
            image,
            quality=78,
        )

        name = "jpeg_compression"

    elif variant_index == 8:

        result = apply_affine(
            image,
            rotation=-0.55,
            scale=0.995,
            tx_ratio=-0.012,
            ty_ratio=0.025,
        )

        name = "affine_left"

    elif variant_index == 9:

        result = apply_affine(
            image,
            rotation=0.55,
            scale=1.005,
            tx_ratio=0.012,
            ty_ratio=-0.025,
        )

        name = "affine_right"

    elif variant_index == 10:

        result = apply_horizontal_scale(
            image,
            scale_factor=0.97,
        )

        name = "horizontal_narrow"

    elif variant_index == 11:

        result = apply_horizontal_scale(
            image,
            scale_factor=1.03,
        )

        name = "horizontal_wide"

    elif variant_index == 12:

        rotation = float(
            rng.uniform(
                -0.40,
                0.40,
            )
        )

        tx_ratio = float(
            rng.uniform(
                -0.008,
                0.008,
            )
        )

        ty_ratio = float(
            rng.uniform(
                -0.018,
                0.018,
            )
        )

        alpha = float(
            rng.uniform(
                0.96,
                1.04,
            )
        )

        beta = float(
            rng.uniform(
                -4.0,
                4.0,
            )
        )

        result = apply_affine(
            image,
            rotation=rotation,
            scale=1.0,
            tx_ratio=tx_ratio,
            ty_ratio=ty_ratio,
        )

        result = adjust_brightness_contrast(
            result,
            alpha=alpha,
            beta=beta,
        )

        result = add_gaussian_noise(
            result,
            sigma=2.0,
            rng=rng,
        )

        name = "combined_mild_jitter"

    else:

        raise ValueError(
            f"Unsupported augmentation index: "
            f"{variant_index}"
        )

    return (
        result,
        name,
    )


# ============================================================
# ORIGINAL RECORD
# ============================================================

def create_original_manifest_record(
    record
):

    return {
        "sample_id":
            str(
                record.get(
                    "sample_id",
                    "",
                )
            ).strip(),

        "image_path":
            str(
                record.get(
                    "image_path",
                    "",
                )
            ).strip(),

        "source_id":
            str(
                record.get(
                    "source_id",
                    "",
                )
            ).strip(),

        "language":
            str(
                record.get(
                    "language",
                    "si",
                )
            ).strip(),

        "ground_truth":
            str(
                record.get(
                    "ground_truth",
                    "",
                )
            ).strip(),

        "parent_sample_id":
            str(
                record.get(
                    "sample_id",
                    "",
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
    records,
):

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()

        writer.writerows(
            records
        )


# ============================================================
# MAIN
# ============================================================

def main():

    AUGMENT_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUGMENT_METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("REAL SINHALA OCR AUGMENTATION V5")
    print("=" * 72)

    train_records = load_csv(
        TRAIN_CSV
    )

    validation_records = load_csv(
        VALIDATION_CSV
    )

    print()
    print(
        "Original combined training samples:",
        len(
            train_records
        ),
    )

    print(
        "Untouched validation samples:",
        len(
            validation_records
        ),
    )

    if (
        len(train_records)
        != EXPECTED_ORIGINAL_TRAIN_COUNT
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_ORIGINAL_TRAIN_COUNT} "
            f"training samples but found "
            f"{len(train_records)}."
        )

    if (
        len(validation_records)
        != EXPECTED_VALIDATION_COUNT
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_VALIDATION_COUNT} "
            f"validation samples but found "
            f"{len(validation_records)}."
        )

    # ========================================================
    # SOURCE LEAKAGE CHECK
    # ========================================================

    train_sources = {
        str(
            row.get(
                "source_id",
                "",
            )
        ).strip()
        for row in train_records
    }

    validation_sources = {
        str(
            row.get(
                "source_id",
                "",
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
                source,
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

    duplicate_sample_ids = []

    seen_sample_ids = set()

    for record in train_records:

        sample_id = str(
            record.get(
                "sample_id",
                "",
            )
        ).strip()

        ground_truth = str(
            record.get(
                "ground_truth",
                "",
            )
        ).strip()

        image_path = (
            resolve_image_path(
                record
            )
        )

        if sample_id in seen_sample_ids:

            duplicate_sample_ids.append(
                sample_id
            )

        seen_sample_ids.add(
            sample_id
        )

        if not ground_truth:

            blank_labels.append(
                sample_id
            )

        if (
            image_path is None
            or
            not image_path.exists()
        ):

            missing_images.append(
                sample_id
            )

    if duplicate_sample_ids:

        raise RuntimeError(
            "Duplicate sample IDs:\n"
            + "\n".join(
                duplicate_sample_ids
            )
        )

    if blank_labels:

        raise RuntimeError(
            "Blank training labels:\n"
            + "\n".join(
                blank_labels
            )
        )

    if missing_images:

        raise RuntimeError(
            "Missing training images:\n"
            + "\n".join(
                missing_images
            )
        )

    # ========================================================
    # CLEAN PREVIOUS V5 GENERATED FILES ONLY
    # ========================================================

    previous_files = list(
        AUGMENT_IMAGES_DIR.glob(
            "*.png"
        )
    )

    if previous_files:

        print()
        print(
            "Removing previous V5 augmentation files:",
            len(
                previous_files
            ),
        )

        for path in previous_files:

            path.unlink()

    # ========================================================
    # GENERATE AUGMENTATIONS
    # ========================================================

    augmented_records = []

    combined_records = []

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
        start=1,
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
                "si",
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
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise RuntimeError(
                f"Could not read image:\n"
                f"{source_path}"
            )

        print()
        print(
            f"[{sample_number:03d}/"
            f"{len(train_records):03d}] "
            f"{sample_id}"
        )

        for variant_index in range(
            1,
            AUGMENTATIONS_PER_SAMPLE + 1,
        ):

            (
                augmented_image,
                augmentation_name,
            ) = create_augmentation(
                image,
                sample_id,
                variant_index,
            )

            augmented_sample_id = (
                f"{sample_id}"
                f"_aug_v5_"
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
                augmented_image,
            )

            if not success:

                raise RuntimeError(
                    f"Could not save:\n"
                    f"{output_path}"
                )

            relative_path = (
                Path(
                    "real_training_expansion_v5"
                )
                / "augmentation_v5"
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
            "variants",
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
            "Generated image count mismatch."
        )

    # ========================================================
    # LEAKAGE CHECK AFTER AUGMENTATION
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
    # SAVE OUTPUTS
    # ========================================================

    save_csv(
        AUGMENTED_ONLY_CSV,
        augmented_records
    )

    save_csv(
        COMBINED_TRAIN_CSV,
        combined_records
    )

    summary = {

        "version":
            "real_training_expansion_v5_augmentation",

        "original_training_samples":
            len(
                train_records
            ),

        "validation_samples_untouched":
            len(
                validation_records
            ),

        "augmentations_per_sample":
            AUGMENTATIONS_PER_SAMPLE,

        "augmented_samples":
            len(
                augmented_records
            ),

        "combined_training_samples":
            len(
                combined_records
            ),

        "training_source_count":
            len(
                train_sources
            ),

        "validation_source_count":
            len(
                validation_sources
            ),

        "source_leakage_before_augmentation":
            len(
                source_overlap
            ),

        "source_leakage_after_augmentation":
            len(
                augmented_validation_overlap
            ),

        "blank_training_labels":
            len(
                blank_labels
            ),

        "missing_training_images":
            len(
                missing_images
            ),

        "duplicate_training_sample_ids":
            len(
                duplicate_sample_ids
            ),

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
        ],
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 72)
    print("AUGMENTATION V5 VALIDATION PASSED")
    print("=" * 72)

    print(
        "Original training samples:",
        len(
            train_records
        ),
    )

    print(
        "Augmentations per sample:",
        AUGMENTATIONS_PER_SAMPLE,
    )

    print(
        "New augmented samples:",
        len(
            augmented_records
        ),
    )

    print(
        "Combined training samples:",
        len(
            combined_records
        ),
    )

    print(
        "Untouched validation samples:",
        len(
            validation_records
        ),
    )

    print()
    print(
        "Training sources:",
        len(
            train_sources
        ),
    )

    print(
        "Validation sources:",
        len(
            validation_sources
        ),
    )

    print(
        "Source leakage after augmentation:",
        len(
            augmented_validation_overlap
        ),
    )

    print(
        "Duplicate sample IDs:",
        len(
            duplicate_sample_ids
        ),
    )

    print(
        "Blank labels:",
        len(
            blank_labels
        ),
    )

    print(
        "Missing images:",
        len(
            missing_images
        ),
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
        "Fresh unseen images were NOT used."
    )

    print(
        "Do NOT start training until generated "
        "V5 augmentation images are visually checked."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print("ERROR:")
        print(error)