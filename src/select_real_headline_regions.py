from pathlib import Path
import csv

import cv2


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

SOURCE_DIR = (
    OCR_ROOT
    / "real_training_sources"
)

REGION_DIR = (
    OCR_ROOT
    / "real_training_regions"
)

METADATA_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
)

REGION_CSV = (
    METADATA_DIR
    / "headline_regions.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp"
}

MAX_DISPLAY_WIDTH = 1000
MAX_DISPLAY_HEIGHT = 800


# ============================================================
# 3. SETUP
# ============================================================

def setup():

    REGION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 4. DISPLAY RESIZE
# ============================================================

def resize_for_display(image):

    height, width = image.shape[:2]

    width_scale = (
        MAX_DISPLAY_WIDTH
        / width
    )

    height_scale = (
        MAX_DISPLAY_HEIGHT
        / height
    )

    scale = min(
        1.0,
        width_scale,
        height_scale
    )

    if scale == 1.0:

        return image.copy(), 1.0

    display_width = int(
        width * scale
    )

    display_height = int(
        height * scale
    )

    resized = cv2.resize(
        image,
        (
            display_width,
            display_height
        ),
        interpolation=cv2.INTER_AREA
    )

    return (
        resized,
        scale
    )


# ============================================================
# 5. SELECT ONE REGION
# ============================================================

def select_region(
    image_path
):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    display_image, scale = (
        resize_for_display(
            image
        )
    )

    window_name = (
        f"Select HEADLINE only - "
        f"{image_path.name}"
    )

    print(
        "\nSelect ONLY the Sinhala headline/text area."
    )

    print(
        "Drag with mouse -> press ENTER or SPACE."
    )

    print(
        "Press C to cancel/skip this image."
    )

    roi = cv2.selectROI(
        window_name,
        display_image,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(
        window_name
    )

    x, y, width, height = [
        int(value)
        for value in roi
    ]

    # Cancel / empty selection
    if width <= 0 or height <= 0:

        return None

    # --------------------------------------------------------
    # Convert display coordinates back to original
    # --------------------------------------------------------

    original_x = int(
        x / scale
    )

    original_y = int(
        y / scale
    )

    original_width = int(
        width / scale
    )

    original_height = int(
        height / scale
    )

    image_height, image_width = (
        image.shape[:2]
    )

    left = max(
        0,
        original_x
    )

    top = max(
        0,
        original_y
    )

    right = min(
        image_width,
        left + original_width
    )

    bottom = min(
        image_height,
        top + original_height
    )

    crop = image[
        top:bottom,
        left:right
    ]

    if crop.size == 0:

        return None

    return {
        "crop": crop,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width": right - left,
        "height": bottom - top
    }


# ============================================================
# 6. LOAD EXISTING METADATA
# ============================================================

def load_existing_rows():

    if not REGION_CSV.exists():

        return []

    rows = []

    with open(
        REGION_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            rows.append(
                row
            )

    return rows


# ============================================================
# 7. SAVE METADATA
# ============================================================

def save_metadata(rows):

    columns = [
        "source_id",
        "source_image",
        "region_image",
        "left",
        "top",
        "right",
        "bottom",
        "width",
        "height"
    ]

    with open(
        REGION_CSV,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "REAL SINHALA NEWS HEADLINE REGION SELECTOR"
    )

    print(
        "=" * 72
    )

    setup()

    source_images = sorted([
        path
        for path in SOURCE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ])

    print(
        f"\nSource images found: "
        f"{len(source_images)}"
    )

    if not source_images:

        raise RuntimeError(
            "No source images found."
        )

    existing_rows = (
        load_existing_rows()
    )

    existing_source_ids = {
        row["source_id"]
        for row in existing_rows
    }

    rows = list(
        existing_rows
    )

    selected_count = 0
    skipped_count = 0
    already_done_count = 0

    for index, image_path in enumerate(
        source_images,
        start=1
    ):

        source_id = (
            image_path.stem
        )

        print(
            "\n" + "-" * 72
        )

        print(
            f"Image {index}/"
            f"{len(source_images)}"
        )

        print(
            image_path.name
        )

        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        if source_id in existing_source_ids:

            print(
                "Already selected - skipping."
            )

            already_done_count += 1

            continue

        result = select_region(
            image_path
        )

        if result is None:

            print(
                "Skipped."
            )

            skipped_count += 1

            continue

        region_name = (
            f"{source_id}_headline.png"
        )

        region_path = (
            REGION_DIR
            / region_name
        )

        success = cv2.imwrite(
            str(region_path),
            result["crop"]
        )

        if not success:

            raise RuntimeError(
                f"Could not save region:\n"
                f"{region_path}"
            )

        rows.append({

            "source_id":
                source_id,

            "source_image":
                image_path.name,

            "region_image":
                (
                    Path(
                        "real_training_regions"
                    )
                    / region_name
                ).as_posix(),

            "left":
                result["left"],

            "top":
                result["top"],

            "right":
                result["right"],

            "bottom":
                result["bottom"],

            "width":
                result["width"],

            "height":
                result["height"]
        })

        # Save after EACH image
        # so progress is not lost.
        save_metadata(
            rows
        )

        selected_count += 1

        print(
            f"Saved: {region_name}"
        )

    cv2.destroyAllWindows()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "HEADLINE REGION SELECTION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nNew regions selected : "
        f"{selected_count}"
    )

    print(
        f"Already completed    : "
        f"{already_done_count}"
    )

    print(
        f"Skipped              : "
        f"{skipped_count}"
    )

    print(
        f"Total metadata rows  : "
        f"{len(rows)}"
    )

    print(
        "\nRegion images:"
    )

    print(
        REGION_DIR
    )

    print(
        "\nMetadata CSV:"
    )

    print(
        REGION_CSV
    )