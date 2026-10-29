from pathlib import Path
import csv

import cv2
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

REGION_DIR = (
    OCR_ROOT
    / "real_training_regions"
)

OUTPUT_DIR = (
    OCR_ROOT
    / "real_training_lines"
)

METADATA_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
)

DEBUG_DIR = (
    METADATA_DIR
    / "line_debug_overlays"
)

LABELS_CSV = (
    METADATA_DIR
    / "real_line_labels.csv"
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

MIN_LINE_HEIGHT = 8

MIN_LINE_WIDTH_RATIO = 0.12

PADDING_X = 8

PADDING_Y = 5


# ============================================================
# 3. SETUP
# ============================================================

def setup():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 4. REMOVE OLD GENERATED CROPS
# ============================================================

def clean_old_outputs():

    removed_lines = 0
    removed_debug = 0

    for path in OUTPUT_DIR.glob(
        "*.png"
    ):

        path.unlink()
        removed_lines += 1

    for path in DEBUG_DIR.glob(
        "*.png"
    ):

        path.unlink()
        removed_debug += 1

    if LABELS_CSV.exists():

        LABELS_CSV.unlink()

    print(
        f"Old line crops removed : "
        f"{removed_lines}"
    )

    print(
        f"Old debug overlays     : "
        f"{removed_debug}"
    )


# ============================================================
# 5. RESIZE FOR DETECTION
# ============================================================

def resize_for_detection(
    image
):

    height, width = (
        image.shape[:2]
    )

    # Small headline crops are enlarged for
    # better edge detection.
    if height >= 180:

        return image.copy(), 1.0

    scale = (
        180.0
        / height
    )

    new_width = int(
        width * scale
    )

    resized = cv2.resize(
        image,
        (
            new_width,
            180
        ),
        interpolation=cv2.INTER_CUBIC
    )

    return (
        resized,
        scale
    )


# ============================================================
# 6. FILL SMALL GAPS IN ROW ACTIVITY
# ============================================================

def close_row_gaps(
    active_rows,
    maximum_gap=4
):

    active_rows = (
        active_rows
        .astype(
            np.uint8
        )
    )

    kernel = np.ones(
        (
            maximum_gap + 1,
            1
        ),
        dtype=np.uint8
    )

    closed = cv2.morphologyEx(
        active_rows.reshape(
            -1,
            1
        ),
        cv2.MORPH_CLOSE,
        kernel
    )

    return (
        closed
        .reshape(-1)
        .astype(bool)
    )


# ============================================================
# 7. FIND CONTIGUOUS ACTIVE ROW BANDS
# ============================================================

def find_row_bands(
    active_rows
):

    bands = []

    start = None

    for row_index, active in enumerate(
        active_rows
    ):

        if active and start is None:

            start = row_index

        elif (
            not active
            and start is not None
        ):

            bands.append(
                (
                    start,
                    row_index - 1
                )
            )

            start = None

    if start is not None:

        bands.append(
            (
                start,
                len(active_rows) - 1
            )
        )

    return bands


# ============================================================
# 8. DETECT TEXT LINES
# ============================================================

def detect_text_lines(
    image
):

    detection_image, scale = (
        resize_for_detection(
            image
        )
    )

    gray = cv2.cvtColor(
        detection_image,
        cv2.COLOR_BGR2GRAY
    )

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(
            8,
            8
        )
    )

    enhanced = clahe.apply(
        gray
    )

    blurred = cv2.GaussianBlur(
        enhanced,
        (
            3,
            3
        ),
        0
    )

    # --------------------------------------------------------
    # Edge-based text detection
    # Works for white/yellow/red text on mixed backgrounds.
    # --------------------------------------------------------

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # Connect character strokes
    edge_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            3,
            3
        )
    )

    edges = cv2.dilate(
        edges,
        edge_kernel,
        iterations=1
    )

    image_height, image_width = (
        edges.shape
    )

    # Horizontal connection to group characters
    horizontal_kernel_width = max(
        9,
        int(
            image_width
            * 0.025
        )
    )

    horizontal_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                horizontal_kernel_width,
                3
            )
        )
    )

    connected = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        horizontal_kernel
    )

    # --------------------------------------------------------
    # Horizontal projection
    # --------------------------------------------------------

    row_counts = np.count_nonzero(
        connected,
        axis=1
    )

    row_threshold = max(
        3,
        int(
            image_width
            * 0.012
        )
    )

    active_rows = (
        row_counts
        >= row_threshold
    )

    active_rows = close_row_gaps(
        active_rows,
        maximum_gap=4
    )

    bands = find_row_bands(
        active_rows
    )

    detected_lines = []

    for top, bottom in bands:

        band_height = (
            bottom
            - top
            + 1
        )

        if band_height < MIN_LINE_HEIGHT:

            continue

        band_mask = connected[
            top:
            bottom + 1,
            :
        ]

        coordinates = cv2.findNonZero(
            band_mask
        )

        if coordinates is None:

            continue

        x, y, width, height = (
            cv2.boundingRect(
                coordinates
            )
        )

        # Ignore tiny fragments / logos
        if (
            width
            < image_width
            * MIN_LINE_WIDTH_RATIO
        ):

            continue

        detected_left = x

        detected_right = (
            x + width
        )

        detected_top = top

        detected_bottom = (
            bottom + 1
        )

        # ----------------------------------------------------
        # Convert coordinates back to original region size
        # ----------------------------------------------------

        left = int(
            detected_left
            / scale
        )

        right = int(
            detected_right
            / scale
        )

        original_top = int(
            detected_top
            / scale
        )

        original_bottom = int(
            detected_bottom
            / scale
        )

        detected_lines.append({

            "left":
                left,

            "top":
                original_top,

            "right":
                right,

            "bottom":
                original_bottom
        })

    # Top-to-bottom order
    detected_lines.sort(
        key=lambda line:
            line[
                "top"
            ]
    )

    return detected_lines


# ============================================================
# 9. CROP LINE FROM ORIGINAL REGION
# ============================================================

def crop_text_line(
    image,
    line
):

    image_height, image_width = (
        image.shape[:2]
    )

    left = max(
        0,
        line["left"]
        - PADDING_X
    )

    top = max(
        0,
        line["top"]
        - PADDING_Y
    )

    right = min(
        image_width,
        line["right"]
        + PADDING_X
    )

    bottom = min(
        image_height,
        line["bottom"]
        + PADDING_Y
    )

    crop = image[
        top:bottom,
        left:right
    ]

    return (
        crop,
        left,
        top,
        right,
        bottom
    )


# ============================================================
# 10. PROCESS ONE HEADLINE REGION
# ============================================================

def process_region(
    region_path,
    global_counter
):

    image = cv2.imread(
        str(region_path),
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            f"Could not read:\n"
            f"{region_path}"
        )

    source_id = (
        region_path.stem
        .replace(
            "_headline",
            ""
        )
    )

    lines = detect_text_lines(
        image
    )

    overlay = image.copy()

    rows = []

    source_line_number = 0

    for line in lines:

        (
            crop,
            left,
            top,
            right,
            bottom
        ) = crop_text_line(
            image,
            line
        )

        if crop.size == 0:

            continue

        crop_height, crop_width = (
            crop.shape[:2]
        )

        if (
            crop_height < 8
            or crop_width < 25
        ):

            continue

        global_counter += 1

        source_line_number += 1

        sample_id = (
            f"real_train_line_"
            f"{global_counter:06d}"
        )

        filename = (
            f"{sample_id}.png"
        )

        output_path = (
            OUTPUT_DIR
            / filename
        )

        success = cv2.imwrite(
            str(output_path),
            crop
        )

        if not success:

            raise RuntimeError(
                f"Could not save:\n"
                f"{output_path}"
            )

        rows.append({

            "sample_id":
                sample_id,

            "image_path":
                (
                    Path(
                        "real_training_lines"
                    )
                    / filename
                ).as_posix(),

            "source_id":
                source_id,

            "region_image":
                region_path.name,

            "line_number":
                source_line_number,

            "language":
                "si",

            # Fill manually after visual review
            "ground_truth":
                "",

            "keep":
                "",

            "left":
                left,

            "top":
                top,

            "right":
                right,

            "bottom":
                bottom,

            "crop_width":
                crop_width,

            "crop_height":
                crop_height
        })

        # Debug rectangle
        cv2.rectangle(
            overlay,
            (
                left,
                top
            ),
            (
                right,
                bottom
            ),
            (
                0,
                255,
                0
            ),
            2
        )

        cv2.putText(
            overlay,
            str(
                source_line_number
            ),
            (
                left,
                max(
                    15,
                    top - 3
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (
                0,
                255,
                0
            ),
            1,
            cv2.LINE_AA
        )

    debug_path = (
        DEBUG_DIR
        / f"{source_id}_lines.png"
    )

    cv2.imwrite(
        str(debug_path),
        overlay
    )

    return (
        rows,
        global_counter
    )


# ============================================================
# 11. SAVE CSV
# ============================================================

def save_labels_csv(
    rows
):

    columns = [
        "sample_id",
        "image_path",
        "source_id",
        "region_image",
        "line_number",
        "language",
        "ground_truth",
        "keep",
        "left",
        "top",
        "right",
        "bottom",
        "crop_width",
        "crop_height"
    ]

    with open(
        LABELS_CSV,
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
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "REAL HEADLINE -> TEXT LINE EXTRACTION"
    )

    print(
        "=" * 72
    )

    setup()

    region_images = sorted([
        path
        for path in REGION_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ])

    print(
        f"\nHeadline regions found: "
        f"{len(region_images)}"
    )

    if not region_images:

        raise RuntimeError(
            "No headline regions found."
        )

    clean_old_outputs()

    all_rows = []

    global_counter = 0

    for index, region_path in enumerate(
        region_images,
        start=1
    ):

        print(
            f"\nProcessing "
            f"{index}/"
            f"{len(region_images)}: "
            f"{region_path.name}"
        )

        try:

            rows, global_counter = (
                process_region(
                    region_path,
                    global_counter
                )
            )

            all_rows.extend(
                rows
            )

            print(
                f"Detected text lines: "
                f"{len(rows)}"
            )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

    save_labels_csv(
        all_rows
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "LINE EXTRACTION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nHeadline regions : "
        f"{len(region_images)}"
    )

    print(
        f"Text line crops  : "
        f"{len(all_rows)}"
    )

    print(
        "\nLine crops:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nLabels CSV:"
    )

    print(
        LABELS_CSV
    )

    print(
        "\nDebug overlays:"
    )

    print(
        DEBUG_DIR
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do not label yet."
    )

    print(
        "First visually review the new line crops."
    )