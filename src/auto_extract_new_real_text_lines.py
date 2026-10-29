from pathlib import Path
import csv

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

REGION_DIR = (
    OCR_ROOT
    / "real_training_regions"
)

OUTPUT_DIR = (
    OCR_ROOT
    / "real_training_auto_lines"
)

METADATA_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
)

OUTPUT_CSV = (
    METADATA_DIR
    / "auto_line_candidates_012_031.csv"
)


# ============================================================
# CONFIG
# ============================================================

START_SOURCE = 12
END_SOURCE = 31

MIN_LINE_HEIGHT_RATIO = 0.06
MAX_LINE_HEIGHT_RATIO = 0.45

MIN_LINE_WIDTH_RATIO = 0.20

VERTICAL_MARGIN_RATIO = 0.12
HORIZONTAL_MARGIN_RATIO = 0.03


# ============================================================
# HELPERS
# ============================================================

def source_id(number):

    return f"real_train_{number:03d}"


def get_region_path(number):

    return (
        REGION_DIR
        / f"{source_id(number)}_headline.png"
    )


# ============================================================
# CREATE TEXT ACTIVITY MASK
# ============================================================

def create_text_mask(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # CLAHE improves local contrast
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    # --------------------------------------------------------
    # Edge mask
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    edges = cv2.Canny(
        blurred,
        40,
        130
    )

    # --------------------------------------------------------
    # Gradient mask
    # --------------------------------------------------------

    gradient = cv2.morphologyEx(
        enhanced,
        cv2.MORPH_GRADIENT,
        np.ones(
            (3, 3),
            dtype=np.uint8
        )
    )

    _, gradient_mask = cv2.threshold(
        gradient,
        20,
        255,
        cv2.THRESH_BINARY
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    mask = cv2.bitwise_or(
        edges,
        gradient_mask
    )

    height, width = (
        mask.shape[:2]
    )

    # Horizontal connection of letters into words/lines
    kernel_width = max(
        5,
        int(width * 0.025)
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            kernel_width,
            3
        )
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        horizontal_kernel,
        iterations=1
    )

    # Remove tiny noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones(
            (2, 2),
            dtype=np.uint8
        ),
        iterations=1
    )

    return mask


# ============================================================
# HORIZONTAL PROJECTION
# ============================================================

def find_line_bands(mask):

    height, width = (
        mask.shape[:2]
    )

    binary = (
        mask > 0
    ).astype(
        np.float32
    )

    row_activity = np.mean(
        binary,
        axis=1
    )

    # Smooth vertical activity profile
    smooth_size = max(
        3,
        int(height * 0.025)
    )

    if smooth_size % 2 == 0:
        smooth_size += 1

    kernel = np.ones(
        smooth_size,
        dtype=np.float32
    ) / smooth_size

    smoothed = np.convolve(
        row_activity,
        kernel,
        mode="same"
    )

    # Dynamic threshold
    positive_values = smoothed[
        smoothed > 0
    ]

    if len(
        positive_values
    ) == 0:

        return []

    percentile_value = np.percentile(
        positive_values,
        35
    )

    threshold = max(
        0.025,
        percentile_value * 0.75
    )

    active_rows = (
        smoothed >= threshold
    )

    # --------------------------------------------------------
    # Convert active rows to bands
    # --------------------------------------------------------

    raw_bands = []

    start = None

    for y, active in enumerate(
        active_rows
    ):

        if active and start is None:

            start = y

        elif not active and start is not None:

            raw_bands.append(
                (
                    start,
                    y - 1
                )
            )

            start = None

    if start is not None:

        raw_bands.append(
            (
                start,
                height - 1
            )
        )

    # --------------------------------------------------------
    # Merge bands separated by very small gaps
    # --------------------------------------------------------

    merge_gap = max(
        2,
        int(height * 0.025)
    )

    merged = []

    for band in raw_bands:

        if not merged:

            merged.append(
                list(
                    band
                )
            )

            continue

        previous = merged[-1]

        gap = (
            band[0]
            - previous[1]
            - 1
        )

        if gap <= merge_gap:

            previous[1] = (
                band[1]
            )

        else:

            merged.append(
                list(
                    band
                )
            )

    return [
        (
            band[0],
            band[1]
        )
        for band in merged
    ]


# ============================================================
# GET TEXT EXTENT INSIDE A BAND
# ============================================================

def get_horizontal_extent(
    mask,
    y1,
    y2
):

    height, width = (
        mask.shape[:2]
    )

    region = mask[
        y1:y2 + 1,
        :
    ]

    column_activity = np.sum(
        region > 0,
        axis=0
    )

    active_columns = np.where(
        column_activity > 0
    )[0]

    if len(
        active_columns
    ) == 0:

        return None

    x1 = int(
        active_columns[0]
    )

    x2 = int(
        active_columns[-1]
    )

    return (
        x1,
        x2
    )


# ============================================================
# FILTER LINE BANDS
# ============================================================

def filter_line_bands(
    mask,
    bands
):

    height, width = (
        mask.shape[:2]
    )

    minimum_height = max(
        5,
        int(
            height
            * MIN_LINE_HEIGHT_RATIO
        )
    )

    maximum_height = max(
        minimum_height + 1,
        int(
            height
            * MAX_LINE_HEIGHT_RATIO
        )
    )

    minimum_width = int(
        width
        * MIN_LINE_WIDTH_RATIO
    )

    candidates = []

    for y1, y2 in bands:

        line_height = (
            y2 - y1 + 1
        )

        if line_height < minimum_height:

            continue

        if line_height > maximum_height:

            continue

        extent = (
            get_horizontal_extent(
                mask,
                y1,
                y2
            )
        )

        if extent is None:

            continue

        x1, x2 = extent

        line_width = (
            x2 - x1 + 1
        )

        if line_width < minimum_width:

            continue

        candidates.append(
            (
                x1,
                y1,
                x2,
                y2
            )
        )

    return candidates


# ============================================================
# MERGE STRONGLY OVERLAPPING BANDS
# ============================================================

def merge_overlapping_candidates(
    candidates
):

    if not candidates:

        return []

    candidates = sorted(
        candidates,
        key=lambda item:
        item[1]
    )

    merged = [
        list(
            candidates[0]
        )
    ]

    for candidate in candidates[1:]:

        x1, y1, x2, y2 = candidate

        previous = merged[-1]

        previous_height = (
            previous[3]
            - previous[1]
            + 1
        )

        current_height = (
            y2
            - y1
            + 1
        )

        overlap = max(
            0,
            min(
                previous[3],
                y2
            )
            - max(
                previous[1],
                y1
            )
            + 1
        )

        overlap_ratio = (
            overlap
            / min(
                previous_height,
                current_height
            )
        )

        if overlap_ratio >= 0.40:

            previous[0] = min(
                previous[0],
                x1
            )

            previous[1] = min(
                previous[1],
                y1
            )

            previous[2] = max(
                previous[2],
                x2
            )

            previous[3] = max(
                previous[3],
                y2
            )

        else:

            merged.append(
                [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            )

    return [
        tuple(
            item
        )
        for item in merged
    ]


# ============================================================
# EXPAND CROP
# ============================================================

def expand_box(
    box,
    image_shape
):

    x1, y1, x2, y2 = box

    height, width = (
        image_shape[:2]
    )

    line_height = (
        y2 - y1 + 1
    )

    vertical_margin = max(
        2,
        int(
            line_height
            * VERTICAL_MARGIN_RATIO
        )
    )

    horizontal_margin = max(
        3,
        int(
            width
            * HORIZONTAL_MARGIN_RATIO
        )
    )

    x1 = max(
        0,
        x1 - horizontal_margin
    )

    x2 = min(
        width - 1,
        x2 + horizontal_margin
    )

    y1 = max(
        0,
        y1 - vertical_margin
    )

    y2 = min(
        height - 1,
        y2 + vertical_margin
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# CREATE PREVIEW
# ============================================================

def create_preview(
    image,
    boxes
):

    preview = image.copy()

    for index, (
        x1,
        y1,
        x2,
        y2
    ) in enumerate(
        boxes,
        start=1
    ):

        cv2.rectangle(
            preview,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            (
                0,
                255,
                0
            ),
            2
        )

        cv2.putText(
            preview,
            str(index),
            (
                x1,
                max(
                    15,
                    y1 - 4
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

    return preview


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "AUTOMATIC REAL SINHALA HEADLINE LINE EXTRACTION"
    )

    print(
        "=" * 72
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Remove ONLY previous automatic outputs
    # --------------------------------------------------------

    removed = 0

    for path in OUTPUT_DIR.glob(
        "auto_*.png"
    ):

        path.unlink()

        removed += 1

    print(
        f"\nOld automatic files removed: "
        f"{removed}"
    )

    records = []

    total_lines = 0

    processed_sources = 0

    skipped_sources = 0

    # ========================================================
    # PROCESS 012–031
    # ========================================================

    for number in range(
        START_SOURCE,
        END_SOURCE + 1
    ):

        current_source = (
            source_id(
                number
            )
        )

        path = get_region_path(
            number
        )

        print(
            "\n" + "-" * 72
        )

        print(
            current_source
        )

        if not path.exists():

            print(
                "Headline region missing - skipped."
            )

            skipped_sources += 1

            continue

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR
        )

        if image is None:

            print(
                "Could not read image - skipped."
            )

            skipped_sources += 1

            continue

        # ----------------------------------------------------
        # Detect lines
        # ----------------------------------------------------

        mask = create_text_mask(
            image
        )

        bands = find_line_bands(
            mask
        )

        candidates = filter_line_bands(
            mask,
            bands
        )

        candidates = (
            merge_overlapping_candidates(
                candidates
            )
        )

        boxes = [
            expand_box(
                candidate,
                image.shape
            )
            for candidate in candidates
        ]

        # ----------------------------------------------------
        # Sort top to bottom
        # ----------------------------------------------------

        boxes = sorted(
            boxes,
            key=lambda box:
            box[1]
        )

        print(
            f"Detected lines: "
            f"{len(boxes)}"
        )

        # ----------------------------------------------------
        # Save mask for debugging
        # ----------------------------------------------------

        mask_path = (
            OUTPUT_DIR
            / f"auto_{current_source}_mask.png"
        )

        cv2.imwrite(
            str(mask_path),
            mask
        )

        # ----------------------------------------------------
        # Save preview
        # ----------------------------------------------------

        preview = create_preview(
            image,
            boxes
        )

        preview_path = (
            OUTPUT_DIR
            / f"auto_{current_source}_preview.png"
        )

        cv2.imwrite(
            str(preview_path),
            preview
        )

        # ----------------------------------------------------
        # Save individual crops
        # ----------------------------------------------------

        for line_number, box in enumerate(
            boxes,
            start=1
        ):

            x1, y1, x2, y2 = box

            crop = image[
                y1:y2 + 1,
                x1:x2 + 1
            ]

            if crop.size == 0:

                continue

            sample_id = (
                f"auto_{current_source}_"
                f"line_{line_number:02d}"
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

                continue

            records.append(
                {
                    "sample_id":
                        sample_id,

                    "image_path":
                        (
                            Path(
                                "real_training_auto_lines"
                            )
                            / filename
                        ).as_posix(),

                    "source_id":
                        current_source,

                    "line_number":
                        line_number,

                    "x1":
                        x1,

                    "y1":
                        y1,

                    "x2":
                        x2,

                    "y2":
                        y2,

                    "ground_truth":
                        "",

                    "keep":
                        ""
                }
            )

            total_lines += 1

        processed_sources += 1

    # ========================================================
    # SAVE CSV
    # ========================================================

    df = pd.DataFrame(
        records
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "AUTOMATIC LINE EXTRACTION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nProcessed sources : "
        f"{processed_sources}"
    )

    print(
        f"Skipped sources   : "
        f"{skipped_sources}"
    )

    print(
        f"Detected line crops: "
        f"{total_lines}"
    )

    print(
        "\nCandidate images:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nCandidate CSV:"
    )

    print(
        OUTPUT_CSV
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Existing clean real_training_lines "
        "were NOT changed."
    )