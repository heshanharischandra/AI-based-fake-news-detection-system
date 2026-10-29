from pathlib import Path
import csv

import cv2
import numpy as np


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
    / "source_images"
)

REAL_LINES_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
    / "real_lines"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
    / "metadata"
)

DEBUG_DIR = (
    METADATA_DIR
    / "debug_overlays"
)

REAL_LINES_DIR.mkdir(
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
# 2. CREATE YELLOW / WHITE TEXT MASKS
# ============================================================

def create_text_masks(image):

    height, width = image.shape[:2]

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Yellow Sinhala headline
    yellow_lower = np.array(
        [14, 70, 90],
        dtype=np.uint8
    )

    yellow_upper = np.array(
        [50, 255, 255],
        dtype=np.uint8
    )

    yellow_mask = cv2.inRange(
        hsv,
        yellow_lower,
        yellow_upper
    )

    # White Sinhala headline
    white_lower = np.array(
        [0, 0, 155],
        dtype=np.uint8
    )

    white_upper = np.array(
        [180, 110, 255],
        dtype=np.uint8
    )

    white_mask = cv2.inRange(
        hsv,
        white_lower,
        white_upper
    )

    # --------------------------------------------------------
    # Only headline section
    # --------------------------------------------------------

    roi_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    y1 = int(height * 0.50)
    y2 = int(height * 0.88)

    x1 = int(width * 0.03)
    x2 = int(width * 0.97)

    roi_mask[
        y1:y2,
        x1:x2
    ] = 255

    yellow_mask = cv2.bitwise_and(
        yellow_mask,
        roi_mask
    )

    white_mask = cv2.bitwise_and(
        white_mask,
        roi_mask
    )

    return yellow_mask, white_mask


# ============================================================
# 3. FIND INDIVIDUAL TEXT LINES
# ============================================================

def find_line_boxes(
    mask,
    color_name
):

    height, width = mask.shape[:2]

    # Horizontal dilation:
    # joins letters in the SAME line,
    # but keeps different lines separated.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (27, 3)
    )

    merged = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        merged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # Ignore tiny objects / logos / punctuation
        if w < width * 0.25:
            continue

        if h < 12:
            continue

        if h > height * 0.12:
            continue

        boxes.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "color": color_name
        })

    boxes.sort(
        key=lambda item: item["y"]
    )

    return boxes


# ============================================================
# 4. REMOVE DUPLICATE DETECTIONS
# ============================================================

def vertical_overlap_ratio(
    box1,
    box2
):

    top1 = box1["y"]
    bottom1 = box1["y"] + box1["h"]

    top2 = box2["y"]
    bottom2 = box2["y"] + box2["h"]

    overlap = max(
        0,
        min(bottom1, bottom2)
        - max(top1, top2)
    )

    smaller = min(
        box1["h"],
        box2["h"]
    )

    if smaller <= 0:
        return 0

    return overlap / smaller


def remove_duplicate_boxes(boxes):

    cleaned = []

    boxes.sort(
        key=lambda item: item["y"]
    )

    for box in boxes:

        duplicate = False

        for existing in cleaned:

            overlap = vertical_overlap_ratio(
                box,
                existing
            )

            if overlap > 0.70:
                duplicate = True
                break

        if not duplicate:
            cleaned.append(box)

    cleaned.sort(
        key=lambda item: item["y"]
    )

    return cleaned


# ============================================================
# 5. TIGHT CROP
# ============================================================

def crop_line(
    image,
    box
):

    height, width = image.shape[:2]

    x = box["x"]
    y = box["y"]
    w = box["w"]
    h = box["h"]

    # Horizontal space is useful
    pad_x = 10

    x1 = max(
        0,
        x - pad_x
    )

    x2 = min(
        width,
        x + w + pad_x
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Slightly crop INSIDE the detected vertical box.
    #
    # This removes fragments from the neighbouring line.
    # --------------------------------------------------------

    shrink_top = 2
    shrink_bottom = 2

    y1 = max(
        0,
        y + shrink_top
    )

    y2 = min(
        height,
        y + h - shrink_bottom
    )

    # Safety
    if y2 <= y1:

        y1 = y
        y2 = y + h

    crop = image[
        y1:y2,
        x1:x2
    ]

    return (
        crop,
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# 6. CLEAR OLD GENERATED CROPS
# ============================================================

def clear_old_generated_files():

    for file_path in REAL_LINES_DIR.glob(
        "real_*.png"
    ):

        file_path.unlink()

    print(
        "Old generated line crops cleared."
    )


# ============================================================
# 7. PROCESS ONE SOURCE IMAGE
# ============================================================

def process_image(
    image_path,
    sample_counter
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"Could not read: {image_path}"
        )

        return [], sample_counter

    yellow_mask, white_mask = (
        create_text_masks(
            image
        )
    )

    yellow_boxes = find_line_boxes(
        yellow_mask,
        "yellow"
    )

    white_boxes = find_line_boxes(
        white_mask,
        "white"
    )

    boxes = (
        yellow_boxes
        + white_boxes
    )

    boxes = remove_duplicate_boxes(
        boxes
    )

    boxes.sort(
        key=lambda item: item["y"]
    )

    print(
        f"\nProcessing: {image_path.name}"
    )

    print(
        f"Detected lines: {len(boxes)}"
    )

    debug_image = image.copy()

    rows = []

    source_id = image_path.stem

    for line_number, box in enumerate(
        boxes,
        start=1
    ):

        (
            crop,
            x1,
            y1,
            x2,
            y2
        ) = crop_line(
            image,
            box
        )

        if crop.size == 0:
            continue

        crop_height, crop_width = (
            crop.shape[:2]
        )

        if crop_width < 50:
            continue

        if crop_height < 15:
            continue

        sample_id = (
            f"real_{sample_counter:06d}"
        )

        crop_filename = (
            f"{sample_id}.png"
        )

        crop_path = (
            REAL_LINES_DIR
            / crop_filename
        )

        cv2.imwrite(
            str(crop_path),
            crop
        )

        rows.append({
            "sample_id":
                sample_id,

            "image_path":
                f"real_lines/{crop_filename}",

            "source_id":
                source_id,

            "line_number":
                line_number,

            "text_color":
                box["color"],

            "language":
                "si",

            "ground_truth":
                ""
        })

        # Debug rectangle
        cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            debug_image,
            str(line_number),
            (
                x1,
                max(20, y1 - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        print(
            f"Saved {crop_filename} "
            f"({crop_width}x{crop_height})"
        )

        sample_counter += 1

    debug_path = (
        DEBUG_DIR
        / (
            f"{source_id}_"
            f"detected_lines.png"
        )
    )

    cv2.imwrite(
        str(debug_path),
        debug_image
    )

    print(
        "Debug overlay saved:"
    )

    print(
        debug_path
    )

    return rows, sample_counter


# ============================================================
# 8. SAVE CSV
# ============================================================

def save_csv(rows):

    csv_path = (
        METADATA_DIR
        / "real_labels.csv"
    )

    fieldnames = [
        "sample_id",
        "image_path",
        "source_id",
        "line_number",
        "text_color",
        "language",
        "ground_truth"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    return csv_path


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 65
    )

    print(
        "SINHALA OCR DATASET PREPARATION"
    )

    print(
        "Individual Text-Line Crop Version"
    )

    print(
        "=" * 65
    )

    image_extensions = {
        ".png",
        ".jpg",
        ".jpeg"
    }

    source_images = [
        path
        for path in SOURCE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in image_extensions
        )
    ]

    source_images.sort()

    if not source_images:

        print(
            "\nNo source images found."
        )

        raise SystemExit

    clear_old_generated_files()

    all_rows = []

    sample_counter = 1

    for image_path in source_images:

        rows, sample_counter = (
            process_image(
                image_path,
                sample_counter
            )
        )

        all_rows.extend(
            rows
        )

    csv_path = save_csv(
        all_rows
    )

    print(
        "\n" + "=" * 65
    )

    print(
        "DATASET PREPARATION COMPLETE"
    )

    print(
        "=" * 65
    )

    print(
        f"\nSource images: "
        f"{len(source_images)}"
    )

    print(
        f"Extracted line images: "
        f"{len(all_rows)}"
    )

    print(
        "\nCSV saved:"
    )

    print(
        csv_path
    )

    print(
        "\nLine images saved:"
    )

    print(
        REAL_LINES_DIR
    )