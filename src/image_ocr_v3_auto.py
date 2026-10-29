from pathlib import Path
from datetime import datetime
import json

import cv2
import numpy as np

import auto_extract_new_real_text_lines as line_extractor

from image_ocr import (
    create_text_masks,
    find_line_boxes,
)

from ocr_v3_inference import get_ocr_v2_engine


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v3_auto"
)

DEBUG_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v3_auto_debug"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEBUG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            f"Could not read image:\n{image_path}"
        )

    return image


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):

    sinhala_count = sum(
        1
        for character in text
        if "\u0D80" <= character <= "\u0DFF"
    )

    english_count = sum(
        1
        for character in text
        if (
            "A" <= character <= "Z"
            or
            "a" <= character <= "z"
        )
    )

    if sinhala_count > 0 and english_count > 0:
        return "si+en"

    if sinhala_count > 0:
        return "si"

    if english_count > 0:
        return "en"

    return "unknown"


# ============================================================
# REMOVE DUPLICATE BOXES
# ============================================================

def remove_duplicate_boxes(boxes):

    if not boxes:
        return []

    boxes = sorted(
        boxes,
        key=lambda box: (
            box["y"],
            box["x"]
        )
    )

    cleaned = []

    for box in boxes:

        center_y = (
            box["y"]
            + box["h"] / 2.0
        )

        duplicate_index = None

        for index, existing in enumerate(cleaned):

            existing_center_y = (
                existing["y"]
                + existing["h"] / 2.0
            )

            if abs(
                center_y
                - existing_center_y
            ) <= 15:

                duplicate_index = index
                break

        if duplicate_index is None:

            cleaned.append(
                box
            )

        else:

            old_box = cleaned[
                duplicate_index
            ]

            old_area = (
                old_box["w"]
                * old_box["h"]
            )

            new_area = (
                box["w"]
                * box["h"]
            )

            if new_area > old_area:

                cleaned[
                    duplicate_index
                ] = box

    cleaned.sort(
        key=lambda box:
            box["y"]
    )

    return cleaned


# ============================================================
# SELECT DENSE HEADLINE CLUSTER
# ============================================================

def select_headline_cluster(boxes):

    if not boxes:
        return []

    boxes = sorted(
        boxes,
        key=lambda box:
            box["y"]
    )

    heights = np.array(
        [
            box["h"]
            for box in boxes
        ],
        dtype=np.float32
    )

    median_height = float(
        np.median(
            heights
        )
    )

    # --------------------------------------------------------
    # Remove tiny objects / captions / thin decorations.
    # --------------------------------------------------------

    minimum_height = max(
        18,
        int(
            round(
                median_height * 0.55
            )
        )
    )

    filtered = [
        box
        for box in boxes
        if box["h"] >= minimum_height
    ]

    if not filtered:

        filtered = boxes.copy()

    filtered.sort(
        key=lambda box:
            box["y"]
    )

    filtered_heights = np.array(
        [
            box["h"]
            for box in filtered
        ],
        dtype=np.float32
    )

    filtered_median_height = float(
        np.median(
            filtered_heights
        )
    )

    # Headline lines should be vertically close.
    gap_limit = max(
        28,
        int(
            round(
                filtered_median_height
                * 0.60
            )
        )
    )

    clusters = []

    current_cluster = [
        filtered[0]
    ]

    for box in filtered[1:]:

        previous = current_cluster[-1]

        previous_bottom = (
            previous["y"]
            + previous["h"]
        )

        vertical_gap = (
            box["y"]
            - previous_bottom
        )

        if vertical_gap <= gap_limit:

            current_cluster.append(
                box
            )

        else:

            clusters.append(
                current_cluster
            )

            current_cluster = [
                box
            ]

    clusters.append(
        current_cluster
    )

    # --------------------------------------------------------
    # Prefer:
    # 1. more lines
    # 2. more total text width
    # 3. lower region if still tied
    # --------------------------------------------------------

    def cluster_score(cluster):

        line_count = len(
            cluster
        )

        total_width = sum(
            box["w"]
            for box in cluster
        )

        average_y = sum(
            box["y"]
            for box in cluster
        ) / line_count

        return (
            line_count,
            total_width,
            average_y
        )

    best_cluster = max(
        clusters,
        key=cluster_score
    )

    best_cluster = sorted(
        best_cluster,
        key=lambda box:
            box["y"]
    )

    return best_cluster


# ============================================================
# AUTOMATIC COLOUR HEADLINE DETECTION
# ============================================================

def detect_headline_lines(image):

    yellow_mask, white_mask = (
        create_text_masks(
            image
        )
    )

    yellow_boxes = (
        find_line_boxes(
            yellow_mask,
            "yellow"
        )
    )

    white_boxes = (
        find_line_boxes(
            white_mask,
            "white"
        )
    )

    all_boxes = (
        yellow_boxes
        + white_boxes
    )

    all_boxes = (
        remove_duplicate_boxes(
            all_boxes
        )
    )

    selected_boxes = (
        select_headline_cluster(
            all_boxes
        )
    )

    return (
        yellow_mask,
        white_mask,
        all_boxes,
        selected_boxes
    )


# ============================================================
# BUILD TRAINING-ALIGNED ROI
# ============================================================

def build_training_aligned_roi(
    image,
    selected_boxes,
    all_boxes
):

    image_height, image_width = (
        image.shape[:2]
    )

    if not selected_boxes:

        raise RuntimeError(
            "No headline cluster detected."
        )

    widths = np.array(
        [
            box["w"]
            for box in selected_boxes
        ],
        dtype=np.float32
    )

    heights = np.array(
        [
            box["h"]
            for box in selected_boxes
        ],
        dtype=np.float32
    )

    centers_x = np.array(
        [
            box["x"]
            + box["w"] / 2.0
            for box in selected_boxes
        ],
        dtype=np.float32
    )

    median_width = float(
        np.median(
            widths
        )
    )

    max_width = float(
        np.max(
            widths
        )
    )

    median_height = float(
        np.median(
            heights
        )
    )

    # --------------------------------------------------------
    # Horizontal center
    # --------------------------------------------------------

    if len(selected_boxes) <= 2:

        widest_index = int(
            np.argmax(
                widths
            )
        )

        estimated_center_x = float(
            centers_x[
                widest_index
            ]
        )

    else:

        estimated_center_x = float(
            np.average(
                centers_x,
                weights=widths
            )
        )

    target_width = int(
        round(
            min(
                max_width,
                median_width * 1.25
            )
        )
    )

    target_width = max(
        target_width,
        int(
            image_width * 0.55
        )
    )

    target_width = min(
        target_width,
        image_width
    )

    x1 = int(
        round(
            estimated_center_x
            - target_width / 2.0
        )
    )

    x2 = (
        x1
        + target_width
    )

    if x1 < 0:

        x2 -= x1
        x1 = 0

    if x2 > image_width:

        extra = (
            x2
            - image_width
        )

        x1 = max(
            0,
            x1 - extra
        )

        x2 = image_width

    # --------------------------------------------------------
    # Vertical region
    # --------------------------------------------------------

    min_y = min(
        box["y"]
        for box in selected_boxes
    )

    max_y = max(
        box["y"]
        + box["h"]
        for box in selected_boxes
    )

    # Two-line headline:
    # tighter ROI avoids nearby captions.
    if len(selected_boxes) <= 2:

        top_margin = int(
            round(
                median_height * 0.25
            )
        )

        bottom_margin = int(
            round(
                median_height * 0.05
            )
        )

    # Three or more lines:
    # retain more glyph/context margin.
    else:

        top_margin = int(
            round(
                median_height * 0.62
            )
        )

        bottom_margin = int(
            round(
                median_height * 0.67
            )
        )

    y1 = max(
        0,
        min_y - top_margin
    )

    y2 = min(
        image_height,
        max_y + bottom_margin
    )

    # --------------------------------------------------------
    # Prevent nearby rejected regions from entering ROI.
    # --------------------------------------------------------

    rejected_boxes = [
        box
        for box in all_boxes
        if box not in selected_boxes
    ]

    previous_boxes = [
        box
        for box in rejected_boxes
        if (
            box["y"]
            + box["h"]
        ) <= min_y
    ]

    if previous_boxes:

        previous_bottom = max(
            box["y"]
            + box["h"]
            for box in previous_boxes
        )

        y1 = max(
            y1,
            previous_bottom + 5
        )

    next_boxes = [
        box
        for box in rejected_boxes
        if box["y"] >= max_y
    ]

    if next_boxes:

        next_top = min(
            box["y"]
            for box in next_boxes
        )

        y2 = min(
            y2,
            next_top - 5
        )

    if y2 <= y1:

        raise RuntimeError(
            "Automatic headline ROI height is invalid."
        )

    roi = image[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:

        raise RuntimeError(
            "Automatic headline ROI is empty."
        )

    info = {

        "x1":
            int(x1),

        "y1":
            int(y1),

        "x2":
            int(x2),

        "y2":
            int(y2),

        "width":
            int(
                x2 - x1
            ),

        "height":
            int(
                y2 - y1
            ),

        "selected_colour_lines":
            len(
                selected_boxes
            ),

        "estimated_center_x":
            float(
                estimated_center_x
            ),

        "median_line_width":
            float(
                median_width
            ),

        "median_line_height":
            float(
                median_height
            )
    }

    return (
        roi,
        info
    )


# ============================================================
# TRAINING-STYLE LINE EXTRACTION
# ============================================================

def extract_training_style_lines(roi):

    mask = (
        line_extractor
        .create_text_mask(
            roi
        )
    )

    bands = (
        line_extractor
        .find_line_bands(
            mask
        )
    )

    candidates = (
        line_extractor
        .filter_line_bands(
            mask,
            bands
        )
    )

    candidates = (
        line_extractor
        .merge_overlapping_candidates(
            candidates
        )
    )

    boxes = [
        line_extractor.expand_box(
            candidate,
            roi.shape
        )
        for candidate in candidates
    ]

    boxes = sorted(
        boxes,
        key=lambda box:
            box[1]
    )

    crops = []

    for number, box in enumerate(
        boxes,
        start=1
    ):

        x1, y1, x2, y2 = box

        crop = roi[
            y1:y2 + 1,
            x1:x2 + 1
        ]

        if crop.size == 0:
            continue

        crops.append(
            {
                "line_number":
                    number,

                "crop":
                    crop,

                "box":
                    {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }
            }
        )

    return (
        mask,
        boxes,
        crops
    )


# ============================================================
# DEBUG PREVIEW
# ============================================================

def create_detection_preview(
    image,
    all_boxes,
    selected_boxes
):

    preview = image.copy()

    for box in all_boxes:

        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]

        cv2.rectangle(
            preview,
            (x, y),
            (
                x + w,
                y + h
            ),
            (0, 0, 255),
            2
        )

    for number, box in enumerate(
        selected_boxes,
        start=1
    ):

        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]

        cv2.rectangle(
            preview,
            (x, y),
            (
                x + w,
                y + h
            ),
            (0, 255, 0),
            3
        )

        cv2.putText(
            preview,
            f"H{number}",
            (
                x,
                max(
                    20,
                    y - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return preview


def create_line_preview(
    roi,
    boxes
):

    preview = roi.copy()

    for number, box in enumerate(
        boxes,
        start=1
    ):

        x1, y1, x2, y2 = box

        cv2.rectangle(
            preview,
            (
                int(x1),
                int(y1)
            ),
            (
                int(x2),
                int(y2)
            ),
            (0, 255, 0),
            2
        )

        cv2.putText(
            preview,
            str(number),
            (
                int(x1),
                max(
                    20,
                    int(y1) - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return preview


# ============================================================
# COMPLETE OCR
# ============================================================

def run_ocr(image_path):

    image = load_image(
        image_path
    )

    engine = (
        get_ocr_v2_engine()
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    debug_dir = (
        DEBUG_DIR
        / f"run_{timestamp}"
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 72)
    print("AUTOMATIC HEADLINE IMAGE -> TEXT OCR V3")
    print("=" * 72)

    print(
        "\nModel:"
    )

    print(
        engine.model_path
    )

    print(
        f"\nCheckpoint epoch: "
        f"{engine.checkpoint_epoch}"
    )

    if engine.validation_cer is not None:

        print(
            f"Validation CER: "
            f"{engine.validation_cer:.4f}"
        )

    if engine.validation_wer is not None:

        print(
            f"Validation WER: "
            f"{engine.validation_wer:.4f}"
        )

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print(
        "STEP 1: Detecting colour text candidates..."
    )

    (
        yellow_mask,
        white_mask,
        all_boxes,
        selected_boxes
    ) = detect_headline_lines(
        image
    )

    print(
        f"All colour candidates: "
        f"{len(all_boxes)}"
    )

    for number, box in enumerate(
        all_boxes,
        start=1
    ):

        print(
            f"  Candidate {number}: "
            f"{box}"
        )

    print()
    print(
        f"Selected headline lines: "
        f"{len(selected_boxes)}"
    )

    for number, box in enumerate(
        selected_boxes,
        start=1
    ):

        print(
            f"  Headline {number}: "
            f"{box}"
        )

    if not selected_boxes:

        raise RuntimeError(
            "No valid headline cluster detected."
        )

    # ========================================================
    # DEBUG FILES
    # ========================================================

    yellow_mask_path = (
        debug_dir
        / "yellow_mask.png"
    )

    white_mask_path = (
        debug_dir
        / "white_mask.png"
    )

    detection_preview_path = (
        debug_dir
        / "headline_cluster_preview.png"
    )

    cv2.imwrite(
        str(
            yellow_mask_path
        ),
        yellow_mask
    )

    cv2.imwrite(
        str(
            white_mask_path
        ),
        white_mask
    )

    detection_preview = (
        create_detection_preview(
            image,
            all_boxes,
            selected_boxes
        )
    )

    cv2.imwrite(
        str(
            detection_preview_path
        ),
        detection_preview
    )

    # ========================================================
    # STEP 2
    # ========================================================

    print()
    print(
        "STEP 2: Building headline ROI..."
    )

    (
        headline_roi,
        roi_info
    ) = build_training_aligned_roi(
        image,
        selected_boxes,
        all_boxes
    )

    print(
        "Automatic headline ROI:"
    )

    print(
        roi_info
    )

    print(
        "ROI shape:",
        headline_roi.shape
    )

    headline_roi_path = (
        debug_dir
        / "automatic_headline_roi.png"
    )

    cv2.imwrite(
        str(
            headline_roi_path
        ),
        headline_roi
    )

    # ========================================================
    # STEP 3
    # ========================================================

    print()
    print(
        "STEP 3: Extracting training-style lines..."
    )

    (
        training_mask,
        training_boxes,
        line_crops
    ) = extract_training_style_lines(
        headline_roi
    )

    print(
        f"Detected OCR lines: "
        f"{len(line_crops)}"
    )

    training_mask_path = (
        debug_dir
        / "training_line_mask.png"
    )

    training_preview_path = (
        debug_dir
        / "training_line_preview.png"
    )

    cv2.imwrite(
        str(
            training_mask_path
        ),
        training_mask
    )

    training_preview = (
        create_line_preview(
            headline_roi,
            training_boxes
        )
    )

    cv2.imwrite(
        str(
            training_preview_path
        ),
        training_preview
    )

    if not line_crops:

        raise RuntimeError(
            "No OCR line crops were produced."
        )

    # ========================================================
    # STEP 4 — OCR
    # ========================================================

    print()
    print(
        "STEP 4: Running V3 OCR..."
    )

    results = []

    for item in line_crops:

        number = (
            item[
                "line_number"
            ]
        )

        crop = (
            item[
                "crop"
            ]
        )

        crop_path = (
            debug_dir
            / f"line_{number:02d}.png"
        )

        cv2.imwrite(
            str(
                crop_path
            ),
            crop
        )

        prediction = (
            engine.predict_line(
                crop_path
            )
        )

        text = (
            prediction[
                "text"
            ]
            .strip()
        )

        print()
        print("-" * 72)

        print(
            f"LINE {number}"
        )

        print("-" * 72)

        print(
            "Crop shape:",
            crop.shape
        )

        print(
            "Prediction:"
        )

        print(
            text
        )

        results.append(
            {
                "line_number":
                    number,

                "text":
                    text,

                "crop_shape":
                    list(
                        crop.shape
                    ),

                "crop_path":
                    str(
                        crop_path
                    ),

                "box":
                    item[
                        "box"
                    ]
            }
        )

    # ========================================================
    # COMBINE TEXT
    # ========================================================

    extracted_text = "\n".join(
        result[
            "text"
        ]
        for result in results
        if result[
            "text"
        ]
    )

    language = (
        detect_language(
            extracted_text
        )
    )

    # ========================================================
    # SAVE TXT
    # ========================================================

    txt_path = (
        RESULTS_DIR
        / f"extracted_text_{timestamp}.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            extracted_text
        )

    # ========================================================
    # SAVE JSON
    # ========================================================

    json_path = (
        RESULTS_DIR
        / f"ocr_result_{timestamp}.json"
    )

    json_data = {

        "timestamp":
            datetime.now().isoformat(),

        "input_type":
            "image",

        "image_path":
            str(
                Path(
                    image_path
                ).resolve()
            ),

        "ocr_engine":
            "CRNN-CTC Crop-Jitter V3",

        "checkpoint_epoch":
            engine.checkpoint_epoch,

        "model_validation_cer":
            engine.validation_cer,

        "model_validation_wer":
            engine.validation_wer,

        "detection_method":
            (
                "colour_detection_"
                "dense_headline_cluster_"
                "training_style_segmentation"
            ),

        "all_colour_candidate_count":
            len(
                all_boxes
            ),

        "selected_headline_line_count":
            len(
                selected_boxes
            ),

        "ocr_line_count":
            len(
                results
            ),

        "headline_roi":
            roi_info,

        "detected_language":
            language,

        "extracted_text":
            extracted_text,

        "text_file":
            str(
                txt_path
            ),

        "lines":
            results,

        "debug_files":
            {
                "yellow_mask":
                    str(
                        yellow_mask_path
                    ),

                "white_mask":
                    str(
                        white_mask_path
                    ),

                "headline_cluster_preview":
                    str(
                        detection_preview_path
                    ),

                "automatic_headline_roi":
                    str(
                        headline_roi_path
                    ),

                "training_line_mask":
                    str(
                        training_mask_path
                    ),

                "training_line_preview":
                    str(
                        training_preview_path
                    )
            }
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 72)
    print("FINAL EXTRACTED TEXT")
    print("=" * 72)

    print()

    if extracted_text:

        print(
            extracted_text
        )

    else:

        print(
            "No text extracted."
        )

    print(
        "\nDetected Language:",
        language
    )

    print(
        "OCR Lines:",
        len(
            results
        )
    )

    print(
        "\nTXT saved:"
    )

    print(
        txt_path
    )

    print(
        "\nJSON saved:"
    )

    print(
        json_path
    )

    print(
        "\nDebug folder:"
    )

    print(
        debug_dir
    )

    print("=" * 72)

    return {
        "text":
            extracted_text,

        "txt_path":
            txt_path,

        "json_path":
            json_path
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        image_path = input(
            "\nEnter news image path: "
        ).strip().strip('"')

        run_ocr(
            image_path
        )

    except Exception as error:

        print()
        print(
            "ERROR:"
        )

        print(
            error
        )