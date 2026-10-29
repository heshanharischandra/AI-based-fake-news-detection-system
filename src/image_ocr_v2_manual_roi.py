from pathlib import Path
from datetime import datetime
import json

import cv2

import auto_extract_new_real_text_lines as line_extractor

from ocr_v3_inference import get_ocr_v2_engine


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v2_manual_roi"
)

DEBUG_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v2_manual_roi_debug"
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

    if (
        sinhala_count > 0
        and english_count > 0
    ):
        return "si+en"

    if sinhala_count > 0:
        return "si"

    if english_count > 0:
        return "en"

    return "unknown"


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    image_path = Path(
        image_path
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

    return image


# ============================================================
# MANUAL HEADLINE ROI SELECTION
# ============================================================

def select_headline_roi(image):

    print()
    print("=" * 72)
    print("HEADLINE REGION SELECTION")
    print("=" * 72)

    print(
        "\nSelect the FULL headline area."
    )

    print(
        "Make sure ALL headline lines are inside the rectangle."
    )

    print(
        "\nPress ENTER or SPACE to confirm."
    )

    print(
        "Press C to cancel."
    )

    # ========================================================
    # FIT IMAGE TO SCREEN FOR EASY ROI SELECTION
    # ========================================================

    original_height, original_width = (
        image.shape[:2]
    )

    max_display_width = 1100
    max_display_height = 650

    scale_width = (
        max_display_width
        / original_width
    )

    scale_height = (
        max_display_height
        / original_height
    )

    display_scale = min(
        1.0,
        scale_width,
        scale_height
    )

    display_width = int(
        round(
            original_width
            * display_scale
        )
    )

    display_height = int(
        round(
            original_height
            * display_scale
        )
    )

    if display_scale < 1.0:

        display_image = cv2.resize(
            image,
            (
                display_width,
                display_height
            ),
            interpolation=cv2.INTER_AREA
        )

    else:

        display_image = (
            image.copy()
        )

    print(
        "\nOriginal image size:",
        (
            original_width,
            original_height
        )
    )

    print(
        "Selection preview size:",
        (
            display_width,
            display_height
        )
    )

    print(
        "Preview scale:",
        round(
            display_scale,
            4
        )
    )

    # ========================================================
    # SELECT ROI ON DISPLAY IMAGE
    # ========================================================

    dx, dy, dw, dh = cv2.selectROI(
        "Select FULL Headline Region",
        display_image,
        fromCenter=False,
        showCrosshair=True
    )

    cv2.destroyAllWindows()

    if dw <= 0 or dh <= 0:

        raise ValueError(
            "Headline ROI selection cancelled."
        )

    # ========================================================
    # MAP DISPLAY COORDINATES BACK TO ORIGINAL IMAGE
    # ========================================================

    x = int(
        round(
            dx
            / display_scale
        )
    )

    y = int(
        round(
            dy
            / display_scale
        )
    )

    w = int(
        round(
            dw
            / display_scale
        )
    )

    h = int(
        round(
            dh
            / display_scale
        )
    )

    # Clamp original selection
    x = max(
        0,
        min(
            x,
            original_width - 1
        )
    )

    y = max(
        0,
        min(
            y,
            original_height - 1
        )
    )

    w = min(
        w,
        original_width - x
    )

    h = min(
        h,
        original_height - y
    )

    # ========================================================
    # TRAINING-STYLE CALIBRATION
    # ========================================================

    horizontal_shift = int(
        round(
            w * 0.02
        )
    )

    top_expansion = int(
        round(
            h * 0.03
        )
    )

    target_height = int(
        round(
            h * 1.16
        )
    )

    adjusted_x = max(
        0,
        x - horizontal_shift
    )

    adjusted_y = max(
        0,
        y - top_expansion
    )

    adjusted_width = min(
        w,
        original_width - adjusted_x
    )

    adjusted_height = min(
        target_height,
        original_height - adjusted_y
    )

    roi = image[
        adjusted_y:
            adjusted_y + adjusted_height,

        adjusted_x:
            adjusted_x + adjusted_width
    ]

    if roi.size == 0:

        raise ValueError(
            "Adjusted headline ROI is empty."
        )

    original_selection = {
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h)
    }

    adjusted_selection = {
        "x": int(adjusted_x),
        "y": int(adjusted_y),
        "width": int(adjusted_width),
        "height": int(adjusted_height),
        "right": int(
            adjusted_x
            + adjusted_width
        ),
        "bottom": int(
            adjusted_y
            + adjusted_height
        )
    }

    print(
        "\nOriginal selection:"
    )

    print(
        original_selection
    )

    print(
        "\nTraining-style adjusted selection:"
    )

    print(
        adjusted_selection
    )

    return (
        roi,
        {
            "display_scale":
                float(
                    display_scale
                ),

            "original_selection":
                original_selection,

            **adjusted_selection
        }
    )
    # --------------------------------------------------------
    # Training-style calibration
    # --------------------------------------------------------

    horizontal_shift = int(
        round(
            w * 0.02
        )
    )

    top_expansion = int(
        round(
            h * 0.03
        )
    )

    target_height = int(
        round(
            h * 1.16
        )
    )

    adjusted_x = max(
        0,
        x - horizontal_shift
    )

    adjusted_y = max(
        0,
        y - top_expansion
    )

    adjusted_width = min(
        w,
        image_width - adjusted_x
    )

    adjusted_height = min(
        target_height,
        image_height - adjusted_y
    )

    roi = image[
        adjusted_y:
            adjusted_y + adjusted_height,

        adjusted_x:
            adjusted_x + adjusted_width
    ]

    if roi.size == 0:

        raise ValueError(
            "Adjusted headline ROI is empty."
        )

    original_selection = {
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h)
    }

    adjusted_selection = {
        "x": int(adjusted_x),
        "y": int(adjusted_y),
        "width": int(adjusted_width),
        "height": int(adjusted_height),
        "right": int(
            adjusted_x
            + adjusted_width
        ),
        "bottom": int(
            adjusted_y
            + adjusted_height
        )
    }

    print(
        "\nOriginal selection:"
    )

    print(
        original_selection
    )

    print(
        "\nTraining-style adjusted selection:"
    )

    print(
        adjusted_selection
    )

    return (
        roi,
        {
            "original_selection":
                original_selection,

            **adjusted_selection
        }
    )


# ============================================================
# TRAINING-STYLE LINE DETECTION
# ============================================================

def detect_lines(roi):

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

    return (
        mask,
        boxes
    )


# ============================================================
# RUN COMPLETE OCR
# ============================================================

def run_ocr(image_path):

    image = load_image(
        image_path
    )

    engine = get_ocr_v2_engine()

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    run_debug_dir = (
        DEBUG_DIR
        / f"run_{timestamp}"
    )

    run_debug_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 72)
    print("MANUAL-ROI TRAINING-ALIGNED V2 OCR")
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
    # SELECT HEADLINE ROI
    # ========================================================

    roi, roi_info = (
        select_headline_roi(
            image
        )
    )

    roi_path = (
        run_debug_dir
        / "headline_roi.png"
    )

    cv2.imwrite(
        str(roi_path),
        roi
    )

    print(
        "\nROI shape:",
        roi.shape
    )

    # ========================================================
    # DETECT LINES
    # ========================================================

    mask, boxes = (
        detect_lines(
            roi
        )
    )

    print(
        f"\nDetected text lines: "
        f"{len(boxes)}"
    )

    mask_path = (
        run_debug_dir
        / "line_mask.png"
    )

    cv2.imwrite(
        str(mask_path),
        mask
    )

    preview = (
        line_extractor
        .create_preview(
            roi,
            boxes
        )
    )

    preview_path = (
        run_debug_dir
        / "line_preview.png"
    )

    cv2.imwrite(
        str(preview_path),
        preview
    )

    # ========================================================
    # OCR EACH LINE
    # ========================================================

    lines = []

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

        crop_path = (
            run_debug_dir
            / f"line_{number:02d}.png"
        )

        cv2.imwrite(
            str(crop_path),
            crop
        )

        result = (
            engine.predict_line(
                crop_path
            )
        )

        text = (
            result[
                "text"
            ]
            .strip()
        )

        print()
        print("-" * 72)
        print(f"LINE {number}")
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

        lines.append(
            {
                "line_number":
                    number,

                "text":
                    text,

                "crop_path":
                    str(
                        crop_path
                    ),

                "crop_shape":
                    list(
                        crop.shape
                    ),

                "box":
                    {
                        "x1":
                            int(x1),

                        "y1":
                            int(y1),

                        "x2":
                            int(x2),

                        "y2":
                            int(y2)
                    }
            }
        )

    # ========================================================
    # FINAL TEXT
    # ========================================================

    extracted_text = "\n".join(
        line[
            "text"
        ]
        for line in lines
        if line[
            "text"
        ]
    )

    language = (
        detect_language(
            extracted_text
        )
    )

    # ========================================================
    # JSON RESULT
    # ========================================================

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

        "ocr_method":
            (
                "manual_headline_roi_"
                "training_aligned_line_detection"
            ),

        "checkpoint_epoch":
            engine.checkpoint_epoch,

        "model_validation_cer":
            engine.validation_cer,

        "model_validation_wer":
            engine.validation_wer,

        "headline_roi":
            roi_info,

        "detected_line_count":
            len(lines),

        "detected_language":
            language,

        "extracted_text":
            extracted_text,

        "manual_verification_recommended":
            True,

        "lines":
            lines,

        "debug_files":
            {
                "roi":
                    str(
                        roi_path
                    ),

                "mask":
                    str(
                        mask_path
                    ),

                "preview":
                    str(
                        preview_path
                    )
            }
    }

    json_path = (
        RESULTS_DIR
        / (
            f"ocr_result_"
            f"{timestamp}.json"
        )
    )

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
    # TERMINAL SUMMARY
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
        "Detected Lines:",
        len(lines)
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
        run_debug_dir
    )

    print("=" * 72)

    return (
        json_data,
        json_path
    )


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

        print(
            "\nERROR:"
        )

        print(
            error
        )

