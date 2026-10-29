from pathlib import Path
from datetime import datetime
import json

import cv2

import image_ocr as legacy_ocr
import auto_extract_new_real_text_lines as training_cropper

from ocr_v2_inference import get_ocr_v2_engine


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v2_aligned"
)

DEBUG_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v2_aligned_debug"
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
# HELPERS
# ============================================================

def is_sinhala(character):

    return (
        "\u0D80"
        <= character
        <= "\u0DFF"
    )


def is_english(character):

    return (
        "A" <= character <= "Z"
        or
        "a" <= character <= "z"
    )


def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.replace(
        "\ufeff",
        ""
    )

    text = text.replace(
        "\u200b",
        ""
    )

    return " ".join(
        text.split()
    )


def detect_language(text):

    sinhala_count = sum(
        1
        for character in text
        if is_sinhala(character)
    )

    english_count = sum(
        1
        for character in text
        if is_english(character)
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
# FIND HEADLINE ANCHORS
# Existing color detector is used ONLY to locate headline area
# ============================================================

def detect_headline_anchors(image):

    yellow_mask, white_mask = (
        legacy_ocr.create_text_masks(
            image
        )
    )

    yellow_boxes = (
        legacy_ocr.find_line_boxes(
            yellow_mask,
            "yellow"
        )
    )

    white_boxes = (
        legacy_ocr.find_line_boxes(
            white_mask,
            "white"
        )
    )

    anchors = (
        yellow_boxes
        + white_boxes
    )

    anchors.sort(
        key=lambda box:
            box["y"]
    )

    return anchors


# ============================================================
# BUILD ONE HEADLINE ROI
# ============================================================

def create_headline_roi(
    image,
    anchors
):

    if not anchors:

        raise ValueError(
            "No headline anchors detected."
        )

    height, width = (
        image.shape[:2]
    )

    x1 = min(
        box["x"]
        for box in anchors
    )

    y1 = min(
        box["y"]
        for box in anchors
    )

    x2 = max(
        box["x"] + box["w"]
        for box in anchors
    )

    y2 = max(
        box["y"] + box["h"]
        for box in anchors
    )

    # Generous ROI instead of tight line crops
    pad_x = max(
        12,
        int(
            width * 0.035
        )
    )

    pad_y = max(
        10,
        int(
            height * 0.025
        )
    )

    x1 = max(
        0,
        x1 - pad_x
    )

    y1 = max(
        0,
        y1 - pad_y
    )

    x2 = min(
        width,
        x2 + pad_x
    )

    y2 = min(
        height,
        y2 + pad_y
    )

    roi = image[
        y1:y2,
        x1:x2
    ]

    if roi.size == 0:

        raise ValueError(
            "Generated headline ROI is empty."
        )

    return (
        roi,
        (
            x1,
            y1,
            x2,
            y2
        )
    )


# ============================================================
# TRAINING-ALIGNED LINE DETECTION
# ============================================================

def detect_training_style_lines(
    roi
):

    mask = (
        training_cropper
        .create_text_mask(
            roi
        )
    )

    bands = (
        training_cropper
        .find_line_bands(
            mask
        )
    )

    candidates = (
        training_cropper
        .filter_line_bands(
            mask,
            bands
        )
    )

    candidates = (
        training_cropper
        .merge_overlapping_candidates(
            candidates
        )
    )

    boxes = [
        training_cropper.expand_box(
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
# GUESS ORIGINAL TEXT COLOR
# Used only for optional Tesseract fallback
# ============================================================

def guess_line_color(
    local_box,
    roi_box,
    anchors
):

    roi_x1, roi_y1, _, _ = (
        roi_box
    )

    _, local_y1, _, local_y2 = (
        local_box
    )

    global_center_y = (
        roi_y1
        + (
            local_y1
            + local_y2
        ) / 2
    )

    if not anchors:
        return "white"

    nearest = min(
        anchors,
        key=lambda anchor:
            abs(
                (
                    anchor["y"]
                    + anchor["h"] / 2
                )
                - global_center_y
            )
    )

    return nearest[
        "color"
    ]


# ============================================================
# OCR ONE TRAINING-STYLE CROP
# ============================================================

def recognize_crop(
    crop,
    crop_path,
    color_name,
    engine,
    tesseract_available
):

    v2_result = (
        engine.predict_line(
            crop_path
        )
    )

    v2_text = clean_text(
        v2_result[
            "text"
        ]
    )

    tesseract_text = ""

    tesseract_confidence = None

    # Run fallback candidate for comparison,
    # but V2 remains the primary result.
    if tesseract_available:

        try:

            best_tesseract, _ = (
                legacy_ocr.recognize_line(
                    crop,
                    color_name
                )
            )

            tesseract_text = clean_text(
                best_tesseract.get(
                    "text",
                    ""
                )
            )

            tesseract_confidence = (
                best_tesseract.get(
                    "confidence"
                )
            )

        except Exception:

            tesseract_text = ""

            tesseract_confidence = None

    # Only use Tesseract if V2 produced nothing useful
    useful_v2 = sum(
        1
        for character in v2_text
        if (
            is_sinhala(character)
            or
            is_english(character)
            or
            character.isdigit()
        )
    )

    if (
        useful_v2 < 2
        and tesseract_text
    ):

        selected_text = (
            tesseract_text
        )

        selected_engine = (
            "Tesseract fallback"
        )

        fallback_used = True

    else:

        selected_text = (
            v2_text
        )

        selected_engine = (
            "CRNN-CTC Expanded V2"
        )

        fallback_used = False

    return {

        "text":
            selected_text,

        "ocr_engine":
            selected_engine,

        "fallback_used":
            fallback_used,

        "v2_text":
            v2_text,

        "tesseract_text":
            tesseract_text,

        "tesseract_confidence":
            tesseract_confidence
    }


# ============================================================
# PROCESS FULL NEWS IMAGE
# ============================================================

def process_image(
    image_path
):

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

    current_debug_dir = (
        DEBUG_DIR
        / f"run_{timestamp}"
    )

    current_debug_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 72)

    print(
        "TRAINING-ALIGNED V2 NEWS IMAGE OCR"
    )

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

    # --------------------------------------------------------
    # Tesseract availability
    # --------------------------------------------------------

    tesseract_available = False

    try:

        languages = (
            legacy_ocr.check_tesseract()
        )

        tesseract_available = True

        print(
            "\nTesseract comparison: AVAILABLE"
        )

        print(
            "Languages:",
            languages
        )

    except Exception as error:

        print(
            "\nTesseract comparison: UNAVAILABLE"
        )

        print(
            error
        )

    # --------------------------------------------------------
    # Step 1: headline anchors
    # --------------------------------------------------------

    anchors = (
        detect_headline_anchors(
            image
        )
    )

    print(
        f"\nHeadline anchors detected: "
        f"{len(anchors)}"
    )

    if not anchors:

        raise ValueError(
            "No colored headline area detected."
        )

    # --------------------------------------------------------
    # Step 2: create headline ROI
    # --------------------------------------------------------

    roi, roi_box = (
        create_headline_roi(
            image,
            anchors
        )
    )

    roi_path = (
        current_debug_dir
        / "headline_roi.png"
    )

    cv2.imwrite(
        str(roi_path),
        roi
    )

    print(
        "\nHeadline ROI shape:"
    )

    print(
        roi.shape
    )

    # --------------------------------------------------------
    # Step 3: SAME line segmentation used for real training
    # --------------------------------------------------------

    mask, line_boxes = (
        detect_training_style_lines(
            roi
        )
    )

    print(
        f"\nTraining-style lines detected: "
        f"{len(line_boxes)}"
    )

    mask_path = (
        current_debug_dir
        / "training_style_mask.png"
    )

    cv2.imwrite(
        str(mask_path),
        mask
    )

    preview = (
        training_cropper
        .create_preview(
            roi,
            line_boxes
        )
    )

    preview_path = (
        current_debug_dir
        / "training_style_preview.png"
    )

    cv2.imwrite(
        str(preview_path),
        preview
    )

    # --------------------------------------------------------
    # Step 4: OCR every detected line
    # --------------------------------------------------------

    lines = []

    for line_number, box in enumerate(
        line_boxes,
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
            current_debug_dir
            / (
                f"line_"
                f"{line_number:02d}.png"
            )
        )

        cv2.imwrite(
            str(crop_path),
            crop
        )

        color_name = (
            guess_line_color(
                local_box=box,
                roi_box=roi_box,
                anchors=anchors
            )
        )

        result = (
            recognize_crop(
                crop=crop,
                crop_path=crop_path,
                color_name=color_name,
                engine=engine,
                tesseract_available=(
                    tesseract_available
                )
            )
        )

        print(
            "\n" + "-" * 72
        )

        print(
            f"LINE {line_number}"
        )

        print(
            "-" * 72
        )

        print(
            "Crop shape:",
            crop.shape
        )

        print(
            "Detected color:",
            color_name
        )

        print(
            "\nV2:"
        )

        print(
            result[
                "v2_text"
            ]
        )

        if tesseract_available:

            print(
                "\nTesseract:"
            )

            print(
                result[
                    "tesseract_text"
                ]
            )

        print(
            "\nSelected:"
        )

        print(
            result[
                "text"
            ]
        )

        lines.append({

            "line_number":
                line_number,

            "text":
                result[
                    "text"
                ],

            "v2_text":
                result[
                    "v2_text"
                ],

            "tesseract_text":
                result[
                    "tesseract_text"
                ],

            "tesseract_confidence":
                result[
                    "tesseract_confidence"
                ],

            "ocr_engine":
                result[
                    "ocr_engine"
                ],

            "fallback_used":
                result[
                    "fallback_used"
                ],

            "color":
                color_name,

            "crop_shape":
                list(
                    crop.shape
                ),

            "box":
                {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },

            "crop_path":
                str(
                    crop_path
                )
        })

    # --------------------------------------------------------
    # Step 5: combine text
    # --------------------------------------------------------

    extracted_text = "\n".join(
        line[
            "text"
        ]
        for line in lines
        if line[
            "text"
        ]
    )

    language = detect_language(
        extracted_text
    )

    fallback_count = sum(
        1
        for line in lines
        if line[
            "fallback_used"
        ]
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

        "ocr_method":
            (
                "headline_roi_plus_"
                "training_aligned_line_segmentation"
            ),

        "primary_ocr_engine":
            "CRNN-CTC Expanded V2",

        "checkpoint_epoch":
            engine.checkpoint_epoch,

        "model_validation_cer":
            engine.validation_cer,

        "model_validation_wer":
            engine.validation_wer,

        "detected_language":
            language,

        "headline_anchor_count":
            len(
                anchors
            ),

        "detected_line_count":
            len(
                lines
            ),

        "fallback_line_count":
            fallback_count,

        "extracted_text":
            extracted_text,

        "manual_verification_recommended":
            True,

        "headline_roi":
            {
                "x1": int(
                    roi_box[0]
                ),

                "y1": int(
                    roi_box[1]
                ),

                "x2": int(
                    roi_box[2]
                ),

                "y2": int(
                    roi_box[3]
                ),

                "shape":
                    list(
                        roi.shape
                    )
            },

        "lines":
            lines,

        "debug":
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
            f"aligned_ocr_"
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

    print(
        "\n" + "=" * 72
    )

    print(
        "FINAL EXTRACTED TEXT"
    )

    print(
        "=" * 72
    )

    print()

    print(
        extracted_text
        if extracted_text
        else "No text extracted."
    )

    print(
        "\nDetected Language:",
        language
    )

    print(
        "Detected Lines:",
        len(
            lines
        )
    )

    print(
        "Fallback Lines:",
        fallback_count
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
        current_debug_dir
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Manual verification remains "
        "recommended at the current OCR accuracy."
    )

    print(
        "=" * 72
    )

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

        process_image(
            image_path
        )

    except Exception as error:

        print(
            "\nERROR:"
        )

        print(
            error
        )