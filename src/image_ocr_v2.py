from pathlib import Path
from datetime import datetime
import json
import re

import cv2

import image_ocr as legacy_ocr

from ocr_v2_inference import (
    get_ocr_v2_engine
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v2"
)

DEBUG_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v2_debug"
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
# CHARACTER HELPERS
# ============================================================

def is_sinhala(
    character
):

    return (
        "\u0D80"
        <= character
        <= "\u0DFF"
    )


def is_english(
    character
):

    return (
        "A" <= character <= "Z"
        or
        "a" <= character <= "z"
    )


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(
    text
):

    if text is None:

        return ""

    text = str(
        text
    )

    text = text.replace(
        "\ufeff",
        ""
    )

    text = text.replace(
        "\u200b",
        ""
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(
    text
):

    sinhala_count = sum(
        1
        for character in text
        if is_sinhala(
            character
        )
    )

    english_count = sum(
        1
        for character in text
        if is_english(
            character
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

def load_image(
    image_path
):

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )

    image = cv2.imread(
        str(
            image_path
        ),
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    return image


# ============================================================
# CHECK WHETHER V2 OUTPUT IS USABLE
# ============================================================

def is_usable_v2_output(
    text
):

    text = clean_text(
        text
    )

    if not text:

        return False

    useful_characters = sum(
        1
        for character in text
        if (
            is_sinhala(
                character
            )
            or
            is_english(
                character
            )
            or
            character.isdigit()
        )
    )

    return (
        useful_characters
        >= 2
    )


# ============================================================
# RECOGNIZE ONE LINE
# V2 CRNN = PRIMARY
# TESSERACT = FALLBACK
# ============================================================

def recognize_line_v2(
    crop,
    crop_path,
    color_name,
    engine,
    tesseract_available
):

    # --------------------------------------------------------
    # 1. PRIMARY: Expanded V2 CRNN
    # --------------------------------------------------------

    v2_result = engine.predict_line(
        crop_path
    )

    v2_text = clean_text(
        v2_result[
            "text"
        ]
    )

    if is_usable_v2_output(
        v2_text
    ):

        return {

            "text":
                v2_text,

            "ocr_engine":
                "CRNN-CTC Expanded V2",

            "fallback_used":
                False,

            "confidence":
                None,

            "checkpoint_epoch":
                v2_result.get(
                    "checkpoint_epoch"
                ),

            "model_validation_cer":
                engine.validation_cer,

            "model_validation_wer":
                engine.validation_wer,

            "tesseract_language":
                None,

            "tesseract_psm":
                None,

            "tesseract_preprocessing":
                None
        }

    # --------------------------------------------------------
    # 2. FALLBACK: Existing Tesseract pipeline
    # --------------------------------------------------------

    if tesseract_available:

        best_result, _ = (
            legacy_ocr.recognize_line(
                crop,
                color_name
            )
        )

        fallback_text = clean_text(
            best_result[
                "text"
            ]
        )

        return {

            "text":
                fallback_text,

            "ocr_engine":
                "Tesseract fallback",

            "fallback_used":
                True,

            "confidence":
                best_result.get(
                    "confidence"
                ),

            "checkpoint_epoch":
                engine.checkpoint_epoch,

            "model_validation_cer":
                engine.validation_cer,

            "model_validation_wer":
                engine.validation_wer,

            "tesseract_language":
                best_result.get(
                    "language"
                ),

            "tesseract_psm":
                best_result.get(
                    "psm"
                ),

            "tesseract_preprocessing":
                best_result.get(
                    "preprocessing"
                )
        }

    # --------------------------------------------------------
    # 3. No usable fallback
    # --------------------------------------------------------

    return {

        "text":
            v2_text,

        "ocr_engine":
            "CRNN-CTC Expanded V2",

        "fallback_used":
            False,

        "confidence":
            None,

        "checkpoint_epoch":
            engine.checkpoint_epoch,

        "model_validation_cer":
            engine.validation_cer,

        "model_validation_wer":
            engine.validation_wer,

        "tesseract_language":
            None,

        "tesseract_psm":
            None,

        "tesseract_preprocessing":
            None
    }


# ============================================================
# EXTRACT COLOR-GUIDED HEADLINE
# ============================================================

def extract_colored_headline_v2(
    image,
    timestamp,
    engine,
    tesseract_available
):

    # Keep existing tested line-detection logic
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

    all_boxes = (
        yellow_boxes
        + white_boxes
    )

    all_boxes.sort(
        key=lambda item:
            item[
                "y"
            ]
    )

    if not all_boxes:

        raise ValueError(
            "No yellow/white headline "
            "lines detected."
        )

    print(
        f"\nDetected candidate lines: "
        f"{len(all_boxes)}"
    )

    debug_overlay = (
        image.copy()
    )

    line_debug_dir = (
        DEBUG_DIR
        / f"lines_{timestamp}"
    )

    line_debug_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    extracted_lines = []

    for index, box in enumerate(
        all_boxes,
        start=1
    ):

        # ----------------------------------------------------
        # Original crop
        # ----------------------------------------------------

        crop = (
            legacy_ocr.crop_original_line(
                image,
                box
            )
        )

        if (
            crop is None
            or crop.size == 0
        ):

            continue

        crop_path = (
            line_debug_dir
            / (
                f"line_{index}_"
                f"{box['color']}.png"
            )
        )

        cv2.imwrite(
            str(
                crop_path
            ),
            crop
        )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        result = recognize_line_v2(
            crop=crop,
            crop_path=crop_path,
            color_name=box[
                "color"
            ],
            engine=engine,
            tesseract_available=(
                tesseract_available
            )
        )

        text = clean_text(
            result[
                "text"
            ]
        )

        # Ignore fully empty output
        if not text:

            continue

        useful_count = sum(
            1
            for character in text
            if (
                is_sinhala(
                    character
                )
                or
                is_english(
                    character
                )
                or
                character.isdigit()
            )
        )

        if useful_count < 2:

            continue

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        extracted_lines.append({

            "line_number":
                len(
                    extracted_lines
                ) + 1,

            "color":
                box[
                    "color"
                ],

            "y":
                box[
                    "y"
                ],

            "text":
                text,

            "ocr_engine":
                result[
                    "ocr_engine"
                ],

            "fallback_used":
                result[
                    "fallback_used"
                ],

            "confidence":
                result[
                    "confidence"
                ],

            "checkpoint_epoch":
                result[
                    "checkpoint_epoch"
                ],

            "model_validation_cer":
                result[
                    "model_validation_cer"
                ],

            "model_validation_wer":
                result[
                    "model_validation_wer"
                ],

            "tesseract_language":
                result[
                    "tesseract_language"
                ],

            "tesseract_psm":
                result[
                    "tesseract_psm"
                ],

            "tesseract_preprocessing":
                result[
                    "tesseract_preprocessing"
                ],

            "debug_crop":
                str(
                    crop_path
                )
        })

        print(
            f"\nLine "
            f"{len(extracted_lines)}"
        )

        print(
            "Engine:",
            result[
                "ocr_engine"
            ]
        )

        print(
            "Text:"
        )

        print(
            text
        )

        # ----------------------------------------------------
        # Debug rectangle
        # ----------------------------------------------------

        x = box[
            "x"
        ]

        y = box[
            "y"
        ]

        w = box[
            "w"
        ]

        h = box[
            "h"
        ]

        cv2.rectangle(
            debug_overlay,
            (
                x,
                y
            ),
            (
                x + w,
                y + h
            ),
            (
                0,
                255,
                0
            ),
            2
        )

    extracted_lines.sort(
        key=lambda item:
            item[
                "y"
            ]
    )

    overlay_path = (
        DEBUG_DIR
        / (
            f"detected_lines_"
            f"{timestamp}.png"
        )
    )

    cv2.imwrite(
        str(
            overlay_path
        ),
        debug_overlay
    )

    return (
        extracted_lines,
        overlay_path
    )


# ============================================================
# QUALITY INFORMATION
# ============================================================

def calculate_quality(
    lines
):

    if not lines:

        return (
            "LOW",
            True
        )

    fallback_count = sum(
        1
        for line in lines
        if line[
            "fallback_used"
        ]
    )

    # Current V2 model improved strongly,
    # but exact word accuracy is not yet high enough
    # to call OCR output HIGH confidence.
    if (
        len(lines) >= 2
        and fallback_count == 0
    ):

        quality = "MEDIUM"

    else:

        quality = "LOW"

    # Keep manual review enabled at this stage
    return (
        quality,
        True
    )


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    image_path,
    lines,
    overlay_path,
    timestamp,
    engine
):

    final_text = "\n".join(
        line[
            "text"
        ]
        for line in lines
    )

    final_text = clean_text(
        final_text
    )

    detected_language = (
        detect_language(
            final_text
        )
    )

    quality_flag, manual_review = (
        calculate_quality(
            lines
        )
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

        "detected_language":
            detected_language,

        "extracted_text":
            final_text,

        # V2 CRNN does not currently provide
        # calibrated OCR confidence.
        "ocr_confidence":
            None,

        "quality_flag":
            quality_flag,

        "manual_verification_recommended":
            manual_review,

        "ocr_engine":
            (
                "CRNN-CTC Expanded V2 "
                "+ Tesseract fallback"
            ),

        "primary_ocr_engine":
            "CRNN-CTC Expanded V2",

        "fallback_ocr_engine":
            "Tesseract",

        "ocr_method":
            (
                "color_guided_line_detection_"
                "crnn_v2_primary_"
                "tesseract_fallback"
            ),

        "model_checkpoint_epoch":
            engine.checkpoint_epoch,

        "model_validation_cer":
            engine.validation_cer,

        "model_validation_wer":
            engine.validation_wer,

        "detected_line_count":
            len(
                lines
            ),

        "fallback_line_count":
            fallback_count,

        "lines":
            lines,

        "debug_overlay":
            str(
                overlay_path
            )
    }

    json_path = (
        RESULTS_DIR
        / (
            f"ocr_v2_result_"
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

    return (
        json_path,
        json_data
    )


# ============================================================
# RUN COMPLETE IMAGE OCR
# ============================================================

def extract_text_from_image(
    image_path
):

    print(
        "=" * 72
    )

    print(
        "SINHALA / ENGLISH NEWS IMAGE OCR"
    )

    print(
        "Expanded V2 CRNN + Tesseract Fallback"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # Load V2 model once
    # --------------------------------------------------------

    engine = (
        get_ocr_v2_engine()
    )

    print(
        "\nPrimary OCR model:"
    )

    print(
        engine.model_path
    )

    print(
        "\nCheckpoint epoch:",
        engine.checkpoint_epoch
    )

    if (
        engine.validation_cer
        is not None
    ):

        print(
            "Validation CER:",
            f"{engine.validation_cer:.4f}"
        )

    # --------------------------------------------------------
    # Check Tesseract fallback
    # --------------------------------------------------------

    tesseract_available = False

    try:

        languages = (
            legacy_ocr.check_tesseract()
        )

        tesseract_available = True

        print(
            "\nTesseract fallback: AVAILABLE"
        )

        print(
            "Languages:",
            languages
        )

    except Exception as error:

        print(
            "\nTesseract fallback: UNAVAILABLE"
        )

        print(
            error
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = load_image(
        image_path
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    # --------------------------------------------------------
    # Detect + OCR lines
    # --------------------------------------------------------

    lines, overlay_path = (
        extract_colored_headline_v2(
            image=image,
            timestamp=timestamp,
            engine=engine,
            tesseract_available=(
                tesseract_available
            )
        )
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    json_path, result = (
        save_result(
            image_path=image_path,
            lines=lines,
            overlay_path=overlay_path,
            timestamp=timestamp,
            engine=engine
        )
    )

    return (
        result,
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

        result, json_path = (
            extract_text_from_image(
                image_path
            )
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

        if result[
            "extracted_text"
        ]:

            print()

            print(
                result[
                    "extracted_text"
                ]
            )

        else:

            print(
                "\nNo reliable text detected."
            )

        print(
            "\nDetected Language:",
            result[
                "detected_language"
            ]
        )

        print(
            "Detected Lines:",
            result[
                "detected_line_count"
            ]
        )

        print(
            "Fallback Lines:",
            result[
                "fallback_line_count"
            ]
        )

        print(
            "OCR Quality:",
            result[
                "quality_flag"
            ]
        )

        print(
            "\nPrimary OCR Engine:"
        )

        print(
            result[
                "primary_ocr_engine"
            ]
        )

        print(
            "\nOCR Method:"
        )

        print(
            result[
                "ocr_method"
            ]
        )

        if result[
            "manual_verification_recommended"
        ]:

            print(
                "\nNOTE:"
            )

            print(
                "Manual verification is currently "
                "recommended for CRNN OCR output."
            )

        print(
            "\nJSON saved:"
        )

        print(
            json_path
        )

        print(
            "\nDebug overlay:"
        )

        print(
            result[
                "debug_overlay"
            ]
        )

        print(
            "\n" + "=" * 72
        )

    except Exception as error:

        print(
            "\nERROR:"
        )

        print(
            error
        )