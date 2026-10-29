from pathlib import Path
from datetime import datetime
import json
import os

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# ============================================================
# REUSE WORKING V4.2 AUTOMATIC HEADLINE DETECTION
# ============================================================

from image_ocr_v4_hybrid import (
    PROJECT_ROOT,
    load_image,
    detect_colour_headline,
    build_colour_roi,
    extract_training_style_lines,
    create_adaptive_colour_crops,
    generic_window_fallback,
    create_preview,
)

# ============================================================
# SINHALA CUSTOM CRNN
# ============================================================

from ocr_v3_inference import get_ocr_v2_engine


# ============================================================
# VERSION
# ============================================================

PIPELINE_VERSION = (
    "Hybrid Multilingual Image OCR V4.4"
)


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v4_4_multilingual"
)

DEBUG_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v4_4_multilingual_debug"
)

TESSERACT_EXE = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

TESSDATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "tessdata_best"
)

ENG_TRAINEDDATA = (
    TESSDATA_DIR
    / "eng.traineddata"
)

SIN_TRAINEDDATA = (
    TESSDATA_DIR
    / "sin.traineddata"
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
# CONFIGURE TESSERACT
# ============================================================

def configure_tesseract():

    if not TESSERACT_EXE.exists():

        raise FileNotFoundError(
            "Tesseract executable not found:\n"
            f"{TESSERACT_EXE}"
        )

    if not ENG_TRAINEDDATA.exists():

        raise FileNotFoundError(
            "English traineddata not found:\n"
            f"{ENG_TRAINEDDATA}"
        )

    if not SIN_TRAINEDDATA.exists():

        raise FileNotFoundError(
            "Sinhala traineddata not found:\n"
            f"{SIN_TRAINEDDATA}"
        )

    # Tesseract executable
    pytesseract.pytesseract.tesseract_cmd = str(
        TESSERACT_EXE
    )

    # IMPORTANT:
    # Use environment variable instead of passing
    # --tessdata-dir through pytesseract config.
    os.environ[
        "TESSDATA_PREFIX"
    ] = str(
        TESSDATA_DIR
    )

    print(
        "\nTesseract configured."
    )

    print(
        "TESSDATA_PREFIX:"
    )

    print(
        os.environ[
            "TESSDATA_PREFIX"
        ]
    )


# ============================================================
# SCRIPT COUNT HELPERS
# ============================================================

def count_english_characters(text):

    return sum(
        1
        for character in text
        if (
            "A" <= character <= "Z"
            or
            "a" <= character <= "z"
        )
    )


def count_sinhala_characters(text):

    return sum(
        1
        for character in text
        if "\u0D80" <= character <= "\u0DFF"
    )


def detect_final_language(text):

    english_count = (
        count_english_characters(
            text
        )
    )

    sinhala_count = (
        count_sinhala_characters(
            text
        )
    )

    if (
        english_count > 0
        and sinhala_count > 0
    ):
        return "si+en"

    if english_count > 0:
        return "en"

    if sinhala_count > 0:
        return "si"

    return "unknown"


# ============================================================
# AUTOMATIC HEADLINE ROI + LINE CROPS
# ============================================================

def extract_headline_and_lines(
    image,
    debug_dir
):

    (
        yellow_mask,
        white_mask,
        scan_info,
        raw_boxes,
        text_boxes,
        selected_boxes
    ) = detect_colour_headline(
        image
    )

    print()
    print(
        "STEP 1: Automatic headline detection"
    )

    print(
        "Colour scan:"
    )

    print(
        scan_info
    )

    print(
        f"\nRaw candidates: "
        f"{len(raw_boxes)}"
    )

    for index, box in enumerate(
        raw_boxes,
        start=1
    ):

        print(
            f"  Raw {index}: "
            f"{box}"
        )

    print(
        f"\nText-like candidates: "
        f"{len(text_boxes)}"
    )

    print(
        f"Selected headline lines: "
        f"{len(selected_boxes)}"
    )

    for index, box in enumerate(
        selected_boxes,
        start=1
    ):

        print(
            f"  Headline {index}: "
            f"{box}"
        )

    # ========================================================
    # SAVE MASKS / PREVIEW
    # ========================================================

    yellow_path = (
        debug_dir
        / "yellow_mask.png"
    )

    white_path = (
        debug_dir
        / "white_mask.png"
    )

    preview_path = (
        debug_dir
        / "headline_detection_preview.png"
    )

    cv2.imwrite(
        str(
            yellow_path
        ),
        yellow_mask
    )

    cv2.imwrite(
        str(
            white_path
        ),
        white_mask
    )

    preview = create_preview(
        image,
        raw_boxes,
        text_boxes,
        selected_boxes
    )

    cv2.imwrite(
        str(
            preview_path
        ),
        preview
    )

    headline_roi = None
    roi_info = None
    segmentation_mask = None
    line_crops = []
    extraction_method = None
    fallback_attempts = []

    # ========================================================
    # COLOUR ROUTE
    # ========================================================

    if selected_boxes:

        (
            headline_roi,
            roi_info
        ) = build_colour_roi(
            image,
            selected_boxes
        )

        (
            segmentation_mask,
            segmentation_boxes,
            training_crops
        ) = extract_training_style_lines(
            headline_roi
        )

        print()
        print(
            "STEP 2: Line segmentation"
        )

        print(
            f"Training-style lines found: "
            f"{len(training_crops)}"
        )

        if (
            len(training_crops)
            == len(selected_boxes)
        ):

            line_crops = (
                training_crops
            )

            extraction_method = (
                "colour_roi_plus_"
                "training_segmentation"
            )

        else:

            print(
                "Training segmentation count mismatch."
            )

            print(
                "Using adaptive colour crops."
            )

            line_crops = (
                create_adaptive_colour_crops(
                    image,
                    selected_boxes
                )
            )

            extraction_method = (
                "adaptive_colour_line_crop"
            )

    # ========================================================
    # GENERIC FALLBACK
    # ========================================================

    if (
        headline_roi is None
        or not line_crops
    ):

        print()
        print(
            "Colour route unavailable."
        )

        print(
            "Using generic fallback..."
        )

        (
            fallback,
            fallback_attempts
        ) = generic_window_fallback(
            image
        )

        if fallback is None:

            raise RuntimeError(
                "Automatic headline extraction failed."
            )

        headline_roi = (
            fallback[
                "roi"
            ]
        )

        segmentation_mask = (
            fallback[
                "mask"
            ]
        )

        line_crops = (
            fallback[
                "crops"
            ]
        )

        roi_info = (
            fallback[
                "info"
            ]
        )

        extraction_method = (
            "generic_window_fallback"
        )

    # ========================================================
    # SAVE FINAL ROI
    # ========================================================

    headline_roi_path = (
        debug_dir
        / "selected_headline_roi.png"
    )

    cv2.imwrite(
        str(
            headline_roi_path
        ),
        headline_roi
    )

    segmentation_mask_path = (
        debug_dir
        / "selected_segmentation_mask.png"
    )

    if segmentation_mask is not None:

        cv2.imwrite(
            str(
                segmentation_mask_path
            ),
            segmentation_mask
        )

    return {

        "headline_roi":
            headline_roi,

        "headline_roi_path":
            headline_roi_path,

        "line_crops":
            line_crops,

        "extraction_method":
            extraction_method,

        "scan_info":
            scan_info,

        "roi_info":
            roi_info,

        "raw_boxes":
            raw_boxes,

        "text_boxes":
            text_boxes,

        "selected_boxes":
            selected_boxes,

        "fallback_attempts":
            fallback_attempts
    }


# ============================================================
# TESSERACT WHOLE-ROI OCR
#
# KEY V4.4 FIX:
#
# English is NOT OCR'd one crop at a time.
#
# Whole detected headline ROI is sent to Tesseract with
# PSM 6, which was confirmed to work on the test image.
# ============================================================

def tesseract_roi_ocr(
    roi,
    language
):

    # ========================================================
    # IMPORTANT:
    #
    # Do NOT put --tessdata-dir here.
    # configure_tesseract() already sets TESSDATA_PREFIX.
    # ========================================================

    config = (
        '--oem 1 '
        '--psm 6 '
        '-c preserve_interword_spaces=1'
    )

    # ========================================================
    # OCR TEXT
    # ========================================================

    text = pytesseract.image_to_string(
        roi,
        lang=language,
        config=config
    )

    text = (
        text
        .replace(
            "\x0c",
            ""
        )
        .strip()
    )

    # ========================================================
    # OCR CONFIDENCE
    # ========================================================

    data = pytesseract.image_to_data(
        roi,
        lang=language,
        config=config,
        output_type=Output.DICT
    )

    confidences = []

    valid_words = []

    for word, confidence in zip(
        data["text"],
        data["conf"]
    ):

        word = str(
            word
        ).strip()

        try:

            confidence_value = float(
                confidence
            )

        except (
            ValueError,
            TypeError
        ):

            confidence_value = -1.0

        if not word:
            continue

        valid_words.append(
            word
        )

        if confidence_value >= 0:

            confidences.append(
                confidence_value
            )

    if confidences:

        mean_confidence = float(
            np.mean(
                confidences
            )
        )

    else:

        mean_confidence = 0.0

    return {

        "text":
            text,

        "confidence":
            mean_confidence,

        "word_count":
            len(
                valid_words
            ),

        "english_characters":
            count_english_characters(
                text
            ),

        "sinhala_characters":
            count_sinhala_characters(
                text
            )
    }


# ============================================================
# AUTOMATIC LANGUAGE ROUTING
#
# Probe the SAME headline ROI using:
#
# eng + PSM 6
# sin + PSM 6
#
# English strong -> use Tesseract whole ROI
#
# Otherwise -> use custom Sinhala CRNN line OCR
# ============================================================

def choose_language_route(
    headline_roi
):

    english_probe = (
        tesseract_roi_ocr(
            headline_roi,
            "eng"
        )
    )

    sinhala_probe = (
        tesseract_roi_ocr(
            headline_roi,
            "sin"
        )
    )

    english_text = (
        english_probe[
            "text"
        ]
    )

    sinhala_text = (
        sinhala_probe[
            "text"
        ]
    )

    english_chars = (
        count_english_characters(
            english_text
        )
    )

    english_sinhala_chars = (
        count_sinhala_characters(
            english_text
        )
    )

    sinhala_chars = (
        count_sinhala_characters(
            sinhala_text
        )
    )

    sinhala_english_chars = (
        count_english_characters(
            sinhala_text
        )
    )

    english_script_total = max(
        1,
        english_chars
        + english_sinhala_chars
    )

    sinhala_script_total = max(
        1,
        sinhala_chars
        + sinhala_english_chars
    )

    english_script_ratio = (
        english_chars
        / float(
            english_script_total
        )
    )

    sinhala_script_ratio = (
        sinhala_chars
        / float(
            sinhala_script_total
        )
    )

    # ========================================================
    # ENGLISH MUST BE CLEARLY STRONG
    # ========================================================

    english_strong = (

        english_chars >= 8

        and english_probe[
            "word_count"
        ] >= 2

        and english_script_ratio >= 0.80

        and english_probe[
            "confidence"
        ] >= 40.0
    )

    # ========================================================
    # SINHALA PROBE SUPPORT
    # ========================================================

    sinhala_strong = (

        sinhala_chars >= 4

        and sinhala_script_ratio >= 0.55

        and sinhala_probe[
            "confidence"
        ] >= 30.0
    )

    # ========================================================
    # FINAL ROUTE
    # ========================================================

    if english_strong:

        # If both probes appear strong,
        # confidence comparison is used.
        if sinhala_strong:

            if (
                english_probe[
                    "confidence"
                ]
                >= sinhala_probe[
                    "confidence"
                ] + 3.0
            ):

                selected_language = "en"

            else:

                selected_language = "si"

        else:

            selected_language = "en"

    else:

        selected_language = "si"

    return {

        "selected_language":
            selected_language,

        "english_probe":
            english_probe,

        "sinhala_probe":
            sinhala_probe,

        "english_script_ratio":
            float(
                english_script_ratio
            ),

        "sinhala_script_ratio":
            float(
                sinhala_script_ratio
            ),

        "english_strong":
            bool(
                english_strong
            ),

        "sinhala_strong":
            bool(
                sinhala_strong
            )
    }


# ============================================================
# SINHALA CRNN OCR
# ============================================================

def run_sinhala_crnn(
    line_crops,
    crnn_engine,
    debug_dir
):

    results = []

    print()
    print(
        "Running Sinhala Custom CRNN V3..."
    )

    for index, item in enumerate(
        line_crops,
        start=1
    ):

        crop = (
            item[
                "crop"
            ]
        )

        crop_path = (
            debug_dir
            / f"sinhala_line_{index:02d}.png"
        )

        cv2.imwrite(
            str(
                crop_path
            ),
            crop
        )

        prediction = (
            crnn_engine.predict_line(
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
            f"SINHALA LINE {index}"
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
                    index,

                "language":
                    "si",

                "engine":
                    "Custom CRNN V3",

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

                "crop_source":
                    item.get(
                        "source",
                        "unknown"
                    )
            }
        )

    combined_text = "\n".join(
        result[
            "text"
        ]
        for result in results
        if result[
            "text"
        ]
    )

    return (
        combined_text,
        results
    )


# ============================================================
# CONVERT ENGLISH WHOLE ROI TEXT INTO LINE RECORDS
# ============================================================

def create_english_line_results(
    english_text,
    confidence
):

    lines = [
        line.strip()
        for line in english_text.splitlines()
        if line.strip()
    ]

    results = []

    for index, line in enumerate(
        lines,
        start=1
    ):

        results.append(
            {
                "line_number":
                    index,

                "language":
                    "en",

                "engine":
                    "Tesseract ENG PSM 6",

                "text":
                    line,

                "confidence":
                    float(
                        confidence
                    )
            }
        )

    return results


# ============================================================
# COMPLETE V4.4 MULTILINGUAL PIPELINE
# ============================================================

def run_multilingual_ocr(
    image_path,
    language_mode="auto"
):

    configure_tesseract()

    image = load_image(
        image_path
    )

    image_height, image_width = (
        image.shape[:2]
    )

    crnn_engine = (
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
    print(PIPELINE_VERSION.upper())
    print("=" * 72)

    print(
        f"\nInput image size: "
        f"{image_width} x {image_height}"
    )

    print(
        "\nRequested language mode:",
        language_mode
    )

    print(
        "\nSinhala OCR:"
    )

    print(
        crnn_engine.model_path
    )

    print(
        "\nEnglish OCR:"
    )

    print(
        TESSERACT_EXE
    )

    # ========================================================
    # STEP 1 — HEADLINE EXTRACTION
    # ========================================================

    extraction = (
        extract_headline_and_lines(
            image,
            debug_dir
        )
    )

    headline_roi = (
        extraction[
            "headline_roi"
        ]
    )

    line_crops = (
        extraction[
            "line_crops"
        ]
    )

    print()
    print(
        "Headline extraction method:"
    )

    print(
        extraction[
            "extraction_method"
        ]
    )

    print(
        f"Sinhala line crops available: "
        f"{len(line_crops)}"
    )

    # ========================================================
    # STEP 2 — LANGUAGE ROUTING
    # ========================================================

    print()
    print(
        "STEP 3: Language routing"
    )

    route_info = None

    if language_mode == "auto":

        route_info = (
            choose_language_route(
                headline_roi
            )
        )

        selected_language = (
            route_info[
                "selected_language"
            ]
        )

        english_probe = (
            route_info[
                "english_probe"
            ]
        )

        sinhala_probe = (
            route_info[
                "sinhala_probe"
            ]
        )

        print()
        print(
            "English ROI OCR:"
        )

        print(
            english_probe[
                "text"
            ]
        )

        print(
            "English confidence:",
            round(
                english_probe[
                    "confidence"
                ],
                2
            )
        )

        print(
            "English characters:",
            english_probe[
                "english_characters"
            ]
        )

        print()
        print(
            "Sinhala ROI probe:"
        )

        print(
            sinhala_probe[
                "text"
            ]
        )

        print(
            "Sinhala confidence:",
            round(
                sinhala_probe[
                    "confidence"
                ],
                2
            )
        )

        print()
        print(
            "AUTO SELECTED LANGUAGE:",
            selected_language
        )

    elif language_mode == "en":

        selected_language = "en"

        route_info = {
            "selected_language":
                "en",

            "routing_mode":
                "forced_en"
        }

    elif language_mode == "si":

        selected_language = "si"

        route_info = {
            "selected_language":
                "si",

            "routing_mode":
                "forced_si"
        }

    else:

        raise ValueError(
            "Language mode must be auto, si, or en."
        )

    # ========================================================
    # STEP 4 — ENGLISH OCR
    # ========================================================

    if selected_language == "en":

        print()
        print(
            "STEP 4: English whole-headline OCR"
        )

        english_result = (
            tesseract_roi_ocr(
                headline_roi,
                "eng"
            )
        )

        extracted_text = (
            english_result[
                "text"
            ]
        )

        ocr_results = (
            create_english_line_results(
                extracted_text,
                english_result[
                    "confidence"
                ]
            )
        )

        selected_engine = (
            "Tesseract ENG PSM 6"
        )

        print()
        print(
            "English OCR result:"
        )

        print(
            extracted_text
        )

        print(
            "\nMean confidence:",
            round(
                english_result[
                    "confidence"
                ],
                2
            )
        )

    # ========================================================
    # STEP 4 — SINHALA OCR
    # ========================================================

    else:

        print()
        print(
            "STEP 4: Sinhala line-based OCR"
        )

        (
            extracted_text,
            ocr_results
        ) = run_sinhala_crnn(
            line_crops,
            crnn_engine,
            debug_dir
        )

        selected_engine = (
            "Custom CRNN V3"
        )

    # ========================================================
    # FINAL LANGUAGE
    # ========================================================

    final_language = (
        detect_final_language(
            extracted_text
        )
    )

    if final_language == "unknown":

        final_language = (
            selected_language
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

        "pipeline_version":
            PIPELINE_VERSION,

        "input_type":
            "image",

        "image_path":
            str(
                Path(
                    image_path
                ).resolve()
            ),

        "image_size":
            {
                "width":
                    int(
                        image_width
                    ),

                "height":
                    int(
                        image_height
                    )
            },

        "requested_language_mode":
            language_mode,

        "selected_language":
            selected_language,

        "final_detected_language":
            final_language,

        "selected_ocr_engine":
            selected_engine,

        "headline_extraction_method":
            extraction[
                "extraction_method"
            ],

        "headline_roi":
            extraction[
                "roi_info"
            ],

        "headline_roi_path":
            str(
                extraction[
                    "headline_roi_path"
                ]
            ),

        "colour_scan":
            extraction[
                "scan_info"
            ],

        "selected_headline_line_count":
            len(
                extraction[
                    "selected_boxes"
                ]
            ),

        "language_routing":
            route_info,

        "ocr_line_count":
            len(
                ocr_results
            ),

        "extracted_text":
            extracted_text,

        "txt_path":
            str(
                txt_path
            ),

        "ocr_results":
            ocr_results,

        "sinhala_model":
            str(
                crnn_engine.model_path
            ),

        "english_engine":
            "Tesseract ENG PSM 6",

        "english_traineddata":
            str(
                ENG_TRAINEDDATA
            )
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
    # FINAL OUTPUT
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

    print()
    print(
        "Selected Language:",
        selected_language
    )

    print(
        "Final Detected Language:",
        final_language
    )

    print(
        "OCR Engine:",
        selected_engine
    )

    print(
        "Headline Extraction:",
        extraction[
            "extraction_method"
        ]
    )

    print(
        "OCR Lines:",
        len(
            ocr_results
        )
    )

    print()
    print(
        "TXT saved:"
    )

    print(
        txt_path
    )

    print()
    print(
        "JSON saved:"
    )

    print(
        json_path
    )

    print()
    print(
        "Debug folder:"
    )

    print(
        debug_dir
    )

    print("=" * 72)

    return {

        "text":
            extracted_text,

        "selected_language":
            selected_language,

        "final_language":
            final_language,

        "engine":
            selected_engine,

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

        print()
        print(
            "Language mode:"
        )

        print(
            "  auto = automatically select Sinhala / English"
        )

        print(
            "  si   = force Sinhala Custom CRNN"
        )

        print(
            "  en   = force English Tesseract PSM 6"
        )

        language_mode = input(
            "\nEnter mode [auto]: "
        ).strip().lower()

        if not language_mode:

            language_mode = "auto"

        if language_mode not in {
            "auto",
            "si",
            "en"
        }:

            raise ValueError(
                "Mode must be auto, si, or en."
            )

        run_multilingual_ocr(
            image_path,
            language_mode
        )

    except Exception as error:

        print()
        print(
            "ERROR:"
        )

        print(
            error
        )