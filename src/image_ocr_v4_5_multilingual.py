from pathlib import Path
from datetime import datetime
import json
import os
import re

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# ============================================================
# REUSE WORKING V4.2 HEADLINE / CROP FUNCTIONS
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
# SINHALA CUSTOM CRNN V3
# ============================================================

from ocr_v3_inference import get_ocr_v2_engine


# ============================================================
# VERSION
# ============================================================

PIPELINE_VERSION = (
    "Hybrid Multilingual Image OCR V4.5"
)


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v4_5_multilingual"
)

DEBUG_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v4_5_multilingual_debug"
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
# TESSERACT SETUP
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

    pytesseract.pytesseract.tesseract_cmd = str(
        TESSERACT_EXE
    )

    os.environ[
        "TESSDATA_PREFIX"
    ] = str(
        TESSDATA_DIR
    )

    print()
    print(
        "Tesseract configured."
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
# CHARACTER HELPERS
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
# CLEAN ENGLISH OCR TEXT
# ============================================================

def clean_english_text(text):

    cleaned_lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Remove OCR punctuation noise before the first
        # useful English/digit character.
        line = re.sub(
            r"^[^A-Za-z0-9]+",
            "",
            line
        )

        line = line.strip()

        english_count = (
            count_english_characters(
                line
            )
        )

        # Remove lines such as:
        # -
        # R
        # ;
        if english_count < 2:
            continue

        cleaned_lines.append(
            line
        )

    return "\n".join(
        cleaned_lines
    ).strip()


# ============================================================
# GENERAL TESSERACT OCR
# ============================================================

def run_tesseract(
    image,
    language,
    psm=6
):

    config = (
        f"--oem 1 "
        f"--psm {psm} "
        "-c preserve_interword_spaces=1"
    )

    raw_text = pytesseract.image_to_string(
        image,
        lang=language,
        config=config
    )

    raw_text = (
        raw_text
        .replace(
            "\x0c",
            ""
        )
        .strip()
    )

    data = pytesseract.image_to_data(
        image,
        lang=language,
        config=config,
        output_type=Output.DICT
    )

    confidences = []

    words = []

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

        words.append(
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

    if language == "eng":

        cleaned_text = (
            clean_english_text(
                raw_text
            )
        )

    else:

        cleaned_text = raw_text

    return {

        "raw_text":
            raw_text,

        "text":
            cleaned_text,

        "confidence":
            mean_confidence,

        "word_count":
            len(
                words
            ),

        "english_characters":
            count_english_characters(
                cleaned_text
            ),

        "sinhala_characters":
            count_sinhala_characters(
                cleaned_text
            )
    }


# ============================================================
# V4.5 ENGLISH-FIRST PROBE
#
# KEY CHANGE:
#
# BEFORE using Sinhala-style colour headline detection,
# inspect the lower part of the image with Tesseract ENG.
#
# Confirmed working region:
# 50% -> 98%
# ============================================================

def english_first_probe(
    image,
    debug_dir
):

    image_height, image_width = (
        image.shape[:2]
    )

    y1 = int(
        image_height * 0.50
    )

    y2 = int(
        image_height * 0.98
    )

    lower_roi = image[
        y1:y2,
        0:image_width
    ]

    roi_path = (
        debug_dir
        / "english_first_probe_roi.png"
    )

    cv2.imwrite(
        str(
            roi_path
        ),
        lower_roi
    )

    # --------------------------------------------------------
    # ENGLISH PROBE
    # --------------------------------------------------------

    english_result = (
        run_tesseract(
            lower_roi,
            "eng",
            psm=6
        )
    )

    # --------------------------------------------------------
    # SINHALA PROBE
    #
    # This is NOT the final Sinhala OCR.
    # It is only used to avoid incorrectly routing Sinhala
    # images to the English OCR branch.
    # --------------------------------------------------------

    sinhala_result = (
        run_tesseract(
            lower_roi,
            "sin",
            psm=6
        )
    )

    english_chars = (
        english_result[
            "english_characters"
        ]
    )

    english_sinhala_chars = (
        english_result[
            "sinhala_characters"
        ]
    )

    sinhala_chars = (
        sinhala_result[
            "sinhala_characters"
        ]
    )

    sinhala_english_chars = (
        sinhala_result[
            "english_characters"
        ]
    )

    english_total = max(
        1,
        english_chars
        + english_sinhala_chars
    )

    sinhala_total = max(
        1,
        sinhala_chars
        + sinhala_english_chars
    )

    english_script_ratio = (
        english_chars
        / float(
            english_total
        )
    )

    sinhala_script_ratio = (
        sinhala_chars
        / float(
            sinhala_total
        )
    )

    # ========================================================
    # ENGLISH STRONG CONDITION
    # ========================================================

    english_strong = (

        english_chars >= 12

        and english_result[
            "word_count"
        ] >= 3

        and english_result[
            "confidence"
        ] >= 50.0

        and english_script_ratio >= 0.85
    )

    # ========================================================
    # SINHALA SUPPORT CONDITION
    # ========================================================

    sinhala_strong = (

        sinhala_chars >= 4

        and sinhala_result[
            "confidence"
        ] >= 20.0

        and sinhala_script_ratio >= 0.50
    )

    # ========================================================
    # ROUTE DECISION
    # ========================================================

    if english_strong:

        if sinhala_strong:

            # If Sinhala also appears strong, English must
            # clearly outperform it.
            if (
                english_result[
                    "confidence"
                ]
                >=
                sinhala_result[
                    "confidence"
                ]
                + 8.0
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

        "roi":
            lower_roi,

        "roi_path":
            str(
                roi_path
            ),

        "roi_coordinates":
            {
                "x1":
                    0,

                "y1":
                    int(
                        y1
                    ),

                "x2":
                    int(
                        image_width
                    ),

                "y2":
                    int(
                        y2
                    )
            },

        "english":
            english_result,

        "sinhala_probe":
            sinhala_result,

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
# SINHALA HEADLINE / LINE EXTRACTION
# ============================================================

def extract_sinhala_headline(
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
        "STEP 2: Sinhala headline detection"
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
    # DEBUG MASKS
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
    # COLOUR DETECTION SUCCESS
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
            "Training-style lines found:",
            len(
                training_crops
            )
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
                "Using adaptive colour line crops."
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
    # FALLBACK
    # ========================================================

    if not line_crops:

        print()
        print(
            "Colour headline route failed."
        )

        print(
            "Trying generic fallback..."
        )

        (
            fallback,
            fallback_attempts
        ) = generic_window_fallback(
            image
        )

        if fallback is None:

            raise RuntimeError(
                "No usable Sinhala OCR lines found."
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
    # SAVE HEADLINE ROI
    # ========================================================

    headline_roi_path = (
        debug_dir
        / "selected_sinhala_headline_roi.png"
    )

    if headline_roi is not None:

        cv2.imwrite(
            str(
                headline_roi_path
            ),
            headline_roi
        )

    segmentation_path = (
        debug_dir
        / "selected_segmentation_mask.png"
    )

    if segmentation_mask is not None:

        cv2.imwrite(
            str(
                segmentation_path
            ),
            segmentation_mask
        )

    return {

        "headline_roi":
            headline_roi,

        "headline_roi_path":
            str(
                headline_roi_path
            ),

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
# CUSTOM SINHALA CRNN
# ============================================================

def run_sinhala_crnn(
    line_crops,
    engine,
    debug_dir
):

    results = []

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
        item[
            "text"
        ]
        for item in results
        if item[
            "text"
        ]
    )

    return (
        combined_text,
        results
    )


# ============================================================
# ENGLISH LINE RECORDS
# ============================================================

def create_english_results(
    text,
    confidence
):

    lines = [
        line.strip()
        for line in text.splitlines()
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
# COMPLETE V4.5 PIPELINE
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

    print()
    print(
        "Input image:"
    )

    print(
        image_path
    )

    print(
        f"\nInput image size: "
        f"{image_width} x {image_height}"
    )

    print(
        "\nRequested language mode:",
        language_mode
    )

    print()
    print(
        "STEP 1: English-first language probe"
    )

    probe_info = None

    # ========================================================
    # FORCED SINHALA
    # ========================================================

    if language_mode == "si":

        selected_language = "si"

        print(
            "Language forced to Sinhala."
        )

    # ========================================================
    # AUTO / ENGLISH
    # ========================================================

    else:

        probe_info = (
            english_first_probe(
                image,
                debug_dir
            )
        )

        print()
        print(
            "English lower-region OCR:"
        )

        print(
            probe_info[
                "english"
            ][
                "text"
            ]
        )

        print(
            "\nEnglish confidence:",
            round(
                probe_info[
                    "english"
                ][
                    "confidence"
                ],
                2
            )
        )

        print(
            "English characters:",
            probe_info[
                "english"
            ][
                "english_characters"
            ]
        )

        print(
            "English words:",
            probe_info[
                "english"
            ][
                "word_count"
            ]
        )

        print()
        print(
            "Sinhala lower-region probe:"
        )

        print(
            probe_info[
                "sinhala_probe"
            ][
                "text"
            ]
        )

        print(
            "\nSinhala probe confidence:",
            round(
                probe_info[
                    "sinhala_probe"
                ][
                    "confidence"
                ],
                2
            )
        )

        print(
            "Sinhala characters:",
            probe_info[
                "sinhala_probe"
            ][
                "sinhala_characters"
            ]
        )

        if language_mode == "en":

            selected_language = "en"

            print()
            print(
                "Language forced to English."
            )

        else:

            selected_language = (
                probe_info[
                    "selected_language"
                ]
            )

            print()
            print(
                "AUTO SELECTED LANGUAGE:",
                selected_language
            )

    # ========================================================
    # ENGLISH BRANCH
    # ========================================================

    if selected_language == "en":

        print()
        print(
            "STEP 2: English OCR branch"
        )

        english_result = (
            probe_info[
                "english"
            ]
        )

        extracted_text = (
            english_result[
                "text"
            ]
        )

        selected_engine = (
            "Tesseract ENG PSM 6"
        )

        extraction_method = (
            "english_lower_region_50_to_98"
        )

        ocr_results = (
            create_english_results(
                extracted_text,
                english_result[
                    "confidence"
                ]
            )
        )

        print()
        print(
            "English OCR result:"
        )

        print(
            extracted_text
        )

        print()
        print(
            "Mean confidence:",
            round(
                english_result[
                    "confidence"
                ],
                2
            )
        )

        sinhala_extraction = None

        sinhala_model_path = None

    # ========================================================
    # SINHALA BRANCH
    # ========================================================

    else:

        print()
        print(
            "STEP 2: Sinhala OCR branch"
        )

        sinhala_extraction = (
            extract_sinhala_headline(
                image,
                debug_dir
            )
        )

        line_crops = (
            sinhala_extraction[
                "line_crops"
            ]
        )

        print()
        print(
            "Sinhala extraction method:"
        )

        print(
            sinhala_extraction[
                "extraction_method"
            ]
        )

        print(
            "Sinhala line crops:",
            len(
                line_crops
            )
        )

        print()
        print(
            "Loading Custom Sinhala CRNN V3..."
        )

        crnn_engine = (
            get_ocr_v2_engine()
        )

        sinhala_model_path = str(
            crnn_engine.model_path
        )

        print(
            sinhala_model_path
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

        extraction_method = (
            sinhala_extraction[
                "extraction_method"
            ]
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
    # SAVE TEXT
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

        "extraction_method":
            extraction_method,

        "english_first_probe":
            None
            if probe_info is None
            else {

                "roi_path":
                    probe_info[
                        "roi_path"
                    ],

                "roi_coordinates":
                    probe_info[
                        "roi_coordinates"
                    ],

                "english":
                    probe_info[
                        "english"
                    ],

                "sinhala_probe":
                    probe_info[
                        "sinhala_probe"
                    ],

                "english_script_ratio":
                    probe_info[
                        "english_script_ratio"
                    ],

                "sinhala_script_ratio":
                    probe_info[
                        "sinhala_script_ratio"
                    ],

                "english_strong":
                    probe_info[
                        "english_strong"
                    ],

                "sinhala_strong":
                    probe_info[
                        "sinhala_strong"
                    ]
            },

        "sinhala_extraction":
            None
            if sinhala_extraction is None
            else {

                "extraction_method":
                    sinhala_extraction[
                        "extraction_method"
                    ],

                "headline_roi":
                    sinhala_extraction[
                        "roi_info"
                    ],

                "headline_roi_path":
                    sinhala_extraction[
                        "headline_roi_path"
                    ],

                "selected_headline_lines":
                    len(
                        sinhala_extraction[
                            "selected_boxes"
                        ]
                    )
            },

        "ocr_line_count":
            len(
                ocr_results
            ),

        "extracted_text":
            extracted_text,

        "ocr_results":
            ocr_results,

        "sinhala_model":
            sinhala_model_path,

        "english_engine":
            "Tesseract ENG PSM 6",

        "txt_path":
            str(
                txt_path
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
        "Extraction Method:",
        extraction_method
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
            "  en   = force English Tesseract"
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