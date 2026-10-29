from pathlib import Path
from datetime import datetime
import json

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# ------------------------------------------------------------
# Reuse the already-working V4.2 automatic headline extraction
# ------------------------------------------------------------

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

# Sinhala custom CRNN V3
from ocr_v3_inference import get_ocr_v2_engine


# ============================================================
# VERSION
# ============================================================

PIPELINE_VERSION = (
    "Hybrid Multilingual Image OCR V4.3"
)


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v4_3_multilingual"
)

DEBUG_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v4_3_multilingual_debug"
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

    pytesseract.pytesseract.tesseract_cmd = str(
        TESSERACT_EXE
    )


# ============================================================
# SCRIPT COUNTS
# ============================================================

def count_latin_characters(text):

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


def count_digits(text):

    return sum(
        1
        for character in text
        if character.isdigit()
    )


# ============================================================
# DETECT FINAL TEXT LANGUAGE
# ============================================================

def detect_language(text):

    sinhala_count = (
        count_sinhala_characters(
            text
        )
    )

    english_count = (
        count_latin_characters(
            text
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
# ENGLISH OCR PREPROCESSING
# ============================================================

def create_tesseract_variants(crop):

    variants = []

    # --------------------------------------------------------
    # VARIANT 1 — enlarged original
    # --------------------------------------------------------

    enlarged = cv2.resize(
        crop,
        None,
        fx=3.0,
        fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )

    variants.append(
        (
            "enlarged_original",
            enlarged
        )
    )

    # --------------------------------------------------------
    # VARIANT 2 — grayscale + CLAHE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    variants.append(
        (
            "grayscale_clahe",
            enhanced
        )
    )

    # --------------------------------------------------------
    # VARIANT 3 — Otsu threshold
    # --------------------------------------------------------

    _, binary = cv2.threshold(
        enhanced,
        0,
        255,
        (
            cv2.THRESH_BINARY
            + cv2.THRESH_OTSU
        )
    )

    # Tesseract usually works better with
    # dark letters on a white background.
    if np.mean(binary) < 127:

        binary = cv2.bitwise_not(
            binary
        )

    variants.append(
        (
            "otsu_binary",
            binary
        )
    )

    # --------------------------------------------------------
    # VARIANT 4 — adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    if np.mean(adaptive) < 127:

        adaptive = cv2.bitwise_not(
            adaptive
        )

    variants.append(
        (
            "adaptive_binary",
            adaptive
        )
    )

    return variants


# ============================================================
# TESSERACT RESULT FROM ONE VARIANT
# ============================================================

def tesseract_variant_result(
    image,
    language
):

    config = (
        '--oem 1 '
        '--psm 7 '
        f'--tessdata-dir "{TESSDATA_DIR}" '
        '-c preserve_interword_spaces=1'
    )

    data = pytesseract.image_to_data(
        image,
        lang=language,
        config=config,
        output_type=Output.DICT
    )

    words = []

    confidences = []

    for text, confidence in zip(
        data["text"],
        data["conf"]
    ):

        text = str(
            text
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

        if not text:

            continue

        words.append(
            text
        )

        if confidence_value >= 0:

            confidences.append(
                confidence_value
            )

    combined_text = " ".join(
        words
    ).strip()

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
            combined_text,

        "confidence":
            mean_confidence
    }


# ============================================================
# BEST TESSERACT RESULT
# ============================================================

def run_tesseract_best(
    crop,
    language
):

    variants = (
        create_tesseract_variants(
            crop
        )
    )

    best = {
        "text": "",
        "confidence": 0.0,
        "variant": None,
        "score": -999999.0
    }

    all_results = []

    for variant_name, variant_image in variants:

        try:

            result = (
                tesseract_variant_result(
                    variant_image,
                    language
                )
            )

        except Exception:

            continue

        text = (
            result[
                "text"
            ]
        )

        confidence = (
            result[
                "confidence"
            ]
        )

        if language == "eng":

            script_count = (
                count_latin_characters(
                    text
                )
            )

        else:

            script_count = (
                count_sinhala_characters(
                    text
                )
            )

        # Confidence is most important.
        # Text length is a small secondary bonus.
        score = (
            confidence
            + min(
                script_count,
                30
            ) * 0.50
        )

        current = {
            "text":
                text,

            "confidence":
                float(
                    confidence
                ),

            "variant":
                variant_name,

            "score":
                float(
                    score
                )
        }

        all_results.append(
            current
        )

        if score > best["score"]:

            best = current

    best[
        "all_variants"
    ] = all_results

    return best


# ============================================================
# ROUTING SCORE
#
# Tesseract SIN is used only as a language/script probe.
# Final Sinhala output still comes from our trained CRNN.
# ============================================================

def calculate_language_route(
    english_result,
    sinhala_probe
):

    english_text = (
        english_result[
            "text"
        ]
    )

    sinhala_probe_text = (
        sinhala_probe[
            "text"
        ]
    )

    eng_latin = (
        count_latin_characters(
            english_text
        )
    )

    eng_sinhala = (
        count_sinhala_characters(
            english_text
        )
    )

    sin_sinhala = (
        count_sinhala_characters(
            sinhala_probe_text
        )
    )

    sin_latin = (
        count_latin_characters(
            sinhala_probe_text
        )
    )

    eng_total_script = max(
        1,
        eng_latin + eng_sinhala
    )

    sin_total_script = max(
        1,
        sin_sinhala + sin_latin
    )

    english_script_ratio = (
        eng_latin
        / float(
            eng_total_script
        )
    )

    sinhala_script_ratio = (
        sin_sinhala
        / float(
            sin_total_script
        )
    )

    english_route_score = (
        english_result[
            "confidence"
        ]
        + english_script_ratio * 30.0
    )

    sinhala_route_score = (
        sinhala_probe[
            "confidence"
        ]
        + sinhala_script_ratio * 30.0
    )

    # --------------------------------------------------------
    # ENGLISH RULE
    #
    # Require real Latin content and acceptable Tesseract
    # confidence before routing away from our Sinhala model.
    # --------------------------------------------------------

    english_is_strong = (

        eng_latin >= 4

        and english_script_ratio >= 0.65

        and english_result[
            "confidence"
        ] >= 40.0

        and (
            english_route_score
            >= sinhala_route_score + 5.0
            or
            sinhala_probe[
                "confidence"
            ] < 30.0
        )
    )

    if english_is_strong:

        selected_language = "en"

    else:

        selected_language = "si"

    return {

        "selected_language":
            selected_language,

        "english_route_score":
            float(
                english_route_score
            ),

        "sinhala_route_score":
            float(
                sinhala_route_score
            ),

        "english_script_ratio":
            float(
                english_script_ratio
            ),

        "sinhala_script_ratio":
            float(
                sinhala_script_ratio
            ),

        "english_latin_characters":
            int(
                eng_latin
            ),

        "sinhala_probe_characters":
            int(
                sin_sinhala
            )
    }


# ============================================================
# AUTOMATIC LINE EXTRACTION
#
# Reuses working V4.2 geometry.
# ============================================================

def get_automatic_line_crops(
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

    print(
        "\nSTEP 1: Automatic headline detection"
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

    print(
        f"Text-like candidates: "
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

    # --------------------------------------------------------
    # DEBUG MASKS
    # --------------------------------------------------------

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

    line_crops = []

    extraction_method = None

    roi_info = None

    headline_roi = None

    segmentation_mask = None

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
    # GENERIC FALLBACK
    # ========================================================

    if not line_crops:

        print()
        print(
            "Colour detection route failed."
        )

        print(
            "Using generic lower-region fallback..."
        )

        (
            fallback,
            fallback_attempts
        ) = generic_window_fallback(
            image
        )

        if fallback is None:

            raise RuntimeError(
                "No usable text lines were detected."
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
    # SAVE SELECTED ROI
    # ========================================================

    roi_path = (
        debug_dir
        / "selected_headline_roi.png"
    )

    if headline_roi is not None:

        cv2.imwrite(
            str(
                roi_path
            ),
            headline_roi
        )

    mask_path = (
        debug_dir
        / "selected_segmentation_mask.png"
    )

    if segmentation_mask is not None:

        cv2.imwrite(
            str(
                mask_path
            ),
            segmentation_mask
        )

    return {

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
# OCR ONE LINE
# ============================================================

def recognise_line(
    crop,
    crop_path,
    crnn_engine,
    language_mode
):

    # --------------------------------------------------------
    # FORCE SINHALA
    # --------------------------------------------------------

    if language_mode == "si":

        crnn_result = (
            crnn_engine.predict_line(
                crop_path
            )
        )

        sinhala_text = (
            crnn_result[
                "text"
            ]
            .strip()
        )

        return {

            "selected_language":
                "si",

            "selected_text":
                sinhala_text,

            "selected_engine":
                "Custom CRNN V3",

            "sinhala_crnn_text":
                sinhala_text,

            "english_tesseract_text":
                "",

            "english_tesseract_confidence":
                None,

            "sinhala_probe_text":
                "",

            "sinhala_probe_confidence":
                None,

            "routing":
                "forced_si"
        }

    # --------------------------------------------------------
    # FORCE ENGLISH
    # --------------------------------------------------------

    if language_mode == "en":

        english_result = (
            run_tesseract_best(
                crop,
                "eng"
            )
        )

        return {

            "selected_language":
                "en",

            "selected_text":
                english_result[
                    "text"
                ],

            "selected_engine":
                "Tesseract ENG",

            "sinhala_crnn_text":
                "",

            "english_tesseract_text":
                english_result[
                    "text"
                ],

            "english_tesseract_confidence":
                english_result[
                    "confidence"
                ],

            "english_variant":
                english_result[
                    "variant"
                ],

            "sinhala_probe_text":
                "",

            "sinhala_probe_confidence":
                None,

            "routing":
                "forced_en"
        }

    # ========================================================
    # AUTO MODE
    # ========================================================

    english_result = (
        run_tesseract_best(
            crop,
            "eng"
        )
    )

    # Tesseract Sinhala is ONLY a language probe.
    sinhala_probe = (
        run_tesseract_best(
            crop,
            "sin"
        )
    )

    route = calculate_language_route(
        english_result,
        sinhala_probe
    )

    # Custom CRNN output
    crnn_result = (
        crnn_engine.predict_line(
            crop_path
        )
    )

    sinhala_crnn_text = (
        crnn_result[
            "text"
        ]
        .strip()
    )

    if (
        route[
            "selected_language"
        ]
        == "en"
    ):

        selected_text = (
            english_result[
                "text"
            ]
        )

        selected_engine = (
            "Tesseract ENG"
        )

    else:

        selected_text = (
            sinhala_crnn_text
        )

        selected_engine = (
            "Custom CRNN V3"
        )

    return {

        "selected_language":
            route[
                "selected_language"
            ],

        "selected_text":
            selected_text,

        "selected_engine":
            selected_engine,

        "sinhala_crnn_text":
            sinhala_crnn_text,

        "english_tesseract_text":
            english_result[
                "text"
            ],

        "english_tesseract_confidence":
            english_result[
                "confidence"
            ],

        "english_variant":
            english_result[
                "variant"
            ],

        "sinhala_probe_text":
            sinhala_probe[
                "text"
            ],

        "sinhala_probe_confidence":
            sinhala_probe[
                "confidence"
            ],

        "sinhala_probe_variant":
            sinhala_probe[
                "variant"
            ],

        "routing":
            route
    }


# ============================================================
# COMPLETE MULTILINGUAL PIPELINE
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
        "\nLanguage mode:",
        language_mode
    )

    print(
        "\nSinhala OCR model:"
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
    # AUTOMATIC LINE CROPS
    # ========================================================

    extraction = (
        get_automatic_line_crops(
            image,
            debug_dir
        )
    )

    line_crops = (
        extraction[
            "line_crops"
        ]
    )

    print()
    print(
        "Automatic extraction method:"
    )

    print(
        extraction[
            "extraction_method"
        ]
    )

    print(
        f"OCR line crops: "
        f"{len(line_crops)}"
    )

    # ========================================================
    # MULTILINGUAL OCR
    # ========================================================

    print()
    print(
        "STEP 3: Multilingual OCR"
    )

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
            / f"line_{index:02d}.png"
        )

        cv2.imwrite(
            str(
                crop_path
            ),
            crop
        )

        recognition = (
            recognise_line(
                crop,
                crop_path,
                crnn_engine,
                language_mode
            )
        )

        print()
        print("-" * 72)

        print(
            f"LINE {index}"
        )

        print("-" * 72)

        print(
            "Crop shape:",
            crop.shape
        )

        print(
            "Detected route:",
            recognition[
                "selected_language"
            ]
        )

        print(
            "Selected OCR engine:",
            recognition[
                "selected_engine"
            ]
        )

        if (
            language_mode
            == "auto"
        ):

            print(
                "English OCR:"
            )

            print(
                recognition[
                    "english_tesseract_text"
                ]
            )

            print(
                "English confidence:",
                round(
                    recognition[
                        "english_tesseract_confidence"
                    ],
                    2
                )
            )

            print(
                "Sinhala probe OCR:"
            )

            print(
                recognition[
                    "sinhala_probe_text"
                ]
            )

            print(
                "Sinhala probe confidence:",
                round(
                    recognition[
                        "sinhala_probe_confidence"
                    ],
                    2
                )
            )

            print(
                "Custom Sinhala CRNN:"
            )

            print(
                recognition[
                    "sinhala_crnn_text"
                ]
            )

        print(
            "\nFINAL LINE TEXT:"
        )

        print(
            recognition[
                "selected_text"
            ]
        )

        results.append(
            {
                "line_number":
                    index,

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
                    ),

                "selected_language":
                    recognition[
                        "selected_language"
                    ],

                "selected_engine":
                    recognition[
                        "selected_engine"
                    ],

                "selected_text":
                    recognition[
                        "selected_text"
                    ],

                "sinhala_crnn_text":
                    recognition.get(
                        "sinhala_crnn_text",
                        ""
                    ),

                "english_tesseract_text":
                    recognition.get(
                        "english_tesseract_text",
                        ""
                    ),

                "english_tesseract_confidence":
                    recognition.get(
                        "english_tesseract_confidence"
                    ),

                "sinhala_probe_text":
                    recognition.get(
                        "sinhala_probe_text",
                        ""
                    ),

                "sinhala_probe_confidence":
                    recognition.get(
                        "sinhala_probe_confidence"
                    ),

                "routing":
                    recognition.get(
                        "routing"
                    )
            }
        )

    # ========================================================
    # FINAL TEXT
    # ========================================================

    extracted_text = "\n".join(
        result[
            "selected_text"
        ]
        for result in results
        if result[
            "selected_text"
        ]
    )

    final_language = (
        detect_language(
            extracted_text
        )
    )

    selected_languages = [
        result[
            "selected_language"
        ]
        for result in results
    ]

    # If script detection cannot decide,
    # use the OCR routing decisions.
    if final_language == "unknown":

        unique_languages = set(
            selected_languages
        )

        if unique_languages == {"en"}:
            final_language = "en"

        elif unique_languages == {"si"}:
            final_language = "si"

        elif (
            "en" in unique_languages
            and "si" in unique_languages
        ):
            final_language = "si+en"

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

        "final_detected_language":
            final_language,

        "headline_extraction_method":
            extraction[
                "extraction_method"
            ],

        "colour_scan":
            extraction[
                "scan_info"
            ],

        "headline_roi":
            extraction[
                "roi_info"
            ],

        "selected_headline_line_count":
            len(
                extraction[
                    "selected_boxes"
                ]
            ),

        "ocr_line_count":
            len(
                results
            ),

        "sinhala_engine":
            "Custom CRNN-CTC Crop-Jitter V3",

        "sinhala_model":
            str(
                crnn_engine.model_path
            ),

        "english_engine":
            "Tesseract ENG",

        "english_traineddata":
            str(
                ENG_TRAINEDDATA
            ),

        "extracted_text":
            extracted_text,

        "txt_path":
            str(
                txt_path
            ),

        "lines":
            results
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
        "Final Detected Language:",
        final_language
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
            results
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

        "language":
            final_language,

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
            "  auto = automatically choose Sinhala / English"
        )

        print(
            "  si   = force Sinhala custom CRNN"
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
                "Language mode must be "
                "auto, si, or en."
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