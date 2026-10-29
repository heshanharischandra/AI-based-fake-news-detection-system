from pathlib import Path
from datetime import datetime

import json

import cv2


# ============================================================
# REUSE STABLE V4.5 PIPELINE COMPONENTS
# ============================================================

from image_ocr_v4_hybrid import (
    PROJECT_ROOT,
    load_image,
)

from image_ocr_v4_5_multilingual import (
    configure_tesseract,
    english_first_probe,
    extract_sinhala_headline,
    create_english_results,
    detect_final_language,
    run_tesseract,
)

from ocr_v3_inference import (
    get_ocr_v2_engine,
)


# ============================================================
# VERSION
# ============================================================

PIPELINE_VERSION = (
    "Hybrid Multilingual Image OCR V4.6"
)


# ============================================================
# HYBRID SINHALA SETTINGS
#
# Development-validation rule:
#
# Tesseract non-blank + confidence >= 60
#     -> Tesseract
#
# Otherwise
#     -> Custom CRNN V3
#
# IMPORTANT:
# 60 is a DEVELOPMENT threshold.
# It must later be checked on fresh unseen test images.
# ============================================================

SINHALA_TESSERACT_THRESHOLD = 60.0

SINHALA_TESSERACT_PSM = 7


# ============================================================
# OUTPUT PATHS
# ============================================================

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v4_6_hybrid_multilingual"
)

DEBUG_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_ocr_v4_6_hybrid_multilingual_debug"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEBUG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SIMPLE TEXT NORMALIZATION
# ============================================================

def normalize_candidate_text(text):

    if text is None:

        return ""

    return str(
        text
    ).strip()


# ============================================================
# SINHALA HYBRID OCR
#
# Each detected Sinhala line is sent to:
#
#   1. Custom CRNN V3
#   2. Tesseract Sinhala PSM 7
#
# Selection:
#
#   high-confidence Tesseract -> Tesseract
#   otherwise                 -> CRNN V3
#
# Extra safety:
#
#   If CRNN returns blank but Tesseract has text,
#   Tesseract is used as a fallback even below threshold.
# ============================================================

def run_sinhala_hybrid(
    line_crops,
    crnn_engine,
    debug_dir,
):

    results = []

    tesseract_selected_count = 0

    crnn_selected_count = 0

    blank_count = 0

    for index, item in enumerate(
        line_crops,
        start=1,
    ):

        crop = item[
            "crop"
        ]

        # ----------------------------------------------------
        # SAVE EXACT LINE CROP
        # ----------------------------------------------------

        crop_path = (
            debug_dir
            / f"sinhala_hybrid_line_{index:02d}.png"
        )

        saved = cv2.imwrite(
            str(
                crop_path
            ),
            crop,
        )

        if not saved:

            raise RuntimeError(
                f"Could not save Sinhala line crop:\n"
                f"{crop_path}"
            )

        # ====================================================
        # ENGINE 1: CUSTOM CRNN V3
        # ====================================================

        crnn_prediction = (
            crnn_engine.predict_line(
                crop_path
            )
        )

        crnn_text = (
            normalize_candidate_text(
                crnn_prediction.get(
                    "text",
                    "",
                )
            )
        )

        # ====================================================
        # ENGINE 2: TESSERACT SINHALA
        # ====================================================

        tesseract_result = (
            run_tesseract(
                crop,
                "sin",
                psm=SINHALA_TESSERACT_PSM,
            )
        )

        tesseract_text = (
            normalize_candidate_text(
                tesseract_result.get(
                    "text",
                    "",
                )
            )
        )

        tesseract_confidence = float(
            tesseract_result.get(
                "confidence",
                0.0,
            )
        )

        tesseract_sinhala_chars = int(
            tesseract_result.get(
                "sinhala_characters",
                0,
            )
        )

        # ====================================================
        # HYBRID DECISION
        # ====================================================

        tesseract_high_confidence = (

            bool(
                tesseract_text
            )

            and

            tesseract_confidence
            >= SINHALA_TESSERACT_THRESHOLD

            and

            tesseract_sinhala_chars > 0
        )

        if tesseract_high_confidence:

            selected_text = (
                tesseract_text
            )

            selected_engine = (
                "Tesseract SIN PSM 7"
            )

            selection_reason = (
                "tesseract_nonblank_and_confidence_"
                f">={SINHALA_TESSERACT_THRESHOLD:.0f}"
            )

            tesseract_selected_count += 1

        elif crnn_text:

            selected_text = (
                crnn_text
            )

            selected_engine = (
                "Custom CRNN V3"
            )

            if not tesseract_text:

                selection_reason = (
                    "tesseract_blank_fallback_to_crnn"
                )

            else:

                selection_reason = (
                    "tesseract_confidence_below_threshold_"
                    "fallback_to_crnn"
                )

            crnn_selected_count += 1

        elif tesseract_text:

            # ------------------------------------------------
            # Emergency fallback:
            #
            # CRNN has nothing, but Tesseract has some text.
            # Better to preserve the OCR candidate than return
            # an empty line.
            # ------------------------------------------------

            selected_text = (
                tesseract_text
            )

            selected_engine = (
                "Tesseract SIN PSM 7"
            )

            selection_reason = (
                "crnn_blank_tesseract_nonblank_"
                "low_confidence_fallback"
            )

            tesseract_selected_count += 1

        else:

            selected_text = ""

            selected_engine = (
                "No OCR Candidate"
            )

            selection_reason = (
                "both_engines_blank"
            )

            blank_count += 1

        # ====================================================
        # CONSOLE OUTPUT
        # ====================================================

        print()
        print(
            "-" * 72
        )

        print(
            f"SINHALA HYBRID LINE {index}"
        )

        print(
            "-" * 72
        )

        print(
            "Crop shape:",
            crop.shape,
        )

        print()
        print(
            "CRNN V3:"
        )

        print(
            crnn_text
            if crnn_text
            else "[BLANK]"
        )

        print()
        print(
            "Tesseract Sinhala:"
        )

        print(
            tesseract_text
            if tesseract_text
            else "[BLANK]"
        )

        print(
            "Tesseract confidence:",
            f"{tesseract_confidence:.2f}",
        )

        print(
            "Tesseract Sinhala chars:",
            tesseract_sinhala_chars,
        )

        print()
        print(
            "SELECTED ENGINE:",
            selected_engine,
        )

        print(
            "SELECTION REASON:",
            selection_reason,
        )

        print()
        print(
            "FINAL LINE:"
        )

        print(
            selected_text
            if selected_text
            else "[BLANK]"
        )

        # ====================================================
        # SAVE LINE RESULT
        # ====================================================

        results.append(
            {
                "line_number":
                    index,

                "language":
                    "si",

                "engine":
                    selected_engine,

                "selection_reason":
                    selection_reason,

                "text":
                    selected_text,

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
                        "unknown",
                    ),

                # --------------------------------------------
                # CUSTOM CRNN CANDIDATE
                # --------------------------------------------

                "crnn_candidate":
                    {
                        "engine":
                            "Custom CRNN V3",

                        "text":
                            crnn_text,

                        "checkpoint_epoch":
                            crnn_prediction.get(
                                "checkpoint_epoch"
                            ),

                        "validation_cer":
                            crnn_prediction.get(
                                "validation_cer"
                            ),

                        "validation_wer":
                            crnn_prediction.get(
                                "validation_wer"
                            ),
                    },

                # --------------------------------------------
                # TESSERACT CANDIDATE
                # --------------------------------------------

                "tesseract_candidate":
                    {
                        "engine":
                            "Tesseract SIN PSM 7",

                        "text":
                            tesseract_text,

                        "raw_text":
                            tesseract_result.get(
                                "raw_text",
                                "",
                            ),

                        "confidence":
                            tesseract_confidence,

                        "word_count":
                            int(
                                tesseract_result.get(
                                    "word_count",
                                    0,
                                )
                            ),

                        "sinhala_characters":
                            tesseract_sinhala_chars,

                        "english_characters":
                            int(
                                tesseract_result.get(
                                    "english_characters",
                                    0,
                                )
                            ),
                    },
            }
        )

    # ========================================================
    # COMBINE SELECTED LINES
    # ========================================================

    combined_text = "\n".join(
        result[
            "text"
        ]
        for result in results
        if result[
            "text"
        ]
    )

    summary = {

        "threshold":
            float(
                SINHALA_TESSERACT_THRESHOLD
            ),

        "tesseract_psm":
            SINHALA_TESSERACT_PSM,

        "total_lines":
            len(
                results
            ),

        "tesseract_selected":
            tesseract_selected_count,

        "crnn_selected":
            crnn_selected_count,

        "both_blank":
            blank_count,
    }

    return (
        combined_text,
        results,
        summary,
    )


# ============================================================
# COMPLETE V4.6 MULTILINGUAL PIPELINE
# ============================================================

def run_multilingual_ocr(
    image_path,
    language_mode="auto",
):

    # ========================================================
    # CONFIGURE TESSERACT
    # ========================================================

    configure_tesseract()

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = load_image(
        image_path
    )

    image_height, image_width = (
        image.shape[:2]
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )

    debug_dir = (
        DEBUG_DIR
        / f"run_{timestamp}"
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 72
    )

    print(
        PIPELINE_VERSION.upper()
    )

    print(
        "=" * 72
    )

    print()
    print(
        "Input image:"
    )

    print(
        image_path
    )

    print()
    print(
        f"Input image size: "
        f"{image_width} x {image_height}"
    )

    print()
    print(
        "Requested language mode:",
        language_mode,
    )

    # ========================================================
    # STEP 1
    # LANGUAGE ROUTING
    # ========================================================

    print()
    print(
        "STEP 1: English-first language probe"
    )

    probe_info = None

    # --------------------------------------------------------
    # FORCED SINHALA
    # --------------------------------------------------------

    if language_mode == "si":

        selected_language = "si"

        print(
            "Language forced to Sinhala."
        )

    # --------------------------------------------------------
    # AUTO / ENGLISH
    # --------------------------------------------------------

    else:

        probe_info = (
            english_first_probe(
                image,
                debug_dir,
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

        print()
        print(
            "English confidence:",
            round(
                probe_info[
                    "english"
                ][
                    "confidence"
                ],
                2,
            ),
        )

        print(
            "English characters:",
            probe_info[
                "english"
            ][
                "english_characters"
            ],
        )

        print(
            "English words:",
            probe_info[
                "english"
            ][
                "word_count"
            ],
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

        print()
        print(
            "Sinhala probe confidence:",
            round(
                probe_info[
                    "sinhala_probe"
                ][
                    "confidence"
                ],
                2,
            ),
        )

        print(
            "Sinhala characters:",
            probe_info[
                "sinhala_probe"
            ][
                "sinhala_characters"
            ],
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
                selected_language,
            )

    # ========================================================
    # VARIABLES USED BY BOTH BRANCHES
    # ========================================================

    sinhala_extraction = None

    sinhala_model_path = None

    sinhala_hybrid_summary = None

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
                ],
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
                2,
            ),
        )

    # ========================================================
    # SINHALA HYBRID BRANCH
    # ========================================================

    else:

        print()
        print(
            "STEP 2: Sinhala HYBRID OCR branch"
        )

        # ----------------------------------------------------
        # SAME V4.5 HEADLINE / LINE EXTRACTION
        # ----------------------------------------------------

        sinhala_extraction = (
            extract_sinhala_headline(
                image,
                debug_dir,
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
            ),
        )

        # ----------------------------------------------------
        # LOAD EXISTING V3 MODEL
        # ----------------------------------------------------

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

        print()
        print(
            "Hybrid Tesseract threshold:",
            SINHALA_TESSERACT_THRESHOLD,
        )

        # ----------------------------------------------------
        # RUN BOTH ENGINES ON EVERY LINE
        # ----------------------------------------------------

        (
            extracted_text,
            ocr_results,
            sinhala_hybrid_summary,
        ) = run_sinhala_hybrid(
            line_crops,
            crnn_engine,
            debug_dir,
        )

        selected_engine = (
            "Hybrid Sinhala OCR "
            "(Tesseract SIN PSM 7 + Custom CRNN V3)"
        )

        extraction_method = (
            sinhala_extraction[
                "extraction_method"
            ]
        )

        print()
        print(
            "-" * 72
        )

        print(
            "SINHALA HYBRID SUMMARY"
        )

        print(
            "-" * 72
        )

        print(
            "Total lines:",
            sinhala_hybrid_summary[
                "total_lines"
            ],
        )

        print(
            "Tesseract selected:",
            sinhala_hybrid_summary[
                "tesseract_selected"
            ],
        )

        print(
            "CRNN selected:",
            sinhala_hybrid_summary[
                "crnn_selected"
            ],
        )

        print(
            "Both blank:",
            sinhala_hybrid_summary[
                "both_blank"
            ],
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
        encoding="utf-8",
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
                    ),
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

        # ====================================================
        # LANGUAGE ROUTING INFORMATION
        # ====================================================

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
                    ],
            },

        # ====================================================
        # SINHALA DETECTION INFORMATION
        # ====================================================

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
                    ),
            },

        # ====================================================
        # HYBRID INFORMATION
        # ====================================================

        "sinhala_hybrid":
            sinhala_hybrid_summary,

        "sinhala_tesseract_threshold":
            float(
                SINHALA_TESSERACT_THRESHOLD
            ),

        "sinhala_tesseract_psm":
            SINHALA_TESSERACT_PSM,

        # ====================================================
        # OCR OUTPUT
        # ====================================================

        "ocr_line_count":
            len(
                ocr_results
            ),

        "extracted_text":
            extracted_text,

        "ocr_results":
            ocr_results,

        # ====================================================
        # ENGINES
        # ====================================================

        "sinhala_model":
            sinhala_model_path,

        "sinhala_crnn_engine":
            "Custom CRNN V3",

        "sinhala_tesseract_engine":
            "Tesseract SIN PSM 7",

        "english_engine":
            "Tesseract ENG PSM 6",

        # ====================================================
        # FILES
        # ====================================================

        "txt_path":
            str(
                txt_path
            ),

        "debug_dir":
            str(
                debug_dir
            ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_data,
            file,
            ensure_ascii=False,
            indent=4,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "FINAL EXTRACTED TEXT"
    )

    print(
        "=" * 72
    )

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
        selected_language,
    )

    print(
        "Final Detected Language:",
        final_language,
    )

    print(
        "OCR Engine:",
        selected_engine,
    )

    print(
        "Extraction Method:",
        extraction_method,
    )

    print(
        "OCR Lines:",
        len(
            ocr_results
        ),
    )

    if (
        sinhala_hybrid_summary
        is not None
    ):

        print()
        print(
            "Hybrid threshold:",
            SINHALA_TESSERACT_THRESHOLD,
        )

        print(
            "Tesseract-selected lines:",
            sinhala_hybrid_summary[
                "tesseract_selected"
            ],
        )

        print(
            "CRNN-selected lines:",
            sinhala_hybrid_summary[
                "crnn_selected"
            ],
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

    print(
        "=" * 72
    )

    return {

        "text":
            extracted_text,

        "selected_language":
            selected_language,

        "final_language":
            final_language,

        "engine":
            selected_engine,

        "extraction_method":
            extraction_method,

        "ocr_results":
            ocr_results,

        "sinhala_hybrid":
            sinhala_hybrid_summary,

        "txt_path":
            txt_path,

        "json_path":
            json_path,

        "debug_dir":
            debug_dir,
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
            "  si   = force Sinhala Hybrid OCR"
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
            "en",
        }:

            raise ValueError(
                "Mode must be auto, si, or en."
            )

        run_multilingual_ocr(
            image_path,
            language_mode,
        )

    except Exception as error:

        print()
        print(
            "ERROR:"
        )

        print(
            error
        )