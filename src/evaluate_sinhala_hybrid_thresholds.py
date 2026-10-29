from pathlib import Path
import csv
import json

from evaluate_ocr_crnn import (
    levenshtein_distance,
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

V3_TESS_CSV = (
    PROJECT_ROOT
    / "results"
    / "v3_vs_tesseract_sinhala"
    / "v3_vs_tesseract_clean_validation.csv"
)

TESS_CONFIDENCE_CSV = (
    PROJECT_ROOT
    / "results"
    / "tesseract_sinhala_confidence"
    / "tesseract_sinhala_confidence.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "sinhala_hybrid_threshold_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_CSV = (
    OUTPUT_DIR
    / "hybrid_threshold_results.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "hybrid_threshold_summary.json"
)


# ============================================================
# CURRENT V4.6 THRESHOLD
# ============================================================

CURRENT_THRESHOLD = 60.0


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8-sig"
    ) as file:

        return list(
            csv.DictReader(file)
        )


def count_sinhala_characters(text):

    return sum(
        1
        for character in text
        if "\u0D80" <= character <= "\u0DFF"
    )


# ============================================================
# METRICS
# ============================================================

def calculate_dataset_metrics(
    records,
    threshold
):

    total_char_errors = 0
    total_ref_chars = 0

    total_word_errors = 0
    total_ref_words = 0

    exact_matches = 0

    tesseract_selected = 0
    crnn_selected = 0

    line_results = []

    for record in records:

        sample_id = record[
            "sample_id"
        ]

        ground_truth = (
            record[
                "ground_truth"
            ].strip()
        )

        v3_text = (
            record[
                "v3_prediction"
            ].strip()
        )

        tess_text = (
            record[
                "tesseract_prediction"
            ].strip()
        )

        tess_confidence = float(
            record[
                "tesseract_confidence"
            ]
        )

        tess_sinhala_chars = (
            count_sinhala_characters(
                tess_text
            )
        )

        # ====================================================
        # SAME BASIC RULE AS V4.6
        # ====================================================

        use_tesseract = (

            bool(
                tess_text
            )

            and

            tess_confidence >= threshold

            and

            tess_sinhala_chars > 0
        )

        if use_tesseract:

            selected_text = (
                tess_text
            )

            selected_engine = (
                "Tesseract"
            )

            tesseract_selected += 1

        elif v3_text:

            selected_text = (
                v3_text
            )

            selected_engine = (
                "V3 CRNN"
            )

            crnn_selected += 1

        elif tess_text:

            selected_text = (
                tess_text
            )

            selected_engine = (
                "Tesseract fallback"
            )

            tesseract_selected += 1

        else:

            selected_text = ""

            selected_engine = (
                "Blank"
            )

        # ====================================================
        # CER
        # ====================================================

        char_errors = (
            levenshtein_distance(
                list(
                    ground_truth
                ),
                list(
                    selected_text
                )
            )
        )

        total_char_errors += (
            char_errors
        )

        total_ref_chars += len(
            ground_truth
        )

        # ====================================================
        # WER
        # ====================================================

        ground_truth_words = (
            ground_truth.split()
        )

        selected_words = (
            selected_text.split()
        )

        word_errors = (
            levenshtein_distance(
                ground_truth_words,
                selected_words
            )
        )

        total_word_errors += (
            word_errors
        )

        total_ref_words += len(
            ground_truth_words
        )

        # ====================================================
        # EXACT
        # ====================================================

        exact = (
            selected_text
            == ground_truth
        )

        if exact:

            exact_matches += 1

        line_results.append(
            {
                "sample_id":
                    sample_id,

                "selected_engine":
                    selected_engine,

                "tesseract_confidence":
                    tess_confidence,

                "ground_truth":
                    ground_truth,

                "selected_text":
                    selected_text,

                "character_errors":
                    char_errors,

                "word_errors":
                    word_errors,

                "exact":
                    exact,
            }
        )

    cer = (
        total_char_errors
        / total_ref_chars
        if total_ref_chars > 0
        else 0.0
    )

    wer = (
        total_word_errors
        / total_ref_words
        if total_ref_words > 0
        else 0.0
    )

    exact_accuracy = (
        exact_matches
        / len(records)
        if records
        else 0.0
    )

    return {
        "threshold":
            float(
                threshold
            ),

        "cer":
            cer,

        "wer":
            wer,

        "exact_matches":
            exact_matches,

        "exact_accuracy":
            exact_accuracy,

        "tesseract_selected":
            tesseract_selected,

        "crnn_selected":
            crnn_selected,

        "line_results":
            line_results,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("SINHALA HYBRID THRESHOLD EVALUATION")
    print("=" * 72)

    comparison_records = (
        load_csv(
            V3_TESS_CSV
        )
    )

    confidence_records = (
        load_csv(
            TESS_CONFIDENCE_CSV
        )
    )

    confidence_by_id = {

        row[
            "sample_id"
        ]:
            row

        for row in confidence_records
    }

    combined_records = []

    for row in comparison_records:

        sample_id = (
            row[
                "sample_id"
            ]
        )

        if sample_id not in confidence_by_id:

            raise RuntimeError(
                "Missing confidence for sample: "
                f"{sample_id}"
            )

        confidence_row = (
            confidence_by_id[
                sample_id
            ]
        )

        combined_records.append(
            {
                "sample_id":
                    sample_id,

                "ground_truth":
                    row[
                        "ground_truth"
                    ],

                "v3_prediction":
                    row[
                        "v3_prediction"
                    ],

                "tesseract_prediction":
                    row[
                        "tesseract_prediction"
                    ],

                "tesseract_confidence":
                    confidence_row[
                        "tesseract_confidence"
                    ],
            }
        )

    print()
    print(
        "Validation samples:",
        len(
            combined_records
        )
    )

    # ========================================================
    # TEST EVERY INTEGER THRESHOLD
    # ========================================================

    threshold_results = []

    for threshold in range(
        0,
        101
    ):

        result = (
            calculate_dataset_metrics(
                combined_records,
                float(
                    threshold
                )
            )
        )

        threshold_results.append(
            result
        )

    # ========================================================
    # CURRENT V4.6 RESULT
    # ========================================================

    current_result = (
        calculate_dataset_metrics(
            combined_records,
            CURRENT_THRESHOLD
        )
    )

    print()
    print("-" * 72)
    print("CURRENT V4.6 RULE")
    print("-" * 72)

    print(
        "Threshold:",
        CURRENT_THRESHOLD
    )

    print(
        "CER:",
        f"{current_result['cer']:.4f}"
    )

    print(
        "WER:",
        f"{current_result['wer']:.4f}"
    )

    print(
        "Exact:",
        f"{current_result['exact_matches']}/"
        f"{len(combined_records)}"
    )

    print(
        "Tesseract selected:",
        current_result[
            "tesseract_selected"
        ]
    )

    print(
        "CRNN selected:",
        current_result[
            "crnn_selected"
        ]
    )

    # ========================================================
    # BEST CER THRESHOLD
    # ========================================================

    best_cer_result = min(
        threshold_results,
        key=lambda item: (
            item[
                "cer"
            ],
            item[
                "wer"
            ]
        )
    )

    # ========================================================
    # BEST WER THRESHOLD
    # ========================================================

    best_wer_result = min(
        threshold_results,
        key=lambda item: (
            item[
                "wer"
            ],
            item[
                "cer"
            ]
        )
    )

    print()
    print("-" * 72)
    print("BEST DEVELOPMENT THRESHOLDS")
    print("-" * 72)

    print()
    print(
        "Best CER threshold:",
        best_cer_result[
            "threshold"
        ]
    )

    print(
        "CER:",
        f"{best_cer_result['cer']:.4f}"
    )

    print(
        "WER:",
        f"{best_cer_result['wer']:.4f}"
    )

    print(
        "Tesseract selected:",
        best_cer_result[
            "tesseract_selected"
        ]
    )

    print(
        "CRNN selected:",
        best_cer_result[
            "crnn_selected"
        ]
    )

    print()
    print(
        "Best WER threshold:",
        best_wer_result[
            "threshold"
        ]
    )

    print(
        "CER:",
        f"{best_wer_result['cer']:.4f}"
    )

    print(
        "WER:",
        f"{best_wer_result['wer']:.4f}"
    )

    print(
        "Tesseract selected:",
        best_wer_result[
            "tesseract_selected"
        ]
    )

    print(
        "CRNN selected:",
        best_wer_result[
            "crnn_selected"
        ]
    )

    # ========================================================
    # PRINT USEFUL THRESHOLDS
    # ========================================================

    print()
    print("-" * 72)
    print("SELECTED THRESHOLD TABLE")
    print("-" * 72)

    useful_thresholds = [
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
    ]

    print()
    print(
        "Threshold | CER    | WER    | TESS | CRNN"
    )

    print(
        "-" * 48
    )

    rows_to_save = []

    for threshold in useful_thresholds:

        result = (
            threshold_results[
                threshold
            ]
        )

        print(
            f"{threshold:9d} | "
            f"{result['cer']:.4f} | "
            f"{result['wer']:.4f} | "
            f"{result['tesseract_selected']:4d} | "
            f"{result['crnn_selected']:4d}"
        )

    # ========================================================
    # SAVE ALL THRESHOLDS
    # ========================================================

    for result in threshold_results:

        rows_to_save.append(
            {
                "threshold":
                    result[
                        "threshold"
                    ],

                "cer":
                    result[
                        "cer"
                    ],

                "wer":
                    result[
                        "wer"
                    ],

                "exact_matches":
                    result[
                        "exact_matches"
                    ],

                "exact_accuracy":
                    result[
                        "exact_accuracy"
                    ],

                "tesseract_selected":
                    result[
                        "tesseract_selected"
                    ],

                "crnn_selected":
                    result[
                        "crnn_selected"
                    ],
            }
        )

    with open(
        RESULTS_CSV,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "threshold",
                "cer",
                "wer",
                "exact_matches",
                "exact_accuracy",
                "tesseract_selected",
                "crnn_selected",
            ]
        )

        writer.writeheader()

        writer.writerows(
            rows_to_save
        )

    summary = {

        "validation_samples":
            len(
                combined_records
            ),

        "current_v4_6":
            {
                key: value
                for key, value
                in current_result.items()
                if key != "line_results"
            },

        "best_cer_threshold":
            {
                key: value
                for key, value
                in best_cer_result.items()
                if key != "line_results"
            },

        "best_wer_threshold":
            {
                key: value
                for key, value
                in best_wer_result.items()
                if key != "line_results"
            },

        "important_note":
            (
                "Thresholds were selected on the "
                "development validation set and must "
                "not be reported as final unseen-test "
                "performance."
            ),
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4
        )

    print()
    print("=" * 72)
    print(
        "IMPORTANT:"
    )

    print(
        "These thresholds are DEVELOPMENT parameters."
    )

    print(
        "Do not call the best threshold final until "
        "fresh unseen images are tested."
    )

    print()
    print(
        "Results CSV:"
    )

    print(
        RESULTS_CSV
    )

    print()
    print(
        "Summary JSON:"
    )

    print(
        SUMMARY_JSON
    )

    print("=" * 72)


if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print(
            "ERROR:"
        )
        print(
            error
        )