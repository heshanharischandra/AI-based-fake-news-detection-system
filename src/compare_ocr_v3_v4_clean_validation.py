from pathlib import Path
import csv
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ocr_dataset import (
    OCRDataset,
    ocr_collate_fn,
)

from ocr_crnn_model import CRNN

from evaluate_ocr_crnn import (
    greedy_ctc_decode,
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

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

VALIDATION_CSV = (
    OCR_ROOT
    / "cleaned_real_ocr_v1"
    / "metadata"
    / "splits"
    / "clean_real_validation_v1.csv"
)

CHARSET_JSON = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)


# ============================================================
# MODELS
# ============================================================

V3_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_crop_jitter_v3"
    / "last_crop_jitter_v3_model.pt"
)

V4_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_clean_v4"
    / "best_clean_v4_model.pt"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "ocr_v3_v4_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_CSV = (
    OUTPUT_DIR
    / "v3_v4_clean_validation_predictions.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "v3_v4_clean_validation_summary.json"
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 4

NUM_WORKERS = 0


# ============================================================
# LOAD CHARSET
# ============================================================

def load_charset():

    with open(
        CHARSET_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )

    index_to_char = {

        int(index): character

        for index, character
        in data[
            "index_to_char"
        ].items()
    }

    blank_index = int(
        data[
            "ctc_blank_index"
        ]
    )

    return (
        index_to_char,
        blank_index
    )


# ============================================================
# LOAD VALIDATION METADATA
# ============================================================

def load_validation_records():

    with open(
        VALIDATION_CSV,
        "r",
        encoding="utf-8-sig"
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_path,
    device
):

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    num_classes = int(
        checkpoint[
            "num_classes"
        ]
    )

    hidden_size = int(
        checkpoint.get(
            "hidden_size",
            256
        )
    )

    model = CRNN(
        num_classes=num_classes,
        hidden_size=hidden_size
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    return (
        model,
        checkpoint
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    loader,
    records,
    device,
    index_to_char,
    blank_index
):

    results = []

    total_character_errors = 0
    total_reference_characters = 0

    total_word_errors = 0
    total_reference_words = 0

    exact_matches = 0

    record_index = 0

    with torch.no_grad():

        for batch in loader:

            images = (
                batch[
                    "images"
                ]
                .to(
                    device,
                    non_blocking=True
                )
            )

            references = batch[
                "texts"
            ]

            log_probs = model(
                images
            )

            predictions = (
                greedy_ctc_decode(
                    log_probs,
                    index_to_char,
                    blank_index
                )
            )

            for reference, prediction in zip(
                references,
                predictions
            ):

                metadata = records[
                    record_index
                ]

                record_index += 1

                character_errors = (
                    levenshtein_distance(
                        list(
                            reference
                        ),
                        list(
                            prediction
                        )
                    )
                )

                reference_character_count = len(
                    reference
                )

                reference_words = (
                    reference.split()
                )

                prediction_words = (
                    prediction.split()
                )

                word_errors = (
                    levenshtein_distance(
                        reference_words,
                        prediction_words
                    )
                )

                reference_word_count = len(
                    reference_words
                )

                sample_cer = (

                    character_errors
                    / reference_character_count

                    if reference_character_count > 0

                    else 0.0
                )

                sample_wer = (

                    word_errors
                    / reference_word_count

                    if reference_word_count > 0

                    else 0.0
                )

                exact = (
                    prediction
                    == reference
                )

                if exact:

                    exact_matches += 1

                total_character_errors += (
                    character_errors
                )

                total_reference_characters += (
                    reference_character_count
                )

                total_word_errors += (
                    word_errors
                )

                total_reference_words += (
                    reference_word_count
                )

                results.append(
                    {
                        "sample_id":
                            metadata.get(
                                "sample_id",
                                ""
                            ),

                        "source_id":
                            metadata.get(
                                "source_id",
                                ""
                            ),

                        "ground_truth":
                            reference,

                        "prediction":
                            prediction,

                        "cer":
                            sample_cer,

                        "wer":
                            sample_wer,

                        "exact":
                            exact,
                    }
                )

    overall_cer = (

        total_character_errors
        / total_reference_characters

        if total_reference_characters > 0

        else 0.0
    )

    overall_wer = (

        total_word_errors
        / total_reference_words

        if total_reference_words > 0

        else 0.0
    )

    exact_accuracy = (

        exact_matches
        / len(
            results
        )

        if results

        else 0.0
    )

    return {

        "cer":
            overall_cer,

        "wer":
            overall_wer,

        "exact_matches":
            exact_matches,

        "exact_accuracy":
            exact_accuracy,

        "sample_count":
            len(
                results
            ),

        "predictions":
            results,
    }


# ============================================================
# PER-SOURCE METRICS
# ============================================================

def calculate_source_metrics(
    results
):

    sources = {}

    for result in results:

        source_id = result[
            "source_id"
        ]

        if source_id not in sources:

            sources[
                source_id
            ] = {
                "character_errors": 0,
                "reference_characters": 0,
                "word_errors": 0,
                "reference_words": 0,
                "exact_matches": 0,
                "samples": 0,
            }

        source = sources[
            source_id
        ]

        reference = result[
            "ground_truth"
        ]

        prediction = result[
            "prediction"
        ]

        char_errors = (
            levenshtein_distance(
                list(
                    reference
                ),
                list(
                    prediction
                )
            )
        )

        reference_words = (
            reference.split()
        )

        prediction_words = (
            prediction.split()
        )

        word_errors = (
            levenshtein_distance(
                reference_words,
                prediction_words
            )
        )

        source[
            "character_errors"
        ] += char_errors

        source[
            "reference_characters"
        ] += len(
            reference
        )

        source[
            "word_errors"
        ] += word_errors

        source[
            "reference_words"
        ] += len(
            reference_words
        )

        source[
            "samples"
        ] += 1

        if reference == prediction:

            source[
                "exact_matches"
            ] += 1

    final = {}

    for source_id, values in sources.items():

        final[
            source_id
        ] = {

            "samples":
                values[
                    "samples"
                ],

            "cer":
                (
                    values[
                        "character_errors"
                    ]
                    / values[
                        "reference_characters"
                    ]

                    if values[
                        "reference_characters"
                    ] > 0

                    else 0.0
                ),

            "wer":
                (
                    values[
                        "word_errors"
                    ]
                    / values[
                        "reference_words"
                    ]

                    if values[
                        "reference_words"
                    ] > 0

                    else 0.0
                ),

            "exact_matches":
                values[
                    "exact_matches"
                ],
        }

    return final


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("FAIR OCR V3 vs V4 COMPARISON")
    print("=" * 72)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    records = (
        load_validation_records()
    )

    print()
    print(
        "Clean validation samples:",
        len(
            records
        )
    )

    print(
        "Validation CSV:"
    )

    print(
        VALIDATION_CSV
    )

    dataset = OCRDataset(
        VALIDATION_CSV
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=ocr_collate_fn
    )

    (
        index_to_char,
        blank_index
    ) = load_charset()

    # ========================================================
    # V3
    # ========================================================

    print()
    print("-" * 72)
    print("EVALUATING V3")
    print("-" * 72)

    print(
        V3_MODEL_PATH
    )

    (
        v3_model,
        v3_checkpoint
    ) = load_model(
        V3_MODEL_PATH,
        device
    )

    v3_result = evaluate_model(
        v3_model,
        loader,
        records,
        device,
        index_to_char,
        blank_index
    )

    print()
    print(
        "V3 CER:",
        f"{v3_result['cer']:.4f}"
    )

    print(
        "V3 WER:",
        f"{v3_result['wer']:.4f}"
    )

    print(
        "V3 Exact:",
        f"{v3_result['exact_matches']}/"
        f"{v3_result['sample_count']}"
    )

    del v3_model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    # ========================================================
    # V4
    # ========================================================

    print()
    print("-" * 72)
    print("EVALUATING CLEAN V4")
    print("-" * 72)

    print(
        V4_MODEL_PATH
    )

    (
        v4_model,
        v4_checkpoint
    ) = load_model(
        V4_MODEL_PATH,
        device
    )

    v4_result = evaluate_model(
        v4_model,
        loader,
        records,
        device,
        index_to_char,
        blank_index
    )

    print()
    print(
        "V4 CER:",
        f"{v4_result['cer']:.4f}"
    )

    print(
        "V4 WER:",
        f"{v4_result['wer']:.4f}"
    )

    print(
        "V4 Exact:",
        f"{v4_result['exact_matches']}/"
        f"{v4_result['sample_count']}"
    )

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    cer_change = (
        v3_result[
            "cer"
        ]
        - v4_result[
            "cer"
        ]
    )

    wer_change = (
        v3_result[
            "wer"
        ]
        - v4_result[
            "wer"
        ]
    )

    cer_relative_improvement = (

        cer_change
        / v3_result[
            "cer"
        ]
        * 100.0

        if v3_result[
            "cer"
        ] > 0

        else 0.0
    )

    wer_relative_improvement = (

        wer_change
        / v3_result[
            "wer"
        ]
        * 100.0

        if v3_result[
            "wer"
        ] > 0

        else 0.0
    )

    print()
    print("=" * 72)
    print("FAIR COMPARISON RESULT")
    print("=" * 72)

    print()
    print(
        "Same validation samples:",
        len(
            records
        )
    )

    print()
    print(
        "V3 CER:",
        f"{v3_result['cer']:.4f}"
    )

    print(
        "V4 CER:",
        f"{v4_result['cer']:.4f}"
    )

    print(
        "CER absolute improvement:",
        f"{cer_change:.4f}"
    )

    print(
        "CER relative improvement:",
        f"{cer_relative_improvement:.2f}%"
    )

    print()
    print(
        "V3 WER:",
        f"{v3_result['wer']:.4f}"
    )

    print(
        "V4 WER:",
        f"{v4_result['wer']:.4f}"
    )

    print(
        "WER absolute improvement:",
        f"{wer_change:.4f}"
    )

    print(
        "WER relative improvement:",
        f"{wer_relative_improvement:.2f}%"
    )

    print()
    print(
        "V3 exact:",
        f"{v3_result['exact_matches']}/"
        f"{v3_result['sample_count']}"
    )

    print(
        "V4 exact:",
        f"{v4_result['exact_matches']}/"
        f"{v4_result['sample_count']}"
    )

    # ========================================================
    # PER-SOURCE
    # ========================================================

    v3_sources = (
        calculate_source_metrics(
            v3_result[
                "predictions"
            ]
        )
    )

    v4_sources = (
        calculate_source_metrics(
            v4_result[
                "predictions"
            ]
        )
    )

    print()
    print("-" * 72)
    print("PER-SOURCE CER")
    print("-" * 72)

    for source_id in sorted(
        v4_sources.keys()
    ):

        v3_cer = (
            v3_sources[
                source_id
            ][
                "cer"
            ]
        )

        v4_cer = (
            v4_sources[
                source_id
            ][
                "cer"
            ]
        )

        print(
            f"{source_id}: "
            f"V3={v3_cer:.4f} "
            f"| V4={v4_cer:.4f} "
            f"| change="
            f"{v3_cer - v4_cer:+.4f}"
        )

    # ========================================================
    # SAVE PREDICTIONS CSV
    # ========================================================

    combined_rows = []

    for (
        v3_prediction,
        v4_prediction
    ) in zip(
        v3_result[
            "predictions"
        ],
        v4_result[
            "predictions"
        ]
    ):

        combined_rows.append(
            {

                "sample_id":
                    v3_prediction[
                        "sample_id"
                    ],

                "source_id":
                    v3_prediction[
                        "source_id"
                    ],

                "ground_truth":
                    v3_prediction[
                        "ground_truth"
                    ],

                "v3_prediction":
                    v3_prediction[
                        "prediction"
                    ],

                "v3_cer":
                    v3_prediction[
                        "cer"
                    ],

                "v3_wer":
                    v3_prediction[
                        "wer"
                    ],

                "v4_prediction":
                    v4_prediction[
                        "prediction"
                    ],

                "v4_cer":
                    v4_prediction[
                        "cer"
                    ],

                "v4_wer":
                    v4_prediction[
                        "wer"
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
                "sample_id",
                "source_id",
                "ground_truth",
                "v3_prediction",
                "v3_cer",
                "v3_wer",
                "v4_prediction",
                "v4_cer",
                "v4_wer",
            ]
        )

        writer.writeheader()

        writer.writerows(
            combined_rows
        )

    # ========================================================
    # SAVE JSON SUMMARY
    # ========================================================

    summary = {

        "validation_csv":
            str(
                VALIDATION_CSV
            ),

        "validation_samples":
            len(
                records
            ),

        "v3": {

            "model":
                str(
                    V3_MODEL_PATH
                ),

            "checkpoint_epoch":
                v3_checkpoint.get(
                    "epoch"
                ),

            "cer":
                v3_result[
                    "cer"
                ],

            "wer":
                v3_result[
                    "wer"
                ],

            "exact_matches":
                v3_result[
                    "exact_matches"
                ],

            "per_source":
                v3_sources,
        },

        "v4": {

            "model":
                str(
                    V4_MODEL_PATH
                ),

            "checkpoint_epoch":
                v4_checkpoint.get(
                    "epoch"
                ),

            "cer":
                v4_result[
                    "cer"
                ],

            "wer":
                v4_result[
                    "wer"
                ],

            "exact_matches":
                v4_result[
                    "exact_matches"
                ],

            "per_source":
                v4_sources,
        },

        "comparison": {

            "cer_absolute_improvement":
                cer_change,

            "cer_relative_improvement_percent":
                cer_relative_improvement,

            "wer_absolute_improvement":
                wer_change,

            "wer_relative_improvement_percent":
                wer_relative_improvement,
        }
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
    print(
        "Prediction CSV:"
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


# ============================================================
# ENTRY POINT
# ============================================================

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