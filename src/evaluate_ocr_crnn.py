from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from ocr_dataset import (
    OCRDataset,
    ocr_collate_fn
)

from ocr_crnn_model import CRNN


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

SPLIT_DIR = (
    OCR_ROOT
    / "metadata"
    / "splits"
)

TEST_CSV = (
    SPLIT_DIR
    / "test.csv"
)

REAL_HOLDOUT_CSV = (
    SPLIT_DIR
    / "real_holdout.csv"
)

CHARSET_JSON = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_full"
    / "best_crnn_model.pt"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

BATCH_SIZE = 4

NUM_WORKERS = 0

MAX_PRINT_SAMPLES = 10


# ============================================================
# 3. LOAD CHARSET
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
# 4. GREEDY CTC DECODER
# ============================================================

def greedy_ctc_decode(
    log_probs,
    index_to_char,
    blank_index
):

    # log_probs:
    # [Time, Batch, Classes]

    predicted_indices = (
        log_probs.argmax(
            dim=2
        )
    )

    # [Time, Batch]
    # →
    # [Batch, Time]

    predicted_indices = (
        predicted_indices
        .permute(
            1,
            0
        )
        .cpu()
        .tolist()
    )

    decoded_texts = []

    for sequence in predicted_indices:

        characters = []

        previous_index = None

        for index in sequence:

            # CTC rule:
            # collapse repeated predictions
            if index == previous_index:

                continue

            previous_index = index

            # Remove CTC blank
            if index == blank_index:

                continue

            character = (
                index_to_char.get(
                    index,
                    ""
                )
            )

            characters.append(
                character
            )

        decoded_text = "".join(
            characters
        )

        decoded_texts.append(
            decoded_text
        )

    return decoded_texts


# ============================================================
# 5. LEVENSHTEIN DISTANCE
# ============================================================

def levenshtein_distance(
    reference,
    hypothesis
):

    previous_row = list(
        range(
            len(hypothesis) + 1
        )
    )

    for i, reference_item in enumerate(
        reference,
        start=1
    ):

        current_row = [
            i
        ]

        for j, hypothesis_item in enumerate(
            hypothesis,
            start=1
        ):

            insert_cost = (
                current_row[
                    j - 1
                ]
                + 1
            )

            delete_cost = (
                previous_row[
                    j
                ]
                + 1
            )

            substitute_cost = (
                previous_row[
                    j - 1
                ]
                + (
                    0
                    if reference_item
                    == hypothesis_item
                    else 1
                )
            )

            current_row.append(
                min(
                    insert_cost,
                    delete_cost,
                    substitute_cost
                )
            )

        previous_row = (
            current_row
        )

    return previous_row[-1]


# ============================================================
# 6. CHARACTER ERROR RATE
# ============================================================

def calculate_cer(
    reference,
    hypothesis
):

    if not reference:

        return (
            0.0
            if not hypothesis
            else 1.0
        )

    distance = (
        levenshtein_distance(
            list(reference),
            list(hypothesis)
        )
    )

    return (
        distance
        / len(reference)
    )


# ============================================================
# 7. WORD ERROR RATE
# ============================================================

def calculate_wer(
    reference,
    hypothesis
):

    reference_words = (
        reference.split()
    )

    hypothesis_words = (
        hypothesis.split()
    )

    if not reference_words:

        return (
            0.0
            if not hypothesis_words
            else 1.0
        )

    distance = (
        levenshtein_distance(
            reference_words,
            hypothesis_words
        )
    )

    return (
        distance
        / len(reference_words)
    )


# ============================================================
# 8. LOAD MODEL
# ============================================================

def load_model(
    device
):

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
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

    print(
        f"\nLoaded checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint train loss: "
        f"{checkpoint['train_loss']:.4f}"
    )

    print(
        f"Checkpoint validation loss: "
        f"{checkpoint['validation_loss']:.4f}"
    )

    return model


# ============================================================
# 9. EVALUATE DATASET
# ============================================================

def evaluate_dataset(
    dataset_name,
    csv_path,
    model,
    device,
    index_to_char,
    blank_index
):

    print(
        "\n" + "=" * 72
    )

    print(
        dataset_name
    )

    print(
        "=" * 72
    )

    dataset = OCRDataset(
        csv_path
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=ocr_collate_fn
    )

    total_character_errors = 0
    total_reference_characters = 0

    total_word_errors = 0
    total_reference_words = 0

    exact_match_count = 0

    printed_samples = 0

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

                # --------------------------------------------
                # Exact match
                # --------------------------------------------

                if prediction == reference:

                    exact_match_count += 1

                # --------------------------------------------
                # Character errors
                # --------------------------------------------

                char_distance = (
                    levenshtein_distance(
                        list(reference),
                        list(prediction)
                    )
                )

                total_character_errors += (
                    char_distance
                )

                total_reference_characters += (
                    len(reference)
                )

                # --------------------------------------------
                # Word errors
                # --------------------------------------------

                reference_words = (
                    reference.split()
                )

                prediction_words = (
                    prediction.split()
                )

                word_distance = (
                    levenshtein_distance(
                        reference_words,
                        prediction_words
                    )
                )

                total_word_errors += (
                    word_distance
                )

                total_reference_words += (
                    len(
                        reference_words
                    )
                )

                # --------------------------------------------
                # Print examples
                # --------------------------------------------

                if (
                    printed_samples
                    < MAX_PRINT_SAMPLES
                ):

                    sample_cer = (
                        calculate_cer(
                            reference,
                            prediction
                        )
                    )

                    sample_wer = (
                        calculate_wer(
                            reference,
                            prediction
                        )
                    )

                    print(
                        f"\nSample "
                        f"{printed_samples + 1}"
                    )

                    print(
                        "Ground Truth:"
                    )

                    print(
                        reference
                    )

                    print(
                        "Prediction:"
                    )

                    print(
                        prediction
                    )

                    print(
                        f"CER: "
                        f"{sample_cer:.4f}"
                    )

                    print(
                        f"WER: "
                        f"{sample_wer:.4f}"
                    )

                    printed_samples += 1

    # ========================================================
    # FINAL METRICS
    # ========================================================

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

        exact_match_count
        / len(dataset)

        if len(dataset) > 0

        else 0.0
    )

    print(
        "\n" + "-" * 72
    )

    print(
        f"{dataset_name} RESULTS"
    )

    print(
        "-" * 72
    )

    print(
        f"Samples          : "
        f"{len(dataset)}"
    )

    print(
        f"CER              : "
        f"{overall_cer:.4f}"
    )

    print(
        f"WER              : "
        f"{overall_wer:.4f}"
    )

    print(
        f"Exact matches    : "
        f"{exact_match_count}/"
        f"{len(dataset)}"
    )

    print(
        f"Exact accuracy   : "
        f"{exact_accuracy * 100:.2f}%"
    )

    return {

        "samples":
            len(dataset),

        "cer":
            overall_cer,

        "wer":
            overall_wer,

        "exact_matches":
            exact_match_count,

        "exact_accuracy":
            exact_accuracy
    }


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "SINHALA OCR CRNN EVALUATION"
    )

    print(
        "=" * 72
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    (
        index_to_char,
        blank_index
    ) = load_charset()

    model = load_model(
        device
    )

    # --------------------------------------------------------
    # Synthetic test
    # --------------------------------------------------------

    synthetic_results = (
        evaluate_dataset(
            "SYNTHETIC TEST SET",
            TEST_CSV,
            model,
            device,
            index_to_char,
            blank_index
        )
    )

    # --------------------------------------------------------
    # Real holdout
    # --------------------------------------------------------

    real_results = (
        evaluate_dataset(
            "REAL NEWS IMAGE HOLDOUT",
            REAL_HOLDOUT_CSV,
            model,
            device,
            index_to_char,
            blank_index
        )
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        "\nSynthetic test:"
    )

    print(
        f"CER: "
        f"{synthetic_results['cer']:.4f}"
    )

    print(
        f"WER: "
        f"{synthetic_results['wer']:.4f}"
    )

    print(
        "\nReal holdout:"
    )

    print(
        f"CER: "
        f"{real_results['cer']:.4f}"
    )

    print(
        f"WER: "
        f"{real_results['wer']:.4f}"
    )

    print(
        "\nEvaluation complete."
    )