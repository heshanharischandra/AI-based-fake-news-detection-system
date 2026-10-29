from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader

from ocr_dataset import (
    OCRDataset,
    ocr_collate_fn
)

from ocr_crnn_model import CRNN

from evaluate_ocr_crnn import (
    greedy_ctc_decode,
    levenshtein_distance
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

CHARSET_JSON = (
    OCR_ROOT
    / "metadata"
    / "charset.json"
)

REAL_VALIDATION_CSV = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "splits_v2"
    / "real_finetune_validation_v2.csv"
)

A9_HOLDOUT_CSV = (
    OCR_ROOT
    / "metadata"
    / "splits"
    / "real_holdout.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_expanded_v2"
    / "best_expanded_v2_model.pt"
)


# ============================================================
# LOAD CHARSET
# ============================================================

def load_charset():

    with open(
        CHARSET_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    index_to_char = {
        int(index): character
        for index, character
        in data["index_to_char"].items()
    }

    blank_index = int(
        data["ctc_blank_index"]
    )

    return (
        index_to_char,
        blank_index
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(device):

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model = CRNN(
        num_classes=int(
            checkpoint["num_classes"]
        ),
        hidden_size=int(
            checkpoint.get(
                "hidden_size",
                256
            )
        )
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
        f"\nLoaded fine-tuned epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint validation CER: "
        f"{checkpoint['validation_cer']:.4f}"
    )

    return model


# ============================================================
# EVALUATE DATASET
# ============================================================

def evaluate_dataset(
    title,
    csv_path,
    model,
    device,
    index_to_char,
    blank_index
):

    print(
        "\n" + "=" * 72
    )

    print(title)

    print(
        "=" * 72
    )

    dataset = OCRDataset(
        csv_path
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=ocr_collate_fn
    )

    total_character_errors = 0
    total_reference_characters = 0

    total_word_errors = 0
    total_reference_words = 0

    exact_matches = 0

    with torch.no_grad():

        for sample_number, batch in enumerate(
            loader,
            start=1
        ):

            images = (
                batch["images"]
                .to(device)
            )

            reference = (
                batch["texts"][0]
            )

            log_probs = model(
                images
            )

            prediction = (
                greedy_ctc_decode(
                    log_probs,
                    index_to_char,
                    blank_index
                )[0]
            )

            # ------------------------------------------------
            # CER
            # ------------------------------------------------

            character_distance = (
                levenshtein_distance(
                    list(reference),
                    list(prediction)
                )
            )

            total_character_errors += (
                character_distance
            )

            total_reference_characters += (
                len(reference)
            )

            sample_cer = (
                character_distance
                / len(reference)
                if reference
                else 0.0
            )

            # ------------------------------------------------
            # WER
            # ------------------------------------------------

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
                len(reference_words)
            )

            sample_wer = (
                word_distance
                / len(reference_words)
                if reference_words
                else 0.0
            )

            # ------------------------------------------------
            # Exact
            # ------------------------------------------------

            if prediction == reference:

                exact_matches += 1

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            print(
                "\n" + "-" * 72
            )

            print(
                f"Sample {sample_number}"
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
                f"CER: {sample_cer:.4f}"
            )

            print(
                f"WER: {sample_wer:.4f}"
            )

    # ========================================================
    # OVERALL
    # ========================================================

    overall_cer = (
        total_character_errors
        / total_reference_characters
        if total_reference_characters
        else 0.0
    )

    overall_wer = (
        total_word_errors
        / total_reference_words
        if total_reference_words
        else 0.0
    )

    exact_accuracy = (
        exact_matches
        / len(dataset)
        if len(dataset)
        else 0.0
    )

    print(
        "\n" + "-" * 72
    )

    print(
        f"{title} RESULTS"
    )

    print(
        "-" * 72
    )

    print(
        f"Samples        : "
        f"{len(dataset)}"
    )

    print(
        f"CER            : "
        f"{overall_cer:.4f}"
    )

    print(
        f"WER            : "
        f"{overall_wer:.4f}"
    )

    print(
        f"Exact matches  : "
        f"{exact_matches}/"
        f"{len(dataset)}"
    )

    print(
        f"Exact accuracy : "
        f"{exact_accuracy * 100:.2f}%"
    )

    return {
        "samples": len(dataset),
        "cer": overall_cer,
        "wer": overall_wer,
        "exact_matches": exact_matches,
        "exact_accuracy": exact_accuracy
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "FINE-TUNED REAL SINHALA OCR EVALUATION"
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
            torch.cuda.get_device_name(0)
        )

    (
        index_to_char,
        blank_index
    ) = load_charset()

    model = load_model(
        device
    )

    # ========================================================
    # 1. REAL VALIDATION
    # ========================================================

    validation_results = (
        evaluate_dataset(
            "REAL FINE-TUNE VALIDATION",
            REAL_VALIDATION_CSV,
            model,
            device,
            index_to_char,
            blank_index
        )
    )

    # ========================================================
    # 2. UNTOUCHED A9 HOLDOUT
    # ========================================================

    holdout_results = (
        evaluate_dataset(
            "A9 EXTERNAL REAL HOLDOUT",
            A9_HOLDOUT_CSV,
            model,
            device,
            index_to_char,
            blank_index
        )
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "FINAL FINE-TUNED MODEL SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        "\nReal validation:"
    )

    print(
        f"CER : "
        f"{validation_results['cer']:.4f}"
    )

    print(
        f"WER : "
        f"{validation_results['wer']:.4f}"
    )

    print(
        "\nA9 external holdout:"
    )

    print(
        f"CER : "
        f"{holdout_results['cer']:.4f}"
    )

    print(
        f"WER : "
        f"{holdout_results['wer']:.4f}"
    )

    print(
        "\nReference baseline before fine-tuning:"
    )

    print(
        "Real validation CER : 0.9286"
    )

    print(
        "A9 holdout CER      : 1.2778"
    )

    print(
        "\nEvaluation complete."
    )