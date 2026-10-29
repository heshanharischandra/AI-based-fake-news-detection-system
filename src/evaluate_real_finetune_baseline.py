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

VALIDATION_CSV = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "splits"
    / "real_finetune_validation.csv"
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
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "REAL OCR FINE-TUNING BASELINE EVALUATION"
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

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = OCRDataset(
        VALIDATION_CSV
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=ocr_collate_fn
    )

    print(
        f"Validation samples: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # Charset
    # --------------------------------------------------------

    (
        index_to_char,
        blank_index
    ) = load_charset()

    # --------------------------------------------------------
    # Load synthetic-trained model
    # --------------------------------------------------------

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
        f"\nLoaded synthetic checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint validation loss: "
        f"{checkpoint['validation_loss']:.4f}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_character_errors = 0
    total_characters = 0

    total_word_errors = 0
    total_words = 0

    exact_matches = 0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

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

            # Character error
            char_distance = (
                levenshtein_distance(
                    list(reference),
                    list(prediction)
                )
            )

            total_character_errors += (
                char_distance
            )

            total_characters += (
                len(reference)
            )

            # Word error
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

            total_words += (
                len(reference_words)
            )

            if prediction == reference:

                exact_matches += 1

            sample_cer = (
                char_distance
                / len(reference)
                if reference
                else 0.0
            )

            sample_wer = (
                word_distance
                / len(reference_words)
                if reference_words
                else 0.0
            )

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

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    overall_cer = (
        total_character_errors
        / total_characters
        if total_characters
        else 0.0
    )

    overall_wer = (
        total_word_errors
        / total_words
        if total_words
        else 0.0
    )

    exact_accuracy = (
        exact_matches
        / len(dataset)
        if len(dataset)
        else 0.0
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "REAL VALIDATION BASELINE RESULTS"
    )

    print(
        "=" * 72
    )

    print(
        f"\nSamples        : "
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

    print(
        "\nThis is the BEFORE fine-tuning baseline."
    )