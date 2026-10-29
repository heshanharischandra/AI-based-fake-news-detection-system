from pathlib import Path
import csv
import json
import os

import cv2
import pytesseract
import torch
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

V3_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sinhala_ocr_crnn_crop_jitter_v3"
    / "last_crop_jitter_v3_model.pt"
)

TESSERACT_EXE = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

TESSDATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "tessdata_best"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "v3_vs_tesseract_sinhala"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_CSV = (
    OUTPUT_DIR
    / "v3_vs_tesseract_clean_validation.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "v3_vs_tesseract_summary.json"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_records():

    with open(
        VALIDATION_CSV,
        "r",
        encoding="utf-8-sig"
    ) as file:

        return list(
            csv.DictReader(file)
        )


# ============================================================
# IMAGE PATH
# ============================================================

def resolve_image_path(record):

    relative_path = str(
        record[
            "image_path"
        ]
    ).strip()

    return (
        OCR_ROOT
        / Path(relative_path)
    )


# ============================================================
# CHARSET
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
# CER / WER
# ============================================================

def calculate_metrics(
    reference,
    prediction
):

    char_errors = (
        levenshtein_distance(
            list(reference),
            list(prediction)
        )
    )

    ref_chars = len(
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

    ref_words = len(
        reference_words
    )

    cer = (
        char_errors / ref_chars
        if ref_chars > 0
        else 0.0
    )

    wer = (
        word_errors / ref_words
        if ref_words > 0
        else 0.0
    )

    return (
        char_errors,
        ref_chars,
        word_errors,
        ref_words,
        cer,
        wer
    )


# ============================================================
# LOAD V3
# ============================================================

def load_v3(
    device
):

    checkpoint = torch.load(
        V3_MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    num_classes = int(
        checkpoint["num_classes"]
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

    return model


# ============================================================
# V3 PREDICTIONS
# ============================================================

def get_v3_predictions(
    records,
    device,
    index_to_char,
    blank_index
):

    dataset = OCRDataset(
        VALIDATION_CSV
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=ocr_collate_fn
    )

    model = load_v3(
        device
    )

    predictions = []

    with torch.no_grad():

        for batch in loader:

            images = (
                batch["images"]
                .to(device)
            )

            log_probs = model(
                images
            )

            batch_predictions = (
                greedy_ctc_decode(
                    log_probs,
                    index_to_char,
                    blank_index
                )
            )

            predictions.extend(
                batch_predictions
            )

    return predictions


# ============================================================
# TESSERACT SINHALA
# ============================================================

def run_tesseract_sinhala(
    image_path
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Cannot read image:\n{image_path}"
        )

    config = (
        "--oem 1 "
        "--psm 7"
    )

    text = pytesseract.image_to_string(
        image,
        lang="sin",
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

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("V3 CRNN vs TESSERACT SINHALA")
    print("=" * 72)

    if not TESSERACT_EXE.exists():

        raise FileNotFoundError(
            f"Tesseract not found:\n"
            f"{TESSERACT_EXE}"
        )

    pytesseract.pytesseract.tesseract_cmd = str(
        TESSERACT_EXE
    )

    os.environ[
        "TESSDATA_PREFIX"
    ] = str(
        TESSDATA_DIR
    )

    records = load_records()

    print()
    print(
        "Validation samples:",
        len(records)
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    (
        index_to_char,
        blank_index
    ) = load_charset()

    # ========================================================
    # V3
    # ========================================================

    print()
    print(
        "Running V3 CRNN..."
    )

    v3_predictions = (
        get_v3_predictions(
            records,
            device,
            index_to_char,
            blank_index
        )
    )

    # ========================================================
    # BOTH SYSTEMS
    # ========================================================

    rows = []

    v3_char_errors = 0
    v3_ref_chars = 0
    v3_word_errors = 0
    v3_ref_words = 0
    v3_exact = 0

    tess_char_errors = 0
    tess_ref_chars = 0
    tess_word_errors = 0
    tess_ref_words = 0
    tess_exact = 0

    v3_wins = 0
    tess_wins = 0
    ties = 0

    print()
    print(
        "Running Tesseract Sinhala..."
    )

    for index, record in enumerate(
        records
    ):

        sample_id = (
            record["sample_id"]
            .strip()
        )

        source_id = (
            record["source_id"]
            .strip()
        )

        ground_truth = (
            record["ground_truth"]
            .strip()
        )

        image_path = (
            resolve_image_path(
                record
            )
        )

        v3_text = (
            v3_predictions[index]
            .strip()
        )

        tess_text = (
            run_tesseract_sinhala(
                image_path
            )
        )

        (
            v3_ce,
            v3_rc,
            v3_we,
            v3_rw,
            v3_cer,
            v3_wer
        ) = calculate_metrics(
            ground_truth,
            v3_text
        )

        (
            tess_ce,
            tess_rc,
            tess_we,
            tess_rw,
            tess_cer,
            tess_wer
        ) = calculate_metrics(
            ground_truth,
            tess_text
        )

        v3_char_errors += v3_ce
        v3_ref_chars += v3_rc

        v3_word_errors += v3_we
        v3_ref_words += v3_rw

        tess_char_errors += tess_ce
        tess_ref_chars += tess_rc

        tess_word_errors += tess_we
        tess_ref_words += tess_rw

        if v3_text == ground_truth:

            v3_exact += 1

        if tess_text == ground_truth:

            tess_exact += 1

        if v3_cer < tess_cer:

            winner = "V3"

            v3_wins += 1

        elif tess_cer < v3_cer:

            winner = "Tesseract"

            tess_wins += 1

        else:

            winner = "Tie"

            ties += 1

        print()
        print(
            f"[{index + 1:02d}/"
            f"{len(records):02d}] "
            f"{sample_id}"
        )

        print(
            "GT   :",
            ground_truth
        )

        print(
            "V3   :",
            v3_text
        )

        print(
            "TESS :",
            tess_text
        )

        print(
            "CER  : "
            f"V3={v3_cer:.4f} "
            f"| TESS={tess_cer:.4f} "
            f"| Winner={winner}"
        )

        rows.append(
            {
                "sample_id":
                    sample_id,

                "source_id":
                    source_id,

                "ground_truth":
                    ground_truth,

                "v3_prediction":
                    v3_text,

                "v3_cer":
                    v3_cer,

                "v3_wer":
                    v3_wer,

                "tesseract_prediction":
                    tess_text,

                "tesseract_cer":
                    tess_cer,

                "tesseract_wer":
                    tess_wer,

                "winner":
                    winner,
            }
        )

    # ========================================================
    # OVERALL
    # ========================================================

    v3_cer = (
        v3_char_errors
        / v3_ref_chars
    )

    v3_wer = (
        v3_word_errors
        / v3_ref_words
    )

    tess_cer = (
        tess_char_errors
        / tess_ref_chars
    )

    tess_wer = (
        tess_word_errors
        / tess_ref_words
    )

    print()
    print("=" * 72)
    print("FINAL COMPARISON")
    print("=" * 72)

    print()
    print(
        "V3 CRNN"
    )

    print(
        "CER:",
        f"{v3_cer:.4f}"
    )

    print(
        "WER:",
        f"{v3_wer:.4f}"
    )

    print(
        "Exact:",
        f"{v3_exact}/"
        f"{len(records)}"
    )

    print()
    print(
        "TESSERACT SINHALA"
    )

    print(
        "CER:",
        f"{tess_cer:.4f}"
    )

    print(
        "WER:",
        f"{tess_wer:.4f}"
    )

    print(
        "Exact:",
        f"{tess_exact}/"
        f"{len(records)}"
    )

    print()
    print(
        "Sample-level CER wins:"
    )

    print(
        "V3 wins:",
        v3_wins
    )

    print(
        "Tesseract wins:",
        tess_wins
    )

    print(
        "Ties:",
        ties
    )

    if tess_cer < v3_cer:

        print()
        print(
            "OVERALL CER WINNER: TESSERACT SINHALA"
        )

    elif v3_cer < tess_cer:

        print()
        print(
            "OVERALL CER WINNER: V3 CRNN"
        )

    else:

        print()
        print(
            "OVERALL CER: TIE"
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

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
                "tesseract_prediction",
                "tesseract_cer",
                "tesseract_wer",
                "winner",
            ]
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {

        "validation_samples":
            len(records),

        "v3": {

            "cer":
                v3_cer,

            "wer":
                v3_wer,

            "exact_matches":
                v3_exact,

            "sample_cer_wins":
                v3_wins,
        },

        "tesseract": {

            "cer":
                tess_cer,

            "wer":
                tess_wer,

            "exact_matches":
                tess_exact,

            "sample_cer_wins":
                tess_wins,
        },

        "ties":
            ties,
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