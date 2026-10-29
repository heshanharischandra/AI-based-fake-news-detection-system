from pathlib import Path
import csv
import os

import cv2
import pandas as pd
import pytesseract


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
    / "tesseract_sinhala_confidence"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "tesseract_sinhala_confidence.csv"
)


# ============================================================
# LOAD RECORDS
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
# RESOLVE IMAGE PATH
# ============================================================

def resolve_image_path(record):

    return (
        OCR_ROOT
        / Path(
            record["image_path"]
        )
    )


# ============================================================
# TESSERACT WITH CONFIDENCE
# ============================================================

def run_tesseract(
    image_path
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Cannot read image:\n"
            f"{image_path}"
        )

    config = (
        "--oem 1 "
        "--psm 7"
    )

    data = pytesseract.image_to_data(
        image,
        lang="sin",
        config=config,
        output_type=pytesseract.Output.DATAFRAME
    )

    valid_rows = data[
        (
            data["conf"].notna()
        )
        &
        (
            data["conf"] >= 0
        )
        &
        (
            data["text"].notna()
        )
    ].copy()

    if len(valid_rows) == 0:

        return (
            "",
            0.0,
            0
        )

    valid_rows["text"] = (
        valid_rows["text"]
        .astype(str)
        .str.strip()
    )

    valid_rows = valid_rows[
        valid_rows["text"] != ""
    ]

    if len(valid_rows) == 0:

        return (
            "",
            0.0,
            0
        )

    text = " ".join(
        valid_rows[
            "text"
        ].tolist()
    )

    confidence = float(
        valid_rows[
            "conf"
        ].mean()
    )

    word_count = len(
        valid_rows
    )

    return (
        text,
        confidence,
        word_count
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("TESSERACT SINHALA CONFIDENCE ANALYSIS")
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

    output_rows = []

    blank_count = 0

    confidences = []

    for index, record in enumerate(
        records,
        start=1
    ):

        sample_id = (
            record[
                "sample_id"
            ].strip()
        )

        ground_truth = (
            record[
                "ground_truth"
            ].strip()
        )

        image_path = (
            resolve_image_path(
                record
            )
        )

        (
            text,
            confidence,
            word_count
        ) = run_tesseract(
            image_path
        )

        if not text:

            blank_count += 1

        else:

            confidences.append(
                confidence
            )

        print()
        print(
            f"[{index:02d}/"
            f"{len(records):02d}] "
            f"{sample_id}"
        )

        print(
            "GT   :",
            ground_truth
        )

        print(
            "TESS :",
            text
            if text
            else "[BLANK]"
        )

        print(
            "Confidence:",
            f"{confidence:.2f}"
        )

        print(
            "OCR words:",
            word_count
        )

        output_rows.append(
            {
                "sample_id":
                    sample_id,

                "source_id":
                    record.get(
                        "source_id",
                        ""
                    ),

                "ground_truth":
                    ground_truth,

                "tesseract_text":
                    text,

                "tesseract_confidence":
                    confidence,

                "ocr_word_count":
                    word_count,

                "blank":
                    not bool(text),
            }
        )

    with open(
        OUTPUT_CSV,
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
                "tesseract_text",
                "tesseract_confidence",
                "ocr_word_count",
                "blank",
            ]
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    print()
    print("=" * 72)
    print("CONFIDENCE SUMMARY")
    print("=" * 72)

    print(
        "Total samples:",
        len(records)
    )

    print(
        "Blank Tesseract outputs:",
        blank_count
    )

    print(
        "Non-blank outputs:",
        len(records) - blank_count
    )

    if confidences:

        print(
            "Minimum confidence:",
            f"{min(confidences):.2f}"
        )

        print(
            "Maximum confidence:",
            f"{max(confidences):.2f}"
        )

        print(
            "Average confidence:",
            f"{sum(confidences) / len(confidences):.2f}"
        )

    print()
    print(
        "Saved CSV:"
    )

    print(
        OUTPUT_CSV
    )

    print("=" * 72)


if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print("ERROR:")
        print(error)