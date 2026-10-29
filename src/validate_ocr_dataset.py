from pathlib import Path
import csv
import unicodedata

import cv2


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

CSV_PATH = (
    OCR_ROOT
    / "metadata"
    / "real_labels.csv"
)


# ============================================================
# TEXT HELPERS
# ============================================================

def contains_sinhala(text):

    return any(
        "\u0D80" <= ch <= "\u0DFF"
        for ch in text
    )


def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.replace(
        "\ufeff",
        ""
    )

    text = unicodedata.normalize(
        "NFC",
        text
    )

    return text.strip()


# ============================================================
# NORMALIZE CSV ROW
# ============================================================

def normalize_row(row):
    """
    Remove BOM, spaces and hidden characters
    from CSV column names and values.
    """

    cleaned = {}

    for key, value in row.items():

        if key is None:
            continue

        clean_key = (
            str(key)
            .replace("\ufeff", "")
            .strip()
        )

        clean_value = (
            ""
            if value is None
            else str(value).strip()
        )

        cleaned[
            clean_key
        ] = clean_value

    return cleaned


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset():

    if not CSV_PATH.exists():

        print(
            "ERROR: CSV not found:"
        )

        print(
            CSV_PATH
        )

        return

    print(
        "=" * 65
    )

    print(
        "SINHALA OCR DATASET VALIDATION"
    )

    print(
        "=" * 65
    )

    with open(
        CSV_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file
        )

        print(
            "\nDetected CSV columns:"
        )

        if reader.fieldnames:

            for column in reader.fieldnames:

                clean_column = (
                    str(column)
                    .replace("\ufeff", "")
                    .strip()
                )

                print(
                    f"  - {clean_column}"
                )

        raw_rows = list(
            reader
        )

    rows = [
        normalize_row(row)
        for row in raw_rows
    ]

    print(
        f"\nTotal CSV samples: "
        f"{len(rows)}"
    )

    valid_count = 0
    invalid_count = 0

    # Detect duplicate IDs
    seen_ids = set()

    for index, row in enumerate(
        rows,
        start=1
    ):

        sample_id = normalize_text(
            row.get(
                "sample_id",
                ""
            )
        )

        relative_path = normalize_text(
            row.get(
                "image_path",
                ""
            )
        )

        source_id = normalize_text(
            row.get(
                "source_id",
                ""
            )
        )

        line_number = normalize_text(
            row.get(
                "line_number",
                ""
            )
        )

        text_color = normalize_text(
            row.get(
                "text_color",
                ""
            )
        )

        language = normalize_text(
            row.get(
                "language",
                ""
            )
        )

        ground_truth = normalize_text(
            row.get(
                "ground_truth",
                ""
            )
        )

        errors = []

        # ----------------------------------------------------
        # CHECK SAMPLE ID
        # ----------------------------------------------------

        if not sample_id:

            errors.append(
                "Missing sample_id"
            )

        elif sample_id in seen_ids:

            errors.append(
                "Duplicate sample_id"
            )

        else:

            seen_ids.add(
                sample_id
            )

        # ----------------------------------------------------
        # CHECK IMAGE PATH
        # ----------------------------------------------------

        if not relative_path:

            errors.append(
                "Missing image_path"
            )

            image_path = None

        else:

            image_path = (
                OCR_ROOT
                / relative_path
            )

            if not image_path.exists():

                errors.append(
                    "Image file does not exist"
                )

            else:

                image = cv2.imread(
                    str(image_path)
                )

                if image is None:

                    errors.append(
                        "Image cannot be opened"
                    )

                else:

                    height, width = (
                        image.shape[:2]
                    )

                    if width < 50:

                        errors.append(
                            "Image width too small"
                        )

                    if height < 15:

                        errors.append(
                            "Image height too small"
                        )

        # ----------------------------------------------------
        # CHECK SOURCE ID
        # ----------------------------------------------------

        if not source_id:

            errors.append(
                "Missing source_id"
            )

        # ----------------------------------------------------
        # CHECK LINE NUMBER
        # ----------------------------------------------------

        if not line_number:

            errors.append(
                "Missing line_number"
            )

        else:

            try:

                int(
                    line_number
                )

            except ValueError:

                errors.append(
                    "Invalid line_number"
                )

        # ----------------------------------------------------
        # CHECK TEXT COLOR
        # ----------------------------------------------------

        if text_color not in {
            "yellow",
            "white"
        }:

            errors.append(
                "Invalid text_color"
            )

        # ----------------------------------------------------
        # CHECK LANGUAGE
        # ----------------------------------------------------

        if language not in {
            "si",
            "en",
            "si+en"
        }:

            errors.append(
                "Invalid language"
            )

        # ----------------------------------------------------
        # CHECK GROUND TRUTH
        # ----------------------------------------------------

        if not ground_truth:

            errors.append(
                "Ground truth is empty"
            )

        # ----------------------------------------------------
        # CHECK SINHALA CHARACTERS
        # ----------------------------------------------------

        if (
            language == "si"
            and ground_truth
            and not contains_sinhala(
                ground_truth
            )
        ):

            errors.append(
                "Sinhala sample contains "
                "no Sinhala characters"
            )

        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        if errors:

            invalid_count += 1

            display_id = (
                sample_id
                if sample_id
                else f"row_{index}"
            )

            print(
                f"\n[INVALID] "
                f"{display_id}"
            )

            for error in errors:

                print(
                    f"  - {error}"
                )

        else:

            valid_count += 1

            print(
                f"\n[VALID] "
                f"{sample_id}"
            )

            print(
                f"  Text: "
                f"{ground_truth}"
            )

            print(
                f"  Language: "
                f"{language}"
            )

            print(
                f"  Color: "
                f"{text_color}"
            )

            print(
                f"  Image: "
                f"{image_path.name}"
            )

            print(
                f"  Size: "
                f"{width}x{height}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 65
    )

    print(
        "VALIDATION SUMMARY"
    )

    print(
        "=" * 65
    )

    print(
        f"Valid samples   : "
        f"{valid_count}"
    )

    print(
        f"Invalid samples : "
        f"{invalid_count}"
    )

    print(
        f"Total samples   : "
        f"{len(rows)}"
    )

    if (
        invalid_count == 0
        and valid_count > 0
    ):

        print(
            "\nDataset validation PASSED."
        )

        print(
            "The real OCR samples are "
            "ready for the next stage."
        )

    else:

        print(
            "\nDataset validation FAILED."
        )

        print(
            "Fix the invalid samples "
            "before training."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    validate_dataset()