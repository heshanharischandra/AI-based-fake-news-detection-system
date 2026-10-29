from pathlib import Path
import csv
import json
import unicodedata

import cv2


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

SYNTHETIC_DIR = (
    OCR_ROOT
    / "synthetic_lines"
)

LABELS_CSV = (
    OCR_ROOT
    / "metadata"
    / "synthetic_labels.csv"
)

MANIFEST_JSON = (
    OCR_ROOT
    / "metadata"
    / "synthetic_manifest.json"
)


# ============================================================
# 2. HELPERS
# ============================================================

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


def contains_sinhala(text):

    return any(
        "\u0D80" <= ch <= "\u0DFF"
        for ch in text
    )


# ============================================================
# 3. VALIDATE
# ============================================================

def validate_dataset():

    print(
        "=" * 72
    )

    print(
        "FULL SYNTHETIC SINHALA OCR DATASET VALIDATION"
    )

    print(
        "=" * 72
    )


    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    if not LABELS_CSV.exists():

        print(
            "\nERROR: synthetic_labels.csv not found."
        )

        print(
            LABELS_CSV
        )

        return


    if not MANIFEST_JSON.exists():

        print(
            "\nERROR: synthetic_manifest.json not found."
        )

        print(
            MANIFEST_JSON
        )

        return


    # --------------------------------------------------------
    # Actual image files
    # --------------------------------------------------------

    actual_images = list(
        SYNTHETIC_DIR.glob(
            "synthetic_*.png"
        )
    )

    actual_image_count = len(
        actual_images
    )


    print(
        f"\nActual synthetic images: "
        f"{actual_image_count}"
    )


    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    with open(
        LABELS_CSV,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file
        )

        rows = list(
            reader
        )


    csv_count = len(
        rows
    )


    print(
        f"CSV label rows: "
        f"{csv_count}"
    )


    # --------------------------------------------------------
    # Read JSON manifest
    # --------------------------------------------------------

    with open(
        MANIFEST_JSON,
        "r",
        encoding="utf-8"
    ) as file:

        manifest = json.load(
            file
        )


    manifest_count = manifest.get(
        "sample_count",
        0
    )


    manifest_samples = manifest.get(
        "samples",
        []
    )


    print(
        f"Manifest sample_count: "
        f"{manifest_count}"
    )

    print(
        f"Manifest sample entries: "
        f"{len(manifest_samples)}"
    )


    # ========================================================
    # SAMPLE VALIDATION
    # ========================================================

    valid_count = 0
    invalid_count = 0

    seen_sample_ids = set()

    errors_found = []


    for row_number, row in enumerate(
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

        ground_truth = normalize_text(
            row.get(
                "ground_truth",
                ""
            )
        )

        language = normalize_text(
            row.get(
                "language",
                ""
            )
        )


        sample_errors = []


        # ----------------------------------------------------
        # Sample ID
        # ----------------------------------------------------

        if not sample_id:

            sample_errors.append(
                "Missing sample_id"
            )

        elif sample_id in seen_sample_ids:

            sample_errors.append(
                "Duplicate sample_id"
            )

        else:

            seen_sample_ids.add(
                sample_id
            )


        # ----------------------------------------------------
        # Image path
        # ----------------------------------------------------

        if not relative_path:

            sample_errors.append(
                "Missing image_path"
            )

            image_path = None

        else:

            image_path = (
                OCR_ROOT
                / relative_path
            )


            if not image_path.exists():

                sample_errors.append(
                    "Image file missing"
                )

            else:

                image = cv2.imread(
                    str(image_path)
                )


                if image is None:

                    sample_errors.append(
                        "Image cannot be opened"
                    )

                else:

                    height, width = (
                        image.shape[:2]
                    )


                    if width < 40:

                        sample_errors.append(
                            "Image width too small"
                        )


                    if height < 20:

                        sample_errors.append(
                            "Image height too small"
                        )


        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        if not ground_truth:

            sample_errors.append(
                "Ground truth empty"
            )


        # ----------------------------------------------------
        # Language
        # ----------------------------------------------------

        if language != "si":

            sample_errors.append(
                f"Unexpected language: {language}"
            )


        # ----------------------------------------------------
        # Sinhala content
        # ----------------------------------------------------

        if (
            ground_truth
            and not contains_sinhala(
                ground_truth
            )
        ):

            sample_errors.append(
                "Ground truth has no Sinhala characters"
            )


        # ----------------------------------------------------
        # Final sample result
        # ----------------------------------------------------

        if sample_errors:

            invalid_count += 1

            errors_found.append({
                "row":
                    row_number,

                "sample_id":
                    sample_id,

                "errors":
                    sample_errors
            })

        else:

            valid_count += 1


    # ========================================================
    # GLOBAL CONSISTENCY CHECKS
    # ========================================================

    global_errors = []


    if actual_image_count != csv_count:

        global_errors.append(
            "Actual image count does not match CSV rows"
        )


    if csv_count != manifest_count:

        global_errors.append(
            "CSV row count does not match manifest sample_count"
        )


    if len(manifest_samples) != manifest_count:

        global_errors.append(
            "Manifest samples array count does not match sample_count"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "VALIDATION SUMMARY"
    )

    print(
        "=" * 72
    )


    print(
        f"\nValid samples   : "
        f"{valid_count}"
    )

    print(
        f"Invalid samples : "
        f"{invalid_count}"
    )

    print(
        f"CSV samples     : "
        f"{csv_count}"
    )

    print(
        f"Image files     : "
        f"{actual_image_count}"
    )

    print(
        f"Manifest count  : "
        f"{manifest_count}"
    )


    # --------------------------------------------------------
    # Print first errors only
    # --------------------------------------------------------

    if errors_found:

        print(
            "\nFirst validation errors:"
        )


        for item in errors_found[:10]:

            print(
                f"\nRow {item['row']} "
                f"- {item['sample_id']}"
            )


            for error in item[
                "errors"
            ]:

                print(
                    f"  - {error}"
                )


    if global_errors:

        print(
            "\nGlobal dataset errors:"
        )


        for error in global_errors:

            print(
                f"  - {error}"
            )


    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if (
        invalid_count == 0
        and not global_errors
        and valid_count > 0
    ):

        print(
            "\n" + "=" * 72
        )

        print(
            "DATASET VALIDATION PASSED"
        )

        print(
            "=" * 72
        )

        print(
            "\nSynthetic OCR dataset "
            "is ready for model training."
        )

    else:

        print(
            "\n" + "=" * 72
        )

        print(
            "DATASET VALIDATION FAILED"
        )

        print(
            "=" * 72
        )

        print(
            "\nFix dataset errors "
            "before model training."
        )


# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":

    validate_dataset()