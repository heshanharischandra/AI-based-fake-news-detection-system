from pathlib import Path
import csv
from collections import Counter


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
    / "real_training_expansion_v5"
)

CSV_PATH = (
    BASE_DIR
    / "metadata"
    / "v5_manual_line_labels.csv"
)


# ============================================================
# EXPECTED VALUES
# ============================================================

EXPECTED_TOTAL = 48
EXPECTED_VERIFIED = 45
EXPECTED_EXCLUDED = 3

VALID_STATUSES = {
    "VERIFIED",
    "EXCLUDE",
}


# ============================================================
# SINHALA CHARACTER CHECK
# ============================================================

def count_sinhala_characters(text):

    return sum(
        1
        for character in text
        if "\u0D80" <= character <= "\u0DFF"
    )


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print("=" * 72)
    print("V5 MANUAL OCR LABEL VALIDATION")
    print("=" * 72)

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{CSV_PATH}"
        )

    print()
    print("CSV:")
    print(CSV_PATH)

    with open(
        CSV_PATH,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        fieldnames = reader.fieldnames

        rows = list(
            reader
        )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = {
        "sample_id",
        "source_id",
        "source_image",
        "line_number",
        "crop_path",
        "extraction_method",
        "ground_truth_text",
        "review_status",
    }

    missing_columns = (
        required_columns
        - set(
            fieldnames or []
        )
    )

    if missing_columns:

        raise RuntimeError(
            "Missing required columns:\n"
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    # ========================================================
    # COUNTERS
    # ========================================================

    errors = []
    warnings = []

    sample_ids = []

    status_counter = Counter()
    source_counter = Counter()

    verified_count = 0
    excluded_count = 0

    # ========================================================
    # ROW VALIDATION
    # ========================================================

    for row_number, row in enumerate(
        rows,
        start=2,
    ):

        sample_id = (
            row.get(
                "sample_id",
                "",
            ).strip()
        )

        source_id = (
            row.get(
                "source_id",
                "",
            ).strip()
        )

        crop_path_text = (
            row.get(
                "crop_path",
                "",
            ).strip()
        )

        ground_truth = (
            row.get(
                "ground_truth_text",
                "",
            ).strip()
        )

        review_status = (
            row.get(
                "review_status",
                "",
            )
            .strip()
            .upper()
        )

        # ----------------------------------------------------
        # BASIC FIELDS
        # ----------------------------------------------------

        if not sample_id:

            errors.append(
                f"Row {row_number}: "
                "sample_id is blank."
            )

            continue

        sample_ids.append(
            sample_id
        )

        if not source_id:

            errors.append(
                f"{sample_id}: "
                "source_id is blank."
            )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        status_counter[
            review_status
        ] += 1

        if review_status not in VALID_STATUSES:

            errors.append(
                f"{sample_id}: "
                f"invalid review_status "
                f"'{review_status}'."
            )

            continue

        # ----------------------------------------------------
        # CROP FILE
        # ----------------------------------------------------

        if not crop_path_text:

            errors.append(
                f"{sample_id}: "
                "crop_path is blank."
            )

        else:

            crop_path = (
                PROJECT_ROOT
                / crop_path_text
            )

            if not crop_path.exists():

                errors.append(
                    f"{sample_id}: "
                    "crop file does not exist:\n"
                    f"  {crop_path}"
                )

        # ----------------------------------------------------
        # VERIFIED ROW
        # ----------------------------------------------------

        if review_status == "VERIFIED":

            verified_count += 1

            source_counter[
                source_id
            ] += 1

            if not ground_truth:

                errors.append(
                    f"{sample_id}: "
                    "VERIFIED but "
                    "ground_truth_text is blank."
                )

            else:

                sinhala_count = (
                    count_sinhala_characters(
                        ground_truth
                    )
                )

                if sinhala_count == 0:

                    warnings.append(
                        f"{sample_id}: "
                        "VERIFIED label has no "
                        "Sinhala characters."
                    )

        # ----------------------------------------------------
        # EXCLUDED ROW
        # ----------------------------------------------------

        elif review_status == "EXCLUDE":

            excluded_count += 1

            if ground_truth:

                warnings.append(
                    f"{sample_id}: "
                    "EXCLUDE row contains "
                    "ground_truth_text."
                )

    # ========================================================
    # DUPLICATE SAMPLE IDs
    # ========================================================

    duplicate_ids = [
        sample_id
        for sample_id, count
        in Counter(
            sample_ids
        ).items()
        if count > 1
    ]

    if duplicate_ids:

        for sample_id in duplicate_ids:

            errors.append(
                "Duplicate sample_id: "
                f"{sample_id}"
            )

    # ========================================================
    # EXPECTED COUNTS
    # ========================================================

    total_rows = len(
        rows
    )

    if total_rows != EXPECTED_TOTAL:

        errors.append(
            f"Expected {EXPECTED_TOTAL} rows, "
            f"found {total_rows}."
        )

    if verified_count != EXPECTED_VERIFIED:

        errors.append(
            f"Expected {EXPECTED_VERIFIED} "
            f"VERIFIED rows, "
            f"found {verified_count}."
        )

    if excluded_count != EXPECTED_EXCLUDED:

        errors.append(
            f"Expected {EXPECTED_EXCLUDED} "
            f"EXCLUDE rows, "
            f"found {excluded_count}."
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("-" * 72)
    print("SUMMARY")
    print("-" * 72)

    print(
        "Total rows:",
        total_rows,
    )

    print(
        "VERIFIED:",
        verified_count,
    )

    print(
        "EXCLUDE:",
        excluded_count,
    )

    print()

    print(
        "Status counts:",
        dict(
            status_counter
        ),
    )

    print()
    print(
        "Verified lines by source:"
    )

    for source_id in sorted(
        source_counter
    ):

        print(
            f"  {source_id}: "
            f"{source_counter[source_id]}"
        )

    # ========================================================
    # WARNINGS
    # ========================================================

    if warnings:

        print()
        print("-" * 72)
        print("WARNINGS")
        print("-" * 72)

        for warning in warnings:

            print(
                "WARNING:",
                warning,
            )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("-" * 72)

    if errors:

        print("VALIDATION FAILED")
        print("-" * 72)

        for error in errors:

            print()
            print(
                "ERROR:",
                error,
            )

        raise RuntimeError(
            f"Validation failed with "
            f"{len(errors)} error(s)."
        )

    print("VALIDATION PASSED")
    print("-" * 72)

    print()
    print(
        "V5 manual OCR labels are "
        "ready for the next preparation step."
    )

    print("=" * 72)


if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print("ERROR:")
        print(error)