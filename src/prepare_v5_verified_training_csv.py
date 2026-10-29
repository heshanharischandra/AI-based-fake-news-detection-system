from pathlib import Path
import csv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
    / "real_training_expansion_v5"
)

INPUT_CSV = (
    BASE_DIR
    / "metadata"
    / "v5_manual_line_labels.csv"
)

OUTPUT_CSV = (
    BASE_DIR
    / "metadata"
    / "v5_verified_real_lines.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("PREPARE V5 VERIFIED REAL OCR TRAINING CSV")
    print("=" * 72)

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Input CSV not found:\n"
            f"{INPUT_CSV}"
        )

    with open(
        INPUT_CSV,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        rows = list(
            reader
        )

    verified_rows = []

    excluded_count = 0

    for row in rows:

        review_status = (
            row.get(
                "review_status",
                "",
            )
            .strip()
            .upper()
        )

        ground_truth = (
            row.get(
                "ground_truth_text",
                "",
            ).strip()
        )

        sample_id = (
            row.get(
                "sample_id",
                "",
            ).strip()
        )

        crop_path = (
            row.get(
                "crop_path",
                "",
            ).strip()
        )

        if review_status == "EXCLUDE":

            excluded_count += 1
            continue

        if review_status != "VERIFIED":

            raise RuntimeError(
                f"{sample_id}: "
                f"unexpected review_status "
                f"'{review_status}'"
            )

        if not ground_truth:

            raise RuntimeError(
                f"{sample_id}: "
                "VERIFIED row has blank "
                "ground_truth_text."
            )

        if not crop_path:

            raise RuntimeError(
                f"{sample_id}: "
                "crop_path is blank."
            )

        full_crop_path = (
            PROJECT_ROOT
            / crop_path
        )

        if not full_crop_path.exists():

            raise FileNotFoundError(
                f"{sample_id}: "
                f"crop file not found:\n"
                f"{full_crop_path}"
            )

        verified_rows.append(
            {
                "sample_id":
                    sample_id,

                "source_id":
                    row[
                        "source_id"
                    ].strip(),

                "image_path":
                    crop_path,

                "text":
                    ground_truth,

                "language":
                    "si",

                "source":
                    "v5_real_manual_verified",
            }
        )

    if len(verified_rows) != 45:

        raise RuntimeError(
            f"Expected 45 verified rows, "
            f"found {len(verified_rows)}."
        )

    fieldnames = [
        "sample_id",
        "source_id",
        "image_path",
        "text",
        "language",
        "source",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            verified_rows
        )

    print()
    print("Original rows:", len(rows))

    print(
        "VERIFIED rows:",
        len(verified_rows),
    )

    print(
        "EXCLUDED rows:",
        excluded_count,
    )

    print()
    print("Training-ready CSV:")
    print(OUTPUT_CSV)

    print()
    print("=" * 72)
    print("V5 VERIFIED DATASET PREPARATION PASSED")
    print("=" * 72)


if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print("ERROR:")
        print(error)