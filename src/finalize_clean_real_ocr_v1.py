from pathlib import Path
import csv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEAN_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
    / "cleaned_real_ocr_v1"
)

INPUT_CSV = (
    CLEAN_ROOT
    / "metadata"
    / "cleaned_real_labels_v1.csv"
)

OUTPUT_CSV = (
    CLEAN_ROOT
    / "metadata"
    / "cleaned_real_labels_v1_final.csv"
)

LINES_DIR = (
    CLEAN_ROOT
    / "lines"
)


# ============================================================
# MANUALLY VERIFIED SPLIT LABELS
# ============================================================

SPLIT_LABELS = {

    "real_train_line_000018_split_1":
        "නාමල්ට අධිචෝදනා ගොනුකරන බව",

    "real_train_line_000018_split_2":
        "CID ය දන්වයි",

    "real_train_line_000043_split_1":
        "ඉදිරි මාස කිහිපය තුළ",

    "real_train_line_000043_split_2":
        "විවෘත සහ සීමිත පදනම මත",

    "real_train_line_000044_split_1":
        "ගුරුවරු බඳවා ගැනීම සඳහා",

    "real_train_line_000044_split_2":
        "කටයුතු කරනවා",

    "real_train_line_000046_split_1":
        "ලංකා ඉතිහාසයේ දැවැන්තම",

    "real_train_line_000046_split_2":
        "ආණ්ඩු විරෝධී රැස්වීම නොවැ. 21",

    "real_train_line_000047_split_1":
        "තියන්න සහෝදර පක්ෂ එකතු වෙලා",

    "real_train_line_000047_split_2":
        "අපි තීන්දු කළා",

    "real_train_line_000092_split_1":
        "ක්ෂණික හැරවුමයි",

    "real_train_line_000092_split_2":
        "සුදු ජාතිකයින් පිරිසක් ඉදිරිපිට",

    "real_train_line_000099_split_1":
        "කබරගල නාය යැමෙන් අසරණ වූ",

    "real_train_line_000099_split_2":
        "පූණාගල පවුල් 50 ට",
}


# ============================================================
# LOAD CSV
# ============================================================

def load_records():

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"Input CSV not found:\n{INPUT_CSV}"
        )

    with open(
        INPUT_CSV,
        "r",
        encoding="utf-8-sig"
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


# ============================================================
# IMAGE PATH CHECK
# ============================================================

def get_image_path(record):

    relative = str(
        record.get(
            "image_path",
            ""
        )
    ).strip()

    if not relative:

        return None

    # image_path in metadata begins:
    # cleaned_real_ocr_v1/lines/...
    filename = Path(
        relative
    ).name

    return (
        LINES_DIR
        / filename
    )


# ============================================================
# MAIN FINALIZATION
# ============================================================

def finalize():

    print("=" * 72)
    print("FINALIZE CLEAN REAL OCR DATASET V1")
    print("=" * 72)

    records = load_records()

    print()
    print(
        "Input records:",
        len(records)
    )

    updated_count = 0

    split_rows_found = 0

    missing_split_labels = []

    unexpected_split_labels = []

    missing_images = []

    blank_ground_truth = []

    seen_split_ids = set()

    # ========================================================
    # UPDATE SPLIT LABELS
    # ========================================================

    for record in records:

        sample_id = str(
            record.get(
                "sample_id",
                ""
            )
        ).strip()

        cleaning_action = str(
            record.get(
                "cleaning_action",
                ""
            )
        ).strip()

        # ----------------------------------------------------
        # SPLIT RECORD
        # ----------------------------------------------------

        if (
            cleaning_action
            == "SPLIT_NEEDS_LABEL_REVIEW"
        ):

            split_rows_found += 1

            if sample_id not in SPLIT_LABELS:

                missing_split_labels.append(
                    sample_id
                )

            else:

                record[
                    "ground_truth"
                ] = SPLIT_LABELS[
                    sample_id
                ]

                record[
                    "cleaning_action"
                ] = (
                    "SPLIT_MANUALLY_VERIFIED"
                )

                updated_count += 1

                seen_split_ids.add(
                    sample_id
                )

        # ----------------------------------------------------
        # CHECK IMAGE EXISTS
        # ----------------------------------------------------

        image_path = (
            get_image_path(
                record
            )
        )

        if (
            image_path is None
            or not image_path.exists()
        ):

            missing_images.append(
                sample_id
            )

        # ----------------------------------------------------
        # CHECK GROUND TRUTH
        # ----------------------------------------------------

        ground_truth = str(
            record.get(
                "ground_truth",
                ""
            )
        ).strip()

        if not ground_truth:

            blank_ground_truth.append(
                sample_id
            )

    # ========================================================
    # CHECK MAPPING ENTRIES NOT PRESENT IN DATASET
    # ========================================================

    for sample_id in SPLIT_LABELS:

        if sample_id not in seen_split_ids:

            unexpected_split_labels.append(
                sample_id
            )

    # ========================================================
    # CRITICAL VALIDATION
    # ========================================================

    print()
    print(
        "Split rows found:",
        split_rows_found
    )

    print(
        "Split labels updated:",
        updated_count
    )

    print(
        "Blank ground-truth rows:",
        len(
            blank_ground_truth
        )
    )

    print(
        "Missing images:",
        len(
            missing_images
        )
    )

    if missing_split_labels:

        print()
        print(
            "ERROR: Missing labels for:"
        )

        for sample_id in missing_split_labels:

            print(
                " ",
                sample_id
            )

        raise RuntimeError(
            "Some split samples do not have "
            "manual ground-truth labels."
        )

    if unexpected_split_labels:

        print()
        print(
            "ERROR: These manual labels were not "
            "matched to split dataset rows:"
        )

        for sample_id in unexpected_split_labels:

            print(
                " ",
                sample_id
            )

        raise RuntimeError(
            "Split label mapping does not match dataset."
        )

    if blank_ground_truth:

        print()
        print(
            "ERROR: Blank ground_truth remains:"
        )

        for sample_id in blank_ground_truth:

            print(
                " ",
                sample_id
            )

        raise RuntimeError(
            "Dataset still contains blank labels."
        )

    if missing_images:

        print()
        print(
            "ERROR: Missing images:"
        )

        for sample_id in missing_images:

            print(
                " ",
                sample_id
            )

        raise RuntimeError(
            "Dataset contains missing image files."
        )

    # ========================================================
    # CHECK DUPLICATE WAS REMOVED
    # ========================================================

    duplicate_id = (
        "real_train_line_000045"
    )

    duplicate_found = any(

        str(
            row.get(
                "sample_id",
                ""
            )
        ).strip()
        == duplicate_id

        for row in records
    )

    if duplicate_found:

        raise RuntimeError(
            "Duplicate sample 000045 is still "
            "present in cleaned dataset."
        )

    # ========================================================
    # SAVE FINAL CSV
    # ========================================================

    if not records:

        raise RuntimeError(
            "No records found."
        )

    fieldnames = list(
        records[0].keys()
    )

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            records
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    verified_split_rows = [

        row
        for row in records

        if str(
            row.get(
                "cleaning_action",
                ""
            )
        ).strip()
        == "SPLIT_MANUALLY_VERIFIED"
    ]

    normal_rows = [

        row
        for row in records

        if str(
            row.get(
                "cleaning_action",
                ""
            )
        ).strip()
        in {
            "KEEP_ORIGINAL",
            "KEEP_MANUAL_CONFIRMED",
        }
    ]

    print()
    print("=" * 72)
    print("FINAL VALIDATION PASSED")
    print("=" * 72)

    print(
        "Final dataset rows:",
        len(records)
    )

    print(
        "Normal / kept rows:",
        len(
            normal_rows
        )
    )

    print(
        "Verified split rows:",
        len(
            verified_split_rows
        )
    )

    print(
        "Blank labels:",
        len(
            blank_ground_truth
        )
    )

    print(
        "Missing images:",
        len(
            missing_images
        )
    )

    print(
        "Duplicate 000045 present:",
        duplicate_found
    )

    print()
    print(
        "Verified split labels:"
    )

    for row in verified_split_rows:

        print()
        print(
            row[
                "sample_id"
            ]
        )

        print(
            "  GT:",
            row[
                "ground_truth"
            ]
        )

    print()
    print(
        "TRAINING-READY CSV:"
    )

    print(
        OUTPUT_CSV
    )

    print()
    print(
        "Image folder:"
    )

    print(
        LINES_DIR
    )

    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        finalize()

    except Exception as error:

        print()
        print(
            "ERROR:"
        )

        print(
            error
        )