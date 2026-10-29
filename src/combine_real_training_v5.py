from pathlib import Path
import csv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

OLD_TRAIN_CSV = (
    OCR_ROOT
    / "cleaned_real_ocr_v1"
    / "metadata"
    / "splits"
    / "clean_real_train_v1.csv"
)

V5_TRAIN_CSV = (
    OCR_ROOT
    / "real_training_expansion_v5"
    / "metadata"
    / "v5_verified_real_lines.csv"
)

OUTPUT_DIR = (
    OCR_ROOT
    / "real_training_expansion_v5"
    / "combined_v5"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "combined_real_train_v5.csv"
)


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(path):

    if not path.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


# ============================================================
# NORMALIZE PATH
# ============================================================

def normalize_to_ocr_relative(
    image_path_text,
    source_type,
):

    image_path_text = (
        str(
            image_path_text
        )
        .strip()
        .replace("\\", "/")
    )

    if not image_path_text:

        raise ValueError(
            "Blank image path."
        )

    # --------------------------------------------------------
    # OLD CLEAN DATA
    #
    # Example:
    # cleaned_real_ocr_v1/lines/...
    # This is already relative to OCR_ROOT.
    # --------------------------------------------------------

    if source_type == "old":

        full_path = (
            OCR_ROOT
            / Path(
                image_path_text
            )
        )

    # --------------------------------------------------------
    # NEW V5 DATA
    #
    # Example:
    # datasets/sinhala_ocr/real_training_expansion_v5/...
    # This is relative to PROJECT_ROOT.
    # --------------------------------------------------------

    elif source_type == "v5":

        full_path = (
            PROJECT_ROOT
            / Path(
                image_path_text
            )
        )

    else:

        raise ValueError(
            f"Unknown source_type: "
            f"{source_type}"
        )

    full_path = full_path.resolve()

    if not full_path.exists():

        raise FileNotFoundError(
            f"Image file not found:\n"
            f"{full_path}"
        )

    try:

        relative_path = (
            full_path
            .relative_to(
                OCR_ROOT.resolve()
            )
        )

    except ValueError:

        raise RuntimeError(
            "Image is outside OCR_ROOT:\n"
            f"{full_path}"
        )

    return str(
        relative_path
    ).replace(
        "\\",
        "/"
    )


# ============================================================
# PREPARE OLD RECORDS
# ============================================================

def prepare_old_records(
    rows
):

    prepared = []

    for row in rows:

        sample_id = str(
            row.get(
                "sample_id",
                ""
            )
        ).strip()

        source_id = str(
            row.get(
                "source_id",
                ""
            )
        ).strip()

        ground_truth = str(
            row.get(
                "ground_truth",
                ""
            )
        ).strip()

        language = str(
            row.get(
                "language",
                "si"
            )
        ).strip()

        image_path = (
            normalize_to_ocr_relative(
                row.get(
                    "image_path",
                    ""
                ),
                "old",
            )
        )

        if not sample_id:

            raise RuntimeError(
                "Old dataset has blank sample_id."
            )

        if not source_id:

            raise RuntimeError(
                f"{sample_id}: "
                "blank source_id."
            )

        if not ground_truth:

            raise RuntimeError(
                f"{sample_id}: "
                "blank ground_truth."
            )

        prepared.append(
            {
                "sample_id":
                    sample_id,

                "source_id":
                    source_id,

                "image_path":
                    image_path,

                "ground_truth":
                    ground_truth,

                "language":
                    language or "si",

                "data_origin":
                    "clean_real_train_v1",
            }
        )

    return prepared


# ============================================================
# PREPARE V5 RECORDS
# ============================================================

def prepare_v5_records(
    rows
):

    prepared = []

    for row in rows:

        sample_id = str(
            row.get(
                "sample_id",
                ""
            )
        ).strip()

        source_id = str(
            row.get(
                "source_id",
                ""
            )
        ).strip()

        ground_truth = str(
            row.get(
                "text",
                ""
            )
        ).strip()

        language = str(
            row.get(
                "language",
                "si"
            )
        ).strip()

        image_path = (
            normalize_to_ocr_relative(
                row.get(
                    "image_path",
                    ""
                ),
                "v5",
            )
        )

        if not sample_id:

            raise RuntimeError(
                "V5 dataset has blank sample_id."
            )

        if not source_id:

            raise RuntimeError(
                f"{sample_id}: "
                "blank source_id."
            )

        if not ground_truth:

            raise RuntimeError(
                f"{sample_id}: "
                "blank ground_truth."
            )

        prepared.append(
            {
                "sample_id":
                    sample_id,

                "source_id":
                    source_id,

                "image_path":
                    image_path,

                "ground_truth":
                    ground_truth,

                "language":
                    language or "si",

                "data_origin":
                    "v5_real_manual_verified",
            }
        )

    return prepared


# ============================================================
# VALIDATE COMBINED DATA
# ============================================================

def validate_combined(
    records
):

    sample_ids = set()

    duplicate_ids = []

    blank_labels = []

    missing_images = []

    old_count = 0

    v5_count = 0

    for record in records:

        sample_id = (
            record[
                "sample_id"
            ]
        )

        if sample_id in sample_ids:

            duplicate_ids.append(
                sample_id
            )

        sample_ids.add(
            sample_id
        )

        if not record[
            "ground_truth"
        ].strip():

            blank_labels.append(
                sample_id
            )

        full_path = (
            OCR_ROOT
            / Path(
                record[
                    "image_path"
                ]
            )
        )

        if not full_path.exists():

            missing_images.append(
                str(
                    full_path
                )
            )

        if (
            record[
                "data_origin"
            ]
            == "clean_real_train_v1"
        ):

            old_count += 1

        elif (
            record[
                "data_origin"
            ]
            == "v5_real_manual_verified"
        ):

            v5_count += 1

    print()
    print("=" * 72)
    print("VALIDATION")
    print("=" * 72)

    print(
        "Old clean records:",
        old_count,
    )

    print(
        "New V5 records:",
        v5_count,
    )

    print(
        "Combined records:",
        len(
            records
        ),
    )

    print(
        "Duplicate sample IDs:",
        len(
            duplicate_ids
        ),
    )

    print(
        "Blank labels:",
        len(
            blank_labels
        ),
    )

    print(
        "Missing image files:",
        len(
            missing_images
        ),
    )

    if old_count != 76:

        raise RuntimeError(
            f"Expected 76 old clean records, "
            f"found {old_count}."
        )

    if v5_count != 45:

        raise RuntimeError(
            f"Expected 45 V5 records, "
            f"found {v5_count}."
        )

    if len(
        records
    ) != 121:

        raise RuntimeError(
            f"Expected 121 combined records, "
            f"found {len(records)}."
        )

    if duplicate_ids:

        raise RuntimeError(
            "Duplicate sample IDs found:\n"
            + "\n".join(
                duplicate_ids
            )
        )

    if blank_labels:

        raise RuntimeError(
            "Blank labels found:\n"
            + "\n".join(
                blank_labels
            )
        )

    if missing_images:

        raise RuntimeError(
            "Missing images found:\n"
            + "\n".join(
                missing_images
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("COMBINE REAL SINHALA OCR TRAINING DATA V5")
    print("=" * 72)

    old_rows = load_csv(
        OLD_TRAIN_CSV
    )

    v5_rows = load_csv(
        V5_TRAIN_CSV
    )

    print()
    print(
        "Old clean CSV rows:",
        len(
            old_rows
        ),
    )

    print(
        "V5 verified CSV rows:",
        len(
            v5_rows
        ),
    )

    old_records = (
        prepare_old_records(
            old_rows
        )
    )

    v5_records = (
        prepare_v5_records(
            v5_rows
        )
    )

    combined = (
        old_records
        + v5_records
    )

    validate_combined(
        combined
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "sample_id",
        "source_id",
        "image_path",
        "ground_truth",
        "language",
        "data_origin",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            combined
        )

    print()
    print("=" * 72)
    print("COMBINATION PASSED")
    print("=" * 72)

    print()
    print(
        "Combined unique real lines:",
        len(
            combined
        ),
    )

    print()
    print(
        "Output CSV:"
    )

    print(
        OUTPUT_CSV
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The existing 23-line validation set "
        "was NOT modified."
    )

    print(
        "Fresh unseen images were NOT added."
    )

    print("=" * 72)


if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print("ERROR:")
        print(error)