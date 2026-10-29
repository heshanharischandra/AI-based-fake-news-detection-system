from pathlib import Path
import csv
import json


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

CLEAN_ROOT = (
    OCR_ROOT
    / "cleaned_real_ocr_v1"
)

INPUT_CSV = (
    CLEAN_ROOT
    / "metadata"
    / "cleaned_real_labels_v1_final.csv"
)

SPLIT_DIR = (
    CLEAN_ROOT
    / "metadata"
    / "splits"
)

TRAIN_CSV = (
    SPLIT_DIR
    / "clean_real_train_v1.csv"
)

VALIDATION_CSV = (
    SPLIT_DIR
    / "clean_real_validation_v1.csv"
)

SUMMARY_JSON = (
    SPLIT_DIR
    / "clean_real_split_summary_v1.json"
)


# ============================================================
# DEVELOPMENT VALIDATION SOURCES
#
# Preserve the same source-level validation set used
# previously so V3 vs cleaned-data model can be compared.
# ============================================================

VALIDATION_SOURCES = {
    "real_train_001",
    "real_train_004",
    "real_train_008",
    "real_train_009",
    "real_train_022",
    "real_train_025",
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
            csv.DictReader(file)
        )


# ============================================================
# IMAGE PATH CHECK
# ============================================================

def resolve_image_path(record):

    relative_path = str(
        record.get(
            "image_path",
            ""
        )
    ).strip()

    if not relative_path:
        return None

    return (
        OCR_ROOT
        / Path(relative_path)
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    path,
    records,
    fieldnames
):

    with open(
        path,
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


# ============================================================
# MAIN SPLIT
# ============================================================

def create_split():

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    records = load_records()

    if not records:

        raise RuntimeError(
            "Clean dataset contains no records."
        )

    print("=" * 72)
    print("CLEAN REAL OCR SOURCE-LEVEL SPLIT V1")
    print("=" * 72)

    print()
    print(
        "Input records:",
        len(records)
    )

    train_records = []

    validation_records = []

    missing_images = []

    blank_labels = []

    # ========================================================
    # ASSIGN BY SOURCE
    # ========================================================

    for record in records:

        sample_id = str(
            record.get(
                "sample_id",
                ""
            )
        ).strip()

        source_id = str(
            record.get(
                "source_id",
                ""
            )
        ).strip()

        ground_truth = str(
            record.get(
                "ground_truth",
                ""
            )
        ).strip()

        # ----------------------------------------------------
        # LABEL CHECK
        # ----------------------------------------------------

        if not ground_truth:

            blank_labels.append(
                sample_id
            )

        # ----------------------------------------------------
        # IMAGE CHECK
        # ----------------------------------------------------

        image_path = (
            resolve_image_path(
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
        # SOURCE-LEVEL SPLIT
        # ----------------------------------------------------

        if source_id in VALIDATION_SOURCES:

            validation_records.append(
                record
            )

        else:

            train_records.append(
                record
            )

    # ========================================================
    # CRITICAL DATA CHECKS
    # ========================================================

    if blank_labels:

        print()
        print(
            "ERROR: Blank labels found:"
        )

        for item in blank_labels:

            print(
                " ",
                item
            )

        raise RuntimeError(
            "Cannot split dataset with blank labels."
        )

    if missing_images:

        print()
        print(
            "ERROR: Missing images found:"
        )

        for item in missing_images:

            print(
                " ",
                item
            )

        raise RuntimeError(
            "Cannot split dataset with missing images."
        )

    # ========================================================
    # LEAKAGE CHECK
    # ========================================================

    train_sources = {
        row["source_id"].strip()
        for row in train_records
    }

    validation_sources_found = {
        row["source_id"].strip()
        for row in validation_records
    }

    overlap = (
        train_sources
        & validation_sources_found
    )

    if overlap:

        print()
        print(
            "ERROR: SOURCE LEAKAGE DETECTED:"
        )

        for source in sorted(overlap):

            print(
                " ",
                source
            )

        raise RuntimeError(
            "Train / validation source leakage detected."
        )

    # ========================================================
    # SAMPLE-ID LEAKAGE CHECK
    # ========================================================

    train_ids = {
        row["sample_id"].strip()
        for row in train_records
    }

    validation_ids = {
        row["sample_id"].strip()
        for row in validation_records
    }

    sample_overlap = (
        train_ids
        & validation_ids
    )

    if sample_overlap:

        raise RuntimeError(
            "Sample IDs appear in both "
            "train and validation."
        )

    # ========================================================
    # ALL EXPECTED VALIDATION SOURCES PRESENT?
    # ========================================================

    missing_validation_sources = (
        VALIDATION_SOURCES
        - validation_sources_found
    )

    if missing_validation_sources:

        print()
        print(
            "WARNING: Expected validation sources "
            "not found:"
        )

        for source in sorted(
            missing_validation_sources
        ):

            print(
                " ",
                source
            )

    # ========================================================
    # SAVE SPLITS
    # ========================================================

    fieldnames = list(
        records[0].keys()
    )

    save_csv(
        TRAIN_CSV,
        train_records,
        fieldnames
    )

    save_csv(
        VALIDATION_CSV,
        validation_records,
        fieldnames
    )

    # ========================================================
    # SOURCE COUNTS
    # ========================================================

    train_source_counts = {}

    for record in train_records:

        source = (
            record[
                "source_id"
            ].strip()
        )

        train_source_counts[
            source
        ] = (
            train_source_counts.get(
                source,
                0
            )
            + 1
        )

    validation_source_counts = {}

    for record in validation_records:

        source = (
            record[
                "source_id"
            ].strip()
        )

        validation_source_counts[
            source
        ] = (
            validation_source_counts.get(
                source,
                0
            )
            + 1
        )

    # ========================================================
    # SAVE SUMMARY JSON
    # ========================================================

    summary = {

        "dataset":
            "cleaned_real_ocr_v1",

        "total_records":
            len(records),

        "train_records":
            len(train_records),

        "validation_records":
            len(validation_records),

        "train_source_count":
            len(train_sources),

        "validation_source_count":
            len(validation_sources_found),

        "train_sources":
            sorted(
                train_sources
            ),

        "validation_sources":
            sorted(
                validation_sources_found
            ),

        "train_source_counts":
            train_source_counts,

        "validation_source_counts":
            validation_source_counts,

        "source_overlap":
            sorted(
                overlap
            ),

        "sample_overlap_count":
            len(
                sample_overlap
            ),

        "blank_label_count":
            len(
                blank_labels
            ),

        "missing_image_count":
            len(
                missing_images
            ),
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

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 72)
    print("SOURCE-LEVEL SPLIT VALIDATION PASSED")
    print("=" * 72)

    print(
        "Total records:",
        len(records)
    )

    print(
        "Training records:",
        len(train_records)
    )

    print(
        "Validation records:",
        len(validation_records)
    )

    print()
    print(
        "Training sources:",
        len(train_sources)
    )

    for source in sorted(
        train_sources
    ):

        print(
            f"  {source}: "
            f"{train_source_counts[source]} samples"
        )

    print()
    print(
        "Validation sources:",
        len(validation_sources_found)
    )

    for source in sorted(
        validation_sources_found
    ):

        print(
            f"  {source}: "
            f"{validation_source_counts[source]} samples"
        )

    print()
    print(
        "Source leakage:",
        len(overlap)
    )

    print(
        "Sample leakage:",
        len(sample_overlap)
    )

    print(
        "Blank labels:",
        len(blank_labels)
    )

    print(
        "Missing images:",
        len(missing_images)
    )

    print()
    print(
        "Training CSV:"
    )

    print(
        TRAIN_CSV
    )

    print()
    print(
        "Validation CSV:"
    )

    print(
        VALIDATION_CSV
    )

    print()
    print(
        "Summary JSON:"
    )

    print(
        SUMMARY_JSON
    )

    print("=" * 72)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This validation set is for development/model "
        "comparison."
    )

    print(
        "A fresh unseen test set should still be collected "
        "for final research evaluation."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        create_split()

    except Exception as error:

        print()
        print(
            "ERROR:"
        )

        print(
            error
        )