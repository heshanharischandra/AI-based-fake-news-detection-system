from pathlib import Path
import json
import random

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

REAL_LABELS_CSV = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "real_manual_line_labels.csv"
)

OUTPUT_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "splits"
)

TRAIN_CSV = (
    OUTPUT_DIR
    / "real_finetune_train.csv"
)

VALIDATION_CSV = (
    OUTPUT_DIR
    / "real_finetune_validation.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "real_finetune_split_summary.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TRAIN_SOURCE_COUNT = 8


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "REAL OCR FINE-TUNING DATASET SPLIT"
    )

    print(
        "=" * 72
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not REAL_LABELS_CSV.exists():

        raise FileNotFoundError(
            f"Real labels CSV not found:\n"
            f"{REAL_LABELS_CSV}"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        REAL_LABELS_CSV,
        encoding="utf-8-sig"
    )

    df.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()

        for column in df.columns
    ]

    required_columns = {
        "sample_id",
        "image_path",
        "source_id",
        "ground_truth"
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing required columns: "
            f"{missing}"
        )

    # --------------------------------------------------------
    # Keep usable labeled rows
    # --------------------------------------------------------

    df["ground_truth"] = (
        df["ground_truth"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if "keep" in df.columns:

        keep_values = (
            df["keep"]
            .fillna(1)
            .astype(str)
            .str.strip()
        )

        df = df[
            keep_values.isin(
                ["1", "1.0", "True", "true"]
            )
        ].copy()

    df = df[
        df["ground_truth"].ne("")
    ].copy()

    # --------------------------------------------------------
    # Get source IDs
    # --------------------------------------------------------

    source_ids = sorted(
        df["source_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"\nTotal real lines  : "
        f"{len(df)}"
    )

    print(
        f"Total source images: "
        f"{len(source_ids)}"
    )

    if len(source_ids) <= TRAIN_SOURCE_COUNT:

        raise ValueError(
            "Not enough source images to create "
            "a separate validation set."
        )

    # --------------------------------------------------------
    # Shuffle source IDs only
    # --------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    shuffled_sources = (
        source_ids.copy()
    )

    rng.shuffle(
        shuffled_sources
    )

    train_sources = set(
        shuffled_sources[
            :TRAIN_SOURCE_COUNT
        ]
    )

    validation_sources = set(
        shuffled_sources[
            TRAIN_SOURCE_COUNT:
        ]
    )

    # --------------------------------------------------------
    # Create splits
    # --------------------------------------------------------

    train_df = df[
        df["source_id"]
        .isin(
            train_sources
        )
    ].copy()

    validation_df = df[
        df["source_id"]
        .isin(
            validation_sources
        )
    ].copy()

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    train_source_set = set(
        train_df["source_id"]
    )

    validation_source_set = set(
        validation_df["source_id"]
    )

    overlap = (
        train_source_set
        & validation_source_set
    )

    if overlap:

        raise RuntimeError(
            f"Source leakage detected: "
            f"{overlap}"
        )

    # --------------------------------------------------------
    # Sample ID overlap check
    # --------------------------------------------------------

    sample_overlap = (
        set(train_df["sample_id"])
        & set(validation_df["sample_id"])
    )

    if sample_overlap:

        raise RuntimeError(
            "Sample ID leakage detected."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_df.to_csv(
        TRAIN_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    validation_df.to_csv(
        VALIDATION_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "random_seed":
            RANDOM_SEED,

        "total_samples":
            len(df),

        "total_sources":
            len(source_ids),

        "train_sources":
            sorted(
                train_sources
            ),

        "validation_sources":
            sorted(
                validation_sources
            ),

        "train_source_count":
            len(
                train_sources
            ),

        "validation_source_count":
            len(
                validation_sources
            ),

        "train_samples":
            len(
                train_df
            ),

        "validation_samples":
            len(
                validation_df
            ),

        "source_leakage":
            False,

        "sample_leakage":
            False,

        "external_holdout_note":
            (
                "The original A9 real news image remains "
                "outside this dataset as an external "
                "real-world holdout test."
            )
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
    # OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "REAL FINE-TUNING SPLIT COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nTrain source images      : "
        f"{len(train_sources)}"
    )

    print(
        f"Validation source images : "
        f"{len(validation_sources)}"
    )

    print(
        f"Train real lines         : "
        f"{len(train_df)}"
    )

    print(
        f"Validation real lines    : "
        f"{len(validation_df)}"
    )

    print(
        "\nTrain sources:"
    )

    for source_id in sorted(
        train_sources
    ):

        print(
            f"  {source_id}"
        )

    print(
        "\nValidation sources:"
    )

    for source_id in sorted(
        validation_sources
    ):

        print(
            f"  {source_id}"
        )

    print(
        "\nSource leakage check : PASSED"
    )

    print(
        "Sample leakage check : PASSED"
    )

    print(
        "\nFiles saved:"
    )

    print(
        TRAIN_CSV
    )

    print(
        VALIDATION_CSV
    )

    print(
        SUMMARY_JSON
    )