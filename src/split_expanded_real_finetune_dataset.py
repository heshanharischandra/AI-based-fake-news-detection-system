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
    / "splits_v2"
)

TRAIN_CSV = (
    OUTPUT_DIR
    / "real_finetune_train_v2.csv"
)

VALIDATION_CSV = (
    OUTPUT_DIR
    / "real_finetune_validation_v2.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "real_finetune_split_v2_summary.json"
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.80


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 72)
    print("EXPANDED REAL OCR FINE-TUNING DATASET SPLIT")
    print("=" * 72)

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
    # Load
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
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Keep usable rows
    # --------------------------------------------------------

    df["ground_truth"] = (
        df["ground_truth"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[
        df["ground_truth"].ne("")
    ].copy()

    if "keep" in df.columns:

        keep_values = (
            df["keep"]
            .fillna(1)
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df = df[
            keep_values.isin(
                [
                    "1",
                    "1.0",
                    "true"
                ]
            )
        ].copy()

    # --------------------------------------------------------
    # Source IDs
    # --------------------------------------------------------

    source_ids = sorted(
        df["source_id"]
        .astype(str)
        .unique()
        .tolist()
    )

    total_sources = len(
        source_ids
    )

    total_samples = len(
        df
    )

    print(
        f"\nTotal real lines   : "
        f"{total_samples}"
    )

    print(
        f"Total source images: "
        f"{total_sources}"
    )

    if total_sources < 5:

        raise ValueError(
            "Not enough source images "
            "for source-level splitting."
        )

    # --------------------------------------------------------
    # Source-level 80/20 split
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

    train_source_count = int(
        total_sources
        * TRAIN_RATIO
    )

    train_source_count = max(
        1,
        min(
            train_source_count,
            total_sources - 1
        )
    )

    train_sources = set(
        shuffled_sources[
            :train_source_count
        ]
    )

    validation_sources = set(
        shuffled_sources[
            train_source_count:
        ]
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_df = df[
        df["source_id"]
        .astype(str)
        .isin(
            train_sources
        )
    ].copy()

    validation_df = df[
        df["source_id"]
        .astype(str)
        .isin(
            validation_sources
        )
    ].copy()

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    source_overlap = (
        set(
            train_df["source_id"]
            .astype(str)
        )
        &
        set(
            validation_df["source_id"]
            .astype(str)
        )
    )

    sample_overlap = (
        set(
            train_df["sample_id"]
            .astype(str)
        )
        &
        set(
            validation_df["sample_id"]
            .astype(str)
        )
    )

    if source_overlap:

        raise RuntimeError(
            f"Source leakage detected: "
            f"{source_overlap}"
        )

    if sample_overlap:

        raise RuntimeError(
            f"Sample leakage detected: "
            f"{sample_overlap}"
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
    # Source distributions
    # --------------------------------------------------------

    train_distribution = (
        train_df[
            "source_id"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    validation_distribution = (
        validation_df[
            "source_id"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "random_seed":
            RANDOM_SEED,

        "train_ratio":
            TRAIN_RATIO,

        "total_samples":
            total_samples,

        "total_sources":
            total_sources,

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

        "train_sources":
            sorted(
                train_sources
            ),

        "validation_sources":
            sorted(
                validation_sources
            ),

        "train_distribution":
            train_distribution,

        "validation_distribution":
            validation_distribution,

        "source_leakage":
            False,

        "sample_leakage":
            False,

        "external_holdout_note":
            (
                "The original A9 news image remains "
                "outside both train and validation "
                "as an external real-world holdout."
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
        "EXPANDED REAL SPLIT COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nTrain sources      : "
        f"{len(train_sources)}"
    )

    print(
        f"Validation sources : "
        f"{len(validation_sources)}"
    )

    print(
        f"Train real lines   : "
        f"{len(train_df)}"
    )

    print(
        f"Validation lines   : "
        f"{len(validation_df)}"
    )

    print(
        "\nTrain sources:"
    )

    for source in sorted(
        train_sources
    ):

        print(
            f"  {source}"
        )

    print(
        "\nValidation sources:"
    )

    for source in sorted(
        validation_sources
    ):

        print(
            f"  {source}"
        )

    print(
        "\nSource leakage check : PASSED"
    )

    print(
        "Sample leakage check : PASSED"
    )

    print(
        "\nSaved files:"
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