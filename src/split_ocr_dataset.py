from pathlib import Path
from datetime import datetime

import json
import unicodedata

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

METADATA_DIR = (
    OCR_ROOT
    / "metadata"
)

SYNTHETIC_LABELS = (
    METADATA_DIR
    / "synthetic_labels.csv"
)

REAL_LABELS = (
    METADATA_DIR
    / "real_labels.csv"
)

SPLIT_DIR = (
    METADATA_DIR
    / "splits"
)

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAIN_CSV = (
    SPLIT_DIR
    / "train.csv"
)

VALIDATION_CSV = (
    SPLIT_DIR
    / "validation.csv"
)

TEST_CSV = (
    SPLIT_DIR
    / "test.csv"
)

REAL_HOLDOUT_CSV = (
    SPLIT_DIR
    / "real_holdout.csv"
)

SUMMARY_JSON = (
    SPLIT_DIR
    / "split_summary.json"
)


RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10


# ============================================================
# 2. TEXT NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value)

    value = value.replace(
        "\ufeff",
        ""
    )

    value = unicodedata.normalize(
        "NFC",
        value
    )

    return value.strip()


# ============================================================
# 3. CLEAN CSV COLUMN NAMES
# ============================================================

def clean_column_names(df):

    cleaned_columns = []

    for column in df.columns:

        clean_column = (
            str(column)
            .replace("\ufeff", "")
            .strip()
        )

        cleaned_columns.append(
            clean_column
        )

    df.columns = cleaned_columns

    return df


# ============================================================
# 4. CLEAN IMPORTANT TEXT COLUMNS
# ============================================================

def clean_dataframe_values(df):

    text_columns = [
        "sample_id",
        "image_path",
        "ground_truth",
        "language",
        "source_id",
        "source",
        "source_text_id"
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .apply(
                    normalize_text
                )
            )

    return df


# ============================================================
# 5. LOAD SYNTHETIC DATA
# ============================================================

def load_synthetic_data():

    if not SYNTHETIC_LABELS.exists():

        raise FileNotFoundError(
            f"Synthetic labels not found:\n"
            f"{SYNTHETIC_LABELS}"
        )

    df = pd.read_csv(
        SYNTHETIC_LABELS,
        encoding="utf-8-sig"
    )

    df = clean_column_names(
        df
    )

    df = clean_dataframe_values(
        df
    )

    print(
        "\nSynthetic CSV columns:"
    )

    for column in df.columns:
        print(
            f"  - {column}"
        )

    required_columns = {
        "sample_id",
        "image_path",
        "ground_truth",
        "language"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing synthetic columns: "
            f"{missing_columns}"
        )

    # Remove invalid empty rows
    df = df[
        df["sample_id"].ne("")
        & df["image_path"].ne("")
        & df["ground_truth"].ne("")
    ].copy()

    df["dataset_type"] = (
        "synthetic"
    )

    return df


# ============================================================
# 6. LOAD REAL DATA
# ============================================================

def load_real_data():

    if not REAL_LABELS.exists():

        raise FileNotFoundError(
            f"Real labels not found:\n"
            f"{REAL_LABELS}"
        )

    df = pd.read_csv(
        REAL_LABELS,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # IMPORTANT FIX:
    # Remove BOM / spaces / invisible header characters
    # --------------------------------------------------------

    df = clean_column_names(
        df
    )

    df = clean_dataframe_values(
        df
    )

    print(
        "\nReal CSV columns:"
    )

    for column in df.columns:
        print(
            f"  - {column}"
        )

    required_columns = {
        "sample_id",
        "image_path",
        "ground_truth",
        "language",
        "source_id"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing real columns: "
            f"{missing_columns}"
        )

    # Remove invalid empty rows
    df = df[
        df["sample_id"].ne("")
        & df["image_path"].ne("")
        & df["ground_truth"].ne("")
    ].copy()

    df["dataset_type"] = (
        "real"
    )

    return df


# ============================================================
# 7. SPLIT SYNTHETIC DATA
# ============================================================

def split_synthetic_data(df):

    # Reproducible shuffle
    df = df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(
        drop=True
    )

    total_samples = len(
        df
    )

    train_count = int(
        total_samples
        * TRAIN_RATIO
    )

    validation_count = int(
        total_samples
        * VALIDATION_RATIO
    )

    test_count = (
        total_samples
        - train_count
        - validation_count
    )

    train_df = df.iloc[
        :train_count
    ].copy()

    validation_df = df.iloc[
        train_count:
        train_count + validation_count
    ].copy()

    test_df = df.iloc[
        train_count + validation_count:
    ].copy()

    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# 8. CHECK SAMPLE-ID OVERLAP
# ============================================================

def check_sample_id_overlap(
    train_df,
    validation_df,
    test_df
):

    train_ids = set(
        train_df[
            "sample_id"
        ]
    )

    validation_ids = set(
        validation_df[
            "sample_id"
        ]
    )

    test_ids = set(
        test_df[
            "sample_id"
        ]
    )

    train_val_overlap = (
        train_ids
        & validation_ids
    )

    train_test_overlap = (
        train_ids
        & test_ids
    )

    val_test_overlap = (
        validation_ids
        & test_ids
    )

    if (
        train_val_overlap
        or train_test_overlap
        or val_test_overlap
    ):

        raise RuntimeError(
            "Sample ID leakage detected "
            "between train/validation/test."
        )


# ============================================================
# 9. CHECK TEXT OVERLAP
# ============================================================

def check_text_overlap(
    train_df,
    validation_df,
    test_df
):

    train_texts = set(
        train_df[
            "ground_truth"
        ]
    )

    validation_texts = set(
        validation_df[
            "ground_truth"
        ]
    )

    test_texts = set(
        test_df[
            "ground_truth"
        ]
    )

    train_val_overlap = (
        train_texts
        & validation_texts
    )

    train_test_overlap = (
        train_texts
        & test_texts
    )

    val_test_overlap = (
        validation_texts
        & test_texts
    )

    if (
        train_val_overlap
        or train_test_overlap
        or val_test_overlap
    ):

        raise RuntimeError(
            "Ground-truth text leakage detected "
            "between train/validation/test."
        )


# ============================================================
# 10. SAVE SPLIT
# ============================================================

def save_split(
    dataframe,
    output_path
):

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 11. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "SINHALA OCR DATASET SPLITTING"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    synthetic_df = (
        load_synthetic_data()
    )

    real_df = (
        load_real_data()
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DATASET COUNTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSynthetic samples: "
        f"{len(synthetic_df)}"
    )

    print(
        f"Real samples: "
        f"{len(real_df)}"
    )

    # --------------------------------------------------------
    # Split synthetic dataset
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df
    ) = split_synthetic_data(
        synthetic_df
    )

    # --------------------------------------------------------
    # Leakage checks
    # --------------------------------------------------------

    check_sample_id_overlap(
        train_df,
        validation_df,
        test_df
    )

    check_text_overlap(
        train_df,
        validation_df,
        test_df
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # All current real samples come from one source image.
    # Therefore they remain together as a real holdout set.
    # They are NOT used for training.
    # --------------------------------------------------------

    real_source_count = (
        real_df[
            "source_id"
        ]
        .nunique()
    )

    # --------------------------------------------------------
    # Save CSV files
    # --------------------------------------------------------

    save_split(
        train_df,
        TRAIN_CSV
    )

    save_split(
        validation_df,
        VALIDATION_CSV
    )

    save_split(
        test_df,
        TEST_CSV
    )

    save_split(
        real_df,
        REAL_HOLDOUT_CSV
    )

    # --------------------------------------------------------
    # Save summary JSON
    # --------------------------------------------------------

    summary = {

        "created_at":
            datetime.now().isoformat(),

        "random_seed":
            RANDOM_SEED,

        "synthetic_total":
            len(
                synthetic_df
            ),

        "train_samples":
            len(
                train_df
            ),

        "validation_samples":
            len(
                validation_df
            ),

        "test_samples":
            len(
                test_df
            ),

        "real_holdout_samples":
            len(
                real_df
            ),

        "real_source_images":
            int(
                real_source_count
            ),

        "train_ratio":
            TRAIN_RATIO,

        "validation_ratio":
            VALIDATION_RATIO,

        "test_ratio":
            TEST_RATIO,

        "sample_id_leakage":
            False,

        "ground_truth_text_leakage":
            False,

        "real_holdout_used_for_training":
            False,

        "real_holdout_note":
            (
                "The current real OCR samples originate "
                "from one source news image and are kept "
                "together as an external real-image holdout."
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
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "OCR DATASET SPLIT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTrain samples      : "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples : "
        f"{len(validation_df)}"
    )

    print(
        f"Test samples       : "
        f"{len(test_df)}"
    )

    print(
        f"Real holdout       : "
        f"{len(real_df)}"
    )

    print(
        f"Real source images : "
        f"{real_source_count}"
    )

    print(
        "\nSample ID leakage  : PASSED"
    )

    print(
        "Text leakage       : PASSED"
    )

    print(
        "Real holdout       : NOT USED FOR TRAINING"
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
        TEST_CSV
    )

    print(
        REAL_HOLDOUT_CSV
    )

    print(
        SUMMARY_JSON
    )