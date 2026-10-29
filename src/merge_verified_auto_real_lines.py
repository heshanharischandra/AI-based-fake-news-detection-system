from pathlib import Path
from datetime import datetime

import json
import shutil

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

METADATA_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
)

VERIFIED_CSV = (
    METADATA_DIR
    / "auto_line_labels_verified.csv"
)

CLEAN_REAL_CSV = (
    METADATA_DIR
    / "real_manual_line_labels.csv"
)

REAL_LINE_DIR = (
    OCR_ROOT
    / "real_training_lines"
)

BACKUP_DIR = (
    METADATA_DIR
    / "before_auto_merge_backup"
)

BACKUP_CSV = (
    BACKUP_DIR
    / "real_manual_line_labels_before_auto_merge.csv"
)

MERGE_MAP_CSV = (
    METADATA_DIR
    / "auto_line_merge_map.csv"
)

MERGE_SUMMARY_JSON = (
    METADATA_DIR
    / "auto_line_merge_summary.json"
)


# ============================================================
# HELPERS
# ============================================================

def clean_column_names(df):

    df.columns = [
        str(column)
        .replace("\ufeff", "")
        .strip()

        for column in df.columns
    ]

    return df


def get_next_sample_number(df):

    numbers = []

    if "sample_id" not in df.columns:

        return 1

    for value in df["sample_id"].astype(str):

        value = value.strip()

        if not value.startswith(
            "real_train_line_"
        ):

            continue

        try:

            number = int(
                value.replace(
                    "real_train_line_",
                    ""
                )
            )

            numbers.append(
                number
            )

        except ValueError:

            continue

    if not numbers:

        return 1

    return max(numbers) + 1


def normalize_keep(value):

    value = str(
        value
    ).strip().lower()

    return value in {
        "1",
        "1.0",
        "true"
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "MERGE VERIFIED AUTO REAL OCR LINES"
    )

    print(
        "=" * 72
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not VERIFIED_CSV.exists():

        raise FileNotFoundError(
            f"Verified CSV not found:\n"
            f"{VERIFIED_CSV}"
        )

    if not CLEAN_REAL_CSV.exists():

        raise FileNotFoundError(
            f"Clean real CSV not found:\n"
            f"{CLEAN_REAL_CSV}"
        )

    REAL_LINE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Prevent accidental second merge
    # --------------------------------------------------------

    if MERGE_MAP_CSV.exists():

        existing_map = pd.read_csv(
            MERGE_MAP_CSV,
            encoding="utf-8-sig"
        )

        if len(existing_map) > 0:

            print(
                "\nMerge map already exists."
            )

            print(
                "This indicates the verified auto lines "
                "may already have been merged."
            )

            print(
                "\nNo changes were made."
            )

            raise SystemExit(
                0
            )

    # --------------------------------------------------------
    # Load clean dataset
    # --------------------------------------------------------

    clean_df = pd.read_csv(
        CLEAN_REAL_CSV,
        encoding="utf-8-sig"
    )

    clean_df = clean_column_names(
        clean_df
    )

    original_clean_count = len(
        clean_df
    )

    # --------------------------------------------------------
    # Load verified new dataset
    # --------------------------------------------------------

    verified_df = pd.read_csv(
        VERIFIED_CSV,
        encoding="utf-8-sig"
    )

    verified_df = clean_column_names(
        verified_df
    )

    print(
        f"\nExisting clean rows : "
        f"{original_clean_count}"
    )

    print(
        f"Verified new rows   : "
        f"{len(verified_df)}"
    )

    # --------------------------------------------------------
    # Keep only verified accepted rows
    # --------------------------------------------------------

    verified_df[
        "ground_truth"
    ] = (
        verified_df[
            "ground_truth"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    accepted_mask = (
        verified_df[
            "review_status"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(
            "accepted"
        )
    )

    keep_mask = (
        verified_df[
            "keep"
        ]
        .map(
            normalize_keep
        )
    )

    label_mask = (
        verified_df[
            "ground_truth"
        ]
        .ne("")
    )

    merge_df = verified_df[
        accepted_mask
        & keep_mask
        & label_mask
    ].copy()

    print(
        f"Rows eligible to merge: "
        f"{len(merge_df)}"
    )

    if len(merge_df) != 62:

        raise ValueError(
            "Expected 62 verified accepted rows, "
            f"but found {len(merge_df)}."
        )

    # --------------------------------------------------------
    # Backup clean CSV
    # --------------------------------------------------------

    shutil.copy2(
        CLEAN_REAL_CSV,
        BACKUP_CSV
    )

    print(
        "\nBackup created:"
    )

    print(
        BACKUP_CSV
    )

    # --------------------------------------------------------
    # Determine next clean sample ID
    # --------------------------------------------------------

    next_sample_number = (
        get_next_sample_number(
            clean_df
        )
    )

    print(
        f"\nNext clean sample number: "
        f"{next_sample_number}"
    )

    # --------------------------------------------------------
    # Ensure required clean columns
    # --------------------------------------------------------

    required_columns = [
        "sample_id",
        "image_path",
        "source_id",
        "line_number",
        "ground_truth",
        "keep"
    ]

    for column in required_columns:

        if column not in clean_df.columns:

            clean_df[
                column
            ] = ""

    # --------------------------------------------------------
    # Create new clean rows
    # --------------------------------------------------------

    new_rows = []

    merge_map = []

    copied_files = []

    try:

        for _, row in merge_df.iterrows():

            old_sample_id = str(
                row[
                    "sample_id"
                ]
            ).strip()

            old_relative_path = str(
                row[
                    "image_path"
                ]
            ).strip()

            source_id = str(
                row[
                    "source_id"
                ]
            ).strip()

            ground_truth = str(
                row[
                    "ground_truth"
                ]
            ).strip()

            line_number = row.get(
                "line_number",
                ""
            )

            old_image_path = (
                OCR_ROOT
                / old_relative_path
            )

            if not old_image_path.exists():

                raise FileNotFoundError(
                    f"Source image missing:\n"
                    f"{old_image_path}"
                )

            # ------------------------------------------------
            # New permanent sample ID
            # ------------------------------------------------

            new_sample_id = (
                f"real_train_line_"
                f"{next_sample_number:06d}"
            )

            new_filename = (
                f"{new_sample_id}.png"
            )

            new_image_path = (
                REAL_LINE_DIR
                / new_filename
            )

            if new_image_path.exists():

                raise FileExistsError(
                    f"Target image already exists:\n"
                    f"{new_image_path}"
                )

            # ------------------------------------------------
            # Copy image into clean dataset
            # ------------------------------------------------

            shutil.copy2(
                old_image_path,
                new_image_path
            )

            copied_files.append(
                new_image_path
            )

            # ------------------------------------------------
            # New row
            # ------------------------------------------------

            new_row = {
                column: ""
                for column in clean_df.columns
            }

            new_row[
                "sample_id"
            ] = new_sample_id

            new_row[
                "image_path"
            ] = (
                Path(
                    "real_training_lines"
                )
                / new_filename
            ).as_posix()

            new_row[
                "source_id"
            ] = source_id

            new_row[
                "line_number"
            ] = line_number

            new_row[
                "ground_truth"
            ] = ground_truth

            new_row[
                "keep"
            ] = 1

            new_rows.append(
                new_row
            )

            merge_map.append(
                {
                    "auto_sample_id":
                        old_sample_id,

                    "new_sample_id":
                        new_sample_id,

                    "source_id":
                        source_id,

                    "ground_truth":
                        ground_truth,

                    "source_image_path":
                        old_relative_path,

                    "merged_image_path":
                        (
                            Path(
                                "real_training_lines"
                            )
                            / new_filename
                        ).as_posix()
                }
            )

            print(
                f"Merged: "
                f"{old_sample_id}"
                f" -> "
                f"{new_sample_id}"
            )

            next_sample_number += 1

    except Exception:

        # ----------------------------------------------------
        # Roll back copied files if merge fails
        # ----------------------------------------------------

        print(
            "\nERROR during merge."
        )

        print(
            "Rolling back newly copied images..."
        )

        for path in copied_files:

            if path.exists():

                path.unlink()

        raise

    # --------------------------------------------------------
    # Append new rows
    # --------------------------------------------------------

    new_df = pd.DataFrame(
        new_rows
    )

    combined_df = pd.concat(
        [
            clean_df,
            new_df
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Final duplicate sample ID check
    # --------------------------------------------------------

    duplicate_ids = (
        combined_df[
            "sample_id"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_ids != 0:

        for path in copied_files:

            if path.exists():

                path.unlink()

        raise RuntimeError(
            "Duplicate sample IDs detected. "
            "Merge cancelled."
        )

    # --------------------------------------------------------
    # Save updated clean CSV
    # --------------------------------------------------------

    combined_df.to_csv(
        CLEAN_REAL_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Save merge map
    # --------------------------------------------------------

    merge_map_df = pd.DataFrame(
        merge_map
    )

    merge_map_df.to_csv(
        MERGE_MAP_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    source_count = (
        combined_df[
            "source_id"
        ]
        .astype(str)
        .nunique()
    )

    summary = {

        "merged_at":
            datetime.now().isoformat(),

        "existing_clean_rows":
            original_clean_count,

        "new_verified_rows":
            len(
                new_df
            ),

        "final_clean_rows":
            len(
                combined_df
            ),

        "sources_represented":
            int(
                source_count
            ),

        "merge_map":
            str(
                MERGE_MAP_CSV
            ),

        "backup_csv":
            str(
                BACKUP_CSV
            )
    }

    with open(
        MERGE_SUMMARY_JSON,
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
        "\n" + "=" * 72
    )

    print(
        "VERIFIED AUTO REAL OCR MERGE COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nPrevious clean rows : "
        f"{original_clean_count}"
    )

    print(
        f"New verified rows   : "
        f"{len(new_df)}"
    )

    print(
        f"Final clean rows    : "
        f"{len(combined_df)}"
    )

    print(
        f"Sources represented : "
        f"{source_count}"
    )

    print(
        f"Images copied       : "
        f"{len(copied_files)}"
    )

    print(
        "\nUpdated clean CSV:"
    )

    print(
        CLEAN_REAL_CSV
    )

    print(
        "\nMerge map:"
    )

    print(
        MERGE_MAP_CSV
    )

    print(
        "\nBackup:"
    )

    print(
        BACKUP_CSV
    )