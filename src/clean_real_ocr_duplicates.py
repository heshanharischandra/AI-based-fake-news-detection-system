from pathlib import Path
import hashlib
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

CSV_PATH = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "real_manual_line_labels.csv"
)

BACKUP_CSV = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "real_manual_line_labels_before_dedup.csv"
)

DUPLICATE_BACKUP_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "duplicate_line_backup"
)


# ============================================================
# FILE HASH
# ============================================================

def sha256_file(path):

    hasher = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "REAL OCR EXACT DUPLICATE CLEANUP"
    )

    print(
        "=" * 72
    )

    if not CSV_PATH.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{CSV_PATH}"
        )

    DUPLICATE_BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        CSV_PATH,
        encoding="utf-8-sig"
    )

    print(
        f"\nOriginal CSV rows: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Backup original CSV
    # --------------------------------------------------------

    shutil.copy2(
        CSV_PATH,
        BACKUP_CSV
    )

    print(
        f"CSV backup created:\n"
        f"{BACKUP_CSV}"
    )

    # --------------------------------------------------------
    # Calculate hashes
    # --------------------------------------------------------

    hash_to_indices = {}

    for index, row in df.iterrows():

        image_path = (
            OCR_ROOT
            / str(
                row["image_path"]
            )
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image missing:\n"
                f"{image_path}"
            )

        file_hash = sha256_file(
            image_path
        )

        hash_to_indices.setdefault(
            file_hash,
            []
        ).append(
            index
        )

    # --------------------------------------------------------
    # Find duplicate rows
    #
    # Keep FIRST occurrence.
    # Move extra image files to backup.
    # --------------------------------------------------------

    duplicate_indices = []

    removed_sample_ids = []

    for file_hash, indices in hash_to_indices.items():

        if len(indices) <= 1:
            continue

        keep_index = indices[0]

        keep_sample_id = str(
            df.at[
                keep_index,
                "sample_id"
            ]
        )

        print(
            "\nDuplicate group:"
        )

        print(
            f"  Keeping: "
            f"{keep_sample_id}"
        )

        for duplicate_index in indices[1:]:

            sample_id = str(
                df.at[
                    duplicate_index,
                    "sample_id"
                ]
            )

            relative_path = str(
                df.at[
                    duplicate_index,
                    "image_path"
                ]
            )

            image_path = (
                OCR_ROOT
                / relative_path
            )

            backup_path = (
                DUPLICATE_BACKUP_DIR
                / image_path.name
            )

            if image_path.exists():

                shutil.move(
                    str(image_path),
                    str(backup_path)
                )

            duplicate_indices.append(
                duplicate_index
            )

            removed_sample_ids.append(
                sample_id
            )

            print(
                f"  Removed duplicate: "
                f"{sample_id}"
            )

    # --------------------------------------------------------
    # Remove duplicate CSV rows
    # --------------------------------------------------------

    cleaned_df = df.drop(
        index=duplicate_indices
    ).reset_index(
        drop=True
    )

    cleaned_df.to_csv(
        CSV_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    print(
        "DUPLICATE CLEANUP COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nOriginal rows      : "
        f"{len(df)}"
    )

    print(
        f"Duplicates removed : "
        f"{len(duplicate_indices)}"
    )

    print(
        f"Clean rows         : "
        f"{len(cleaned_df)}"
    )

    print(
        f"Sources represented: "
        f"{cleaned_df['source_id'].nunique()}"
    )

    print(
        "\nRemoved sample IDs:"
    )

    for sample_id in removed_sample_ids:

        print(
            f"  {sample_id}"
        )

    print(
        "\nDuplicate images backed up to:"
    )

    print(
        DUPLICATE_BACKUP_DIR
    )