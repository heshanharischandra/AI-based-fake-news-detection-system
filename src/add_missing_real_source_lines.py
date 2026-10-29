from pathlib import Path
import re

import cv2
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

REGION_PATH = (
    OCR_ROOT
    / "real_training_regions"
    / "real_train_001_headline.png"
)

OUTPUT_DIR = (
    OCR_ROOT
    / "real_training_lines"
)

CSV_PATH = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "real_manual_line_labels.csv"
)

SOURCE_ID = "real_train_001"


# ============================================================
# LOAD EXISTING DATA
# ============================================================

if not REGION_PATH.exists():
    raise FileNotFoundError(
        f"Headline region not found:\n{REGION_PATH}"
    )

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Labels CSV not found:\n{CSV_PATH}"
    )

df = pd.read_csv(
    CSV_PATH,
    encoding="utf-8-sig"
)


# ============================================================
# SAFETY CHECK
# ============================================================

if SOURCE_ID in set(
    df["source_id"].astype(str)
):
    print(
        f"{SOURCE_ID} already exists in CSV."
    )
    raise SystemExit


# ============================================================
# FIND NEXT SAMPLE NUMBER
# ============================================================

numbers = []

for sample_id in df["sample_id"].astype(str):

    match = re.search(
        r"(\d+)$",
        sample_id
    )

    if match:
        numbers.append(
            int(match.group(1))
        )

next_number = (
    max(numbers) + 1
    if numbers
    else 1
)


# ============================================================
# LOAD HEADLINE REGION
# ============================================================

image = cv2.imread(
    str(REGION_PATH),
    cv2.IMREAD_COLOR
)

if image is None:
    raise ValueError(
        f"Could not read:\n{REGION_PATH}"
    )


print("=" * 72)
print("ADD MISSING REAL SOURCE LINES")
print("=" * 72)

print(
    "\nSource:",
    SOURCE_ID
)

print(
    "Select EACH Sinhala text line separately."
)

print(
    "After each rectangle press ENTER or SPACE."
)

print(
    "When all lines are selected press ESC."
)

print(
    "Do not select logos, dates, URLs, or blank areas."
)


# ============================================================
# SELECT MULTIPLE TEXT LINES
# ============================================================

window_name = (
    "Select lines - real_train_001"
)

rois = cv2.selectROIs(
    window_name,
    image,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()


if rois is None or len(rois) == 0:

    print(
        "\nNo lines selected."
    )

    raise SystemExit


# Top-to-bottom
rois = sorted(
    rois,
    key=lambda roi: roi[1]
)


# ============================================================
# CREATE NEW RECORDS
# ============================================================

new_rows = []

line_number = 0


for roi in rois:

    x, y, width, height = [
        int(value)
        for value in roi
    ]

    if width <= 0 or height <= 0:
        continue

    crop = image[
        y:y + height,
        x:x + width
    ]

    if crop.size == 0:
        continue

    sample_id = (
        f"real_train_line_"
        f"{next_number:06d}"
    )

    filename = (
        f"{sample_id}.png"
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    success = cv2.imwrite(
        str(output_path),
        crop
    )

    if not success:
        raise RuntimeError(
            f"Could not save:\n"
            f"{output_path}"
        )

    line_number += 1

    new_rows.append({

        "sample_id":
            sample_id,

        "image_path":
            (
                Path(
                    "real_training_lines"
                )
                / filename
            ).as_posix(),

        "source_id":
            SOURCE_ID,

        "region_image":
            REGION_PATH.name,

        "line_number":
            line_number,

        "language":
            "si",

        "ground_truth":
            "",

        "keep":
            "1",

        "left":
            x,

        "top":
            y,

        "right":
            x + width,

        "bottom":
            y + height,

        "crop_width":
            width,

        "crop_height":
            height
    })

    print(
        f"Saved: {sample_id}"
    )

    next_number += 1


# ============================================================
# APPEND TO CSV
# ============================================================

new_df = pd.DataFrame(
    new_rows
)

updated_df = pd.concat(
    [
        df,
        new_df
    ],
    ignore_index=True
)

updated_df.to_csv(
    CSV_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 72
)

print(
    "MISSING SOURCE RECOVERY COMPLETE"
)

print(
    "=" * 72
)

print(
    f"\nNew lines added : "
    f"{len(new_rows)}"
)

print(
    f"Total CSV rows  : "
    f"{len(updated_df)}"
)

print(
    f"Sources now     : "
    f"{updated_df['source_id'].nunique()}"
)