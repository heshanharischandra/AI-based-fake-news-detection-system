from pathlib import Path
import csv
import shutil

import cv2

import auto_extract_new_real_text_lines as line_extractor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

SOURCE_LINES_DIR = (
    OCR_ROOT
    / "real_training_lines"
)

SOURCE_LABELS_CSV = (
    OCR_ROOT
    / "metadata"
    / "real_training"
    / "real_manual_line_labels.csv"
)

CLEAN_ROOT = (
    OCR_ROOT
    / "cleaned_real_ocr_v1"
)

CLEAN_LINES_DIR = (
    CLEAN_ROOT
    / "lines"
)

CLEAN_METADATA_DIR = (
    CLEAN_ROOT
    / "metadata"
)

CLEAN_LABELS_CSV = (
    CLEAN_METADATA_DIR
    / "cleaned_real_labels_v1.csv"
)

REVIEW_LABELS_CSV = (
    CLEAN_METADATA_DIR
    / "split_label_review_v1.csv"
)

SPLIT_PREVIEW_DIR = (
    CLEAN_ROOT
    / "split_review"
)


# ============================================================
# CLEANUP DECISIONS
# ============================================================

SPLIT_IDS = {
    "real_train_line_000018",
    "real_train_line_000043",
    "real_train_line_000044",
    "real_train_line_000046",
    "real_train_line_000047",
    "real_train_line_000092",
    "real_train_line_000099",
}

EXCLUDE_IDS = {
    "real_train_line_000045",
}

KEEP_SPECIAL_IDS = {
    "real_train_line_000071",
}


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

def create_output_folders():

    CLEAN_LINES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CLEAN_METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    SPLIT_PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOAD ORIGINAL LABELS
# ============================================================

def load_labels():

    with open(
        SOURCE_LABELS_CSV,
        "r",
        encoding="utf-8-sig"
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


# ============================================================
# GET SOURCE IMAGE
# ============================================================

def get_source_image_path(record):

    image_path = str(
        record.get(
            "image_path",
            ""
        )
    ).strip()

    if not image_path:

        return None

    return (
        OCR_ROOT
        / image_path
    )


# ============================================================
# FIND VISUAL TEXT LINES
# ============================================================

def find_visual_line_boxes(image):

    mask = (
        line_extractor
        .create_text_mask(
            image
        )
    )

    bands = (
        line_extractor
        .find_line_bands(
            mask
        )
    )

    candidates = (
        line_extractor
        .filter_line_bands(
            mask,
            bands
        )
    )

    candidates = (
        line_extractor
        .merge_overlapping_candidates(
            candidates
        )
    )

    boxes = []

    for candidate in candidates:

        box = (
            line_extractor
            .expand_box(
                candidate,
                image.shape
            )
        )

        x1, y1, x2, y2 = box

        boxes.append(
            {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }
        )

    boxes = sorted(
        boxes,
        key=lambda item: item["y1"]
    )

    return (
        mask,
        boxes
    )


# ============================================================
# SAVE PREVIEW
# ============================================================

def save_split_preview(
    sample_id,
    image,
    boxes
):

    preview = image.copy()

    for index, box in enumerate(
        boxes,
        start=1
    ):

        cv2.rectangle(
            preview,
            (
                box["x1"],
                box["y1"]
            ),
            (
                box["x2"],
                box["y2"]
            ),
            (
                0,
                255,
                0
            ),
            2
        )

        cv2.putText(
            preview,
            f"LINE {index}",
            (
                box["x1"],
                max(
                    20,
                    box["y1"] - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                0,
                255,
                0
            ),
            2,
            cv2.LINE_AA
        )

    preview_path = (
        SPLIT_PREVIEW_DIR
        / f"{sample_id}_preview.png"
    )

    cv2.imwrite(
        str(preview_path),
        preview
    )

    return preview_path


# ============================================================
# SPLIT ONE MULTI-LINE SAMPLE
# ============================================================

def split_sample(
    record
):

    sample_id = (
        record["sample_id"]
        .strip()
    )

    source_image_path = (
        get_source_image_path(
            record
        )
    )

    if (
        source_image_path is None
        or not source_image_path.exists()
    ):

        raise FileNotFoundError(
            f"Source image missing for {sample_id}"
        )

    image = cv2.imread(
        str(source_image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Could not read {source_image_path}"
        )

    mask, boxes = (
        find_visual_line_boxes(
            image
        )
    )

    print()
    print(
        f"Splitting {sample_id}"
    )

    print(
        "Detected visual lines:",
        len(boxes)
    )

    # These samples were previously audited as 2-line crops.
    if len(boxes) != 2:

        raise RuntimeError(
            f"{sample_id}: expected 2 visual lines "
            f"but detected {len(boxes)}. "
            f"Stop and manually inspect this sample."
        )

    save_split_preview(
        sample_id,
        image,
        boxes
    )

    created_records = []

    for part_number, box in enumerate(
        boxes,
        start=1
    ):

        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]

        crop = image[
            y1:y2 + 1,
            x1:x2 + 1
        ]

        if crop.size == 0:

            raise RuntimeError(
                f"Empty crop created for "
                f"{sample_id} part {part_number}"
            )

        new_sample_id = (
            f"{sample_id}_split_{part_number}"
        )

        output_filename = (
            f"{new_sample_id}.png"
        )

        output_path = (
            CLEAN_LINES_DIR
            / output_filename
        )

        cv2.imwrite(
            str(output_path),
            crop
        )

        crop_height, crop_width = (
            crop.shape[:2]
        )

        # IMPORTANT:
        # ground_truth intentionally blank.
        # We will manually verify the exact text later.
        new_record = {
            "sample_id":
                new_sample_id,

            "image_path":
                (
                    "cleaned_real_ocr_v1/"
                    "lines/"
                    + output_filename
                ),

            "source_id":
                record.get(
                    "source_id",
                    ""
                ),

            "parent_sample_id":
                sample_id,

            "split_part":
                part_number,

            "original_line_number":
                record.get(
                    "line_number",
                    ""
                ),

            "language":
                record.get(
                    "language",
                    "si"
                ),

            "ground_truth":
                "",

            "original_ground_truth":
                record.get(
                    "ground_truth",
                    ""
                ),

            "keep":
                1,

            "left":
                x1,

            "top":
                y1,

            "right":
                x2,

            "bottom":
                y2,

            "crop_width":
                crop_width,

            "crop_height":
                crop_height,

            "cleaning_action":
                "SPLIT_NEEDS_LABEL_REVIEW"
        }

        created_records.append(
            new_record
        )

        print(
            f"  Part {part_number}: "
            f"{crop_width} x {crop_height}"
        )

        print(
            f"  Saved: {output_path}"
        )

    return created_records


# ============================================================
# COPY NORMAL SAMPLE
# ============================================================

def copy_normal_sample(record):

    sample_id = (
        record["sample_id"]
        .strip()
    )

    source_image_path = (
        get_source_image_path(
            record
        )
    )

    if (
        source_image_path is None
        or not source_image_path.exists()
    ):

        raise FileNotFoundError(
            f"Source image missing for {sample_id}"
        )

    output_filename = (
        f"{sample_id}.png"
    )

    output_path = (
        CLEAN_LINES_DIR
        / output_filename
    )

    shutil.copy2(
        source_image_path,
        output_path
    )

    image = cv2.imread(
        str(source_image_path)
    )

    if image is None:

        crop_width = (
            record.get(
                "crop_width",
                ""
            )
        )

        crop_height = (
            record.get(
                "crop_height",
                ""
            )
        )

    else:

        crop_height, crop_width = (
            image.shape[:2]
        )

    new_record = {
        "sample_id":
            sample_id,

        "image_path":
            (
                "cleaned_real_ocr_v1/"
                "lines/"
                + output_filename
            ),

        "source_id":
            record.get(
                "source_id",
                ""
            ),

        "parent_sample_id":
            "",

        "split_part":
            "",

        "original_line_number":
            record.get(
                "line_number",
                ""
            ),

        "language":
            record.get(
                "language",
                "si"
            ),

        "ground_truth":
            record.get(
                "ground_truth",
                ""
            ),

        "original_ground_truth":
            record.get(
                "ground_truth",
                ""
            ),

        "keep":
            record.get(
                "keep",
                "1"
            ),

        "left":
            record.get(
                "left",
                ""
            ),

        "top":
            record.get(
                "top",
                ""
            ),

        "right":
            record.get(
                "right",
                ""
            ),

        "bottom":
            record.get(
                "bottom",
                ""
            ),

        "crop_width":
            crop_width,

        "crop_height":
            crop_height,

        "cleaning_action":
            (
                "KEEP_MANUAL_CONFIRMED"
                if sample_id in KEEP_SPECIAL_IDS
                else "KEEP_ORIGINAL"
            )
    }

    return new_record


# ============================================================
# SAVE CSV
# ============================================================

FIELDNAMES = [
    "sample_id",
    "image_path",
    "source_id",
    "parent_sample_id",
    "split_part",
    "original_line_number",
    "language",
    "ground_truth",
    "original_ground_truth",
    "keep",
    "left",
    "top",
    "right",
    "bottom",
    "crop_width",
    "crop_height",
    "cleaning_action",
]


def save_csv(
    path,
    records
):

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()

        writer.writerows(
            records
        )


# ============================================================
# MAIN CLEANING PROCESS
# ============================================================

def prepare_clean_dataset():

    create_output_folders()

    records = load_labels()

    print("=" * 72)
    print("PREPARE CLEAN REAL OCR DATASET V1")
    print("=" * 72)

    print()
    print(
        "Original records:",
        len(records)
    )

    clean_records = []

    split_review_records = []

    excluded_records = []

    split_parent_count = 0

    copied_count = 0

    # ========================================================
    # PROCESS EACH ORIGINAL SAMPLE
    # ========================================================

    for record in records:

        sample_id = (
            record.get(
                "sample_id",
                ""
            ).strip()
        )

        # ----------------------------------------------------
        # EXCLUDE DUPLICATE
        # ----------------------------------------------------

        if sample_id in EXCLUDE_IDS:

            print()
            print(
                f"EXCLUDE duplicate: {sample_id}"
            )

            excluded_records.append(
                sample_id
            )

            continue

        # ----------------------------------------------------
        # SPLIT MULTI-LINE SAMPLE
        # ----------------------------------------------------

        if sample_id in SPLIT_IDS:

            new_records = (
                split_sample(
                    record
                )
            )

            clean_records.extend(
                new_records
            )

            split_review_records.extend(
                new_records
            )

            split_parent_count += 1

            continue

        # ----------------------------------------------------
        # KEEP NORMAL SAMPLE
        # ----------------------------------------------------

        new_record = (
            copy_normal_sample(
                record
            )
        )

        clean_records.append(
            new_record
        )

        copied_count += 1

    # ========================================================
    # SAVE METADATA
    # ========================================================

    save_csv(
        CLEAN_LABELS_CSV,
        clean_records
    )

    save_csv(
        REVIEW_LABELS_CSV,
        split_review_records
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("CLEAN DATASET PREPARATION SUMMARY")
    print("=" * 72)

    print(
        "Original metadata rows:",
        len(records)
    )

    print(
        "Original samples copied unchanged:",
        copied_count
    )

    print(
        "Multi-line parent samples split:",
        split_parent_count
    )

    print(
        "New split line images:",
        len(
            split_review_records
        )
    )

    print(
        "Excluded duplicates:",
        len(
            excluded_records
        )
    )

    print(
        "Final clean metadata rows:",
        len(
            clean_records
        )
    )

    print()
    print(
        "Excluded sample IDs:"
    )

    for sample_id in excluded_records:

        print(
            " ",
            sample_id
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The split samples still have BLANK "
        "ground_truth fields."
    )

    print(
        "Do NOT train using this cleaned dataset yet."
    )

    print()
    print(
        "Clean lines folder:"
    )

    print(
        CLEAN_LINES_DIR
    )

    print()
    print(
        "Clean metadata:"
    )

    print(
        CLEAN_LABELS_CSV
    )

    print()
    print(
        "Split label review CSV:"
    )

    print(
        REVIEW_LABELS_CSV
    )

    print()
    print(
        "Split previews:"
    )

    print(
        SPLIT_PREVIEW_DIR
    )

    print("=" * 72)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        prepare_clean_dataset()

    except Exception as error:

        print()
        print(
            "ERROR:"
        )

        print(
            error
        )