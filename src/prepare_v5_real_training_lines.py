from pathlib import Path
import csv

import cv2

from image_ocr_v4_hybrid import (
    PROJECT_ROOT,
    load_image,
    detect_colour_headline,
    build_colour_roi,
    extract_training_style_lines,
    create_adaptive_colour_crops,
    create_preview,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
    / "real_training_expansion_v5"
)

SOURCE_DIR = (
    BASE_DIR
    / "source_images"
)

LINE_DIR = (
    BASE_DIR
    / "line_crops"
)

METADATA_DIR = (
    BASE_DIR
    / "metadata"
)

DEBUG_DIR = (
    BASE_DIR
    / "debug_overlays"
)

CSV_PATH = (
    METADATA_DIR
    / "v5_manual_line_labels.csv"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

LINE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEBUG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_source_image(image_path):

    source_id = image_path.stem

    print()
    print("=" * 72)
    print("SOURCE:")
    print(source_id)
    print("=" * 72)

    image = load_image(
        image_path
    )

    (
        yellow_mask,
        white_mask,
        scan_info,
        raw_boxes,
        text_boxes,
        selected_boxes,
    ) = detect_colour_headline(
        image
    )

    print(
        "Raw candidates:",
        len(raw_boxes),
    )

    print(
        "Text-like candidates:",
        len(text_boxes),
    )

    print(
        "Selected headline lines:",
        len(selected_boxes),
    )

    # --------------------------------------------------------
    # SAVE DEBUG PREVIEW
    # --------------------------------------------------------

    preview = create_preview(
        image,
        raw_boxes,
        text_boxes,
        selected_boxes,
    )

    preview_path = (
        DEBUG_DIR
        / f"{source_id}_detection_preview.png"
    )

    cv2.imwrite(
        str(preview_path),
        preview,
    )

    yellow_path = (
        DEBUG_DIR
        / f"{source_id}_yellow_mask.png"
    )

    white_path = (
        DEBUG_DIR
        / f"{source_id}_white_mask.png"
    )

    cv2.imwrite(
        str(yellow_path),
        yellow_mask,
    )

    cv2.imwrite(
        str(white_path),
        white_mask,
    )

    # --------------------------------------------------------
    # NO HEADLINE FOUND
    # --------------------------------------------------------

    if not selected_boxes:

        print(
            "WARNING: No headline lines detected."
        )

        return []

    # --------------------------------------------------------
    # BUILD HEADLINE ROI
    # --------------------------------------------------------

    (
        headline_roi,
        roi_info,
    ) = build_colour_roi(
        image,
        selected_boxes,
    )

    roi_path = (
        DEBUG_DIR
        / f"{source_id}_headline_roi.png"
    )

    cv2.imwrite(
        str(roi_path),
        headline_roi,
    )

    # --------------------------------------------------------
    # TRY TRAINING-STYLE SEGMENTATION
    # --------------------------------------------------------

    (
        segmentation_mask,
        segmentation_boxes,
        training_crops,
    ) = extract_training_style_lines(
        headline_roi
    )

    segmentation_path = (
        DEBUG_DIR
        / f"{source_id}_segmentation_mask.png"
    )

    cv2.imwrite(
        str(segmentation_path),
        segmentation_mask,
    )

    if (
        len(training_crops)
        == len(selected_boxes)
        and len(training_crops) > 0
    ):

        line_crops = (
            training_crops
        )

        extraction_method = (
            "colour_roi_plus_training_segmentation"
        )

    else:

        print(
            "Training segmentation count mismatch."
        )

        print(
            "Using adaptive colour crops."
        )

        line_crops = (
            create_adaptive_colour_crops(
                image,
                selected_boxes,
            )
        )

        extraction_method = (
            "adaptive_colour_line_crop"
        )

    print(
        "Final line crops:",
        len(line_crops),
    )

    print(
        "Extraction method:",
        extraction_method,
    )

    # --------------------------------------------------------
    # SAVE LINE CROPS
    # --------------------------------------------------------

    rows = []

    for line_number, item in enumerate(
        line_crops,
        start=1,
    ):

        crop = item[
            "crop"
        ]

        crop_name = (
            f"{source_id}_line_"
            f"{line_number:02d}.png"
        )

        crop_path = (
            LINE_DIR
            / crop_name
        )

        saved = cv2.imwrite(
            str(crop_path),
            crop,
        )

        if not saved:

            print(
                "WARNING: Could not save:",
                crop_path,
            )

            continue

        sample_id = (
            f"{source_id}_line_"
            f"{line_number:02d}"
        )

        rows.append(
            {
                "sample_id":
                    sample_id,

                "source_id":
                    source_id,

                "source_image":
                    str(
                        image_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "line_number":
                    line_number,

                "crop_path":
                    str(
                        crop_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "extraction_method":
                    extraction_method,

                "ground_truth_text":
                    "",

                "review_status":
                    "PENDING",
            }
        )

        print(
            f"  Saved line {line_number}: "
            f"{crop_name}"
        )

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    if not SOURCE_DIR.exists():

        raise FileNotFoundError(
            f"Source folder not found:\n"
            f"{SOURCE_DIR}"
        )

    source_images = sorted(
        [
            path
            for path in SOURCE_DIR.iterdir()
            if (
                path.is_file()
                and
                path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )

    if not source_images:

        raise RuntimeError(
            "No source images found."
        )

    print("=" * 72)
    print("V5 REAL SINHALA OCR LINE PREPARATION")
    print("=" * 72)

    print()
    print(
        "Source images:",
        len(source_images),
    )

    all_rows = []

    for image_path in source_images:

        rows = process_source_image(
            image_path
        )

        all_rows.extend(
            rows
        )

    # --------------------------------------------------------
    # WRITE LABEL TEMPLATE CSV
    # --------------------------------------------------------

    fieldnames = [
        "sample_id",
        "source_id",
        "source_image",
        "line_number",
        "crop_path",
        "extraction_method",
        "ground_truth_text",
        "review_status",
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )

    print()
    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)

    print(
        "Source images processed:",
        len(source_images),
    )

    print(
        "Total line crops:",
        len(all_rows),
    )

    print()
    print(
        "Line crops folder:"
    )

    print(
        LINE_DIR
    )

    print()
    print(
        "Ground Truth CSV:"
    )

    print(
        CSV_PATH
    )

    print()
    print(
        "Debug previews:"
    )

    print(
        DEBUG_DIR
    )

    print("=" * 72)


if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print("ERROR:")
        print(error)