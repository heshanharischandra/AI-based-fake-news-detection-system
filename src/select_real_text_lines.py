from pathlib import Path
import csv

import cv2


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

REGION_DIR = (
    OCR_ROOT
    / "real_training_regions"
)

OUTPUT_DIR = (
    OCR_ROOT
    / "real_training_lines"
)

METADATA_DIR = (
    OCR_ROOT
    / "metadata"
    / "real_training"
)

LABELS_CSV = (
    METADATA_DIR
    / "real_manual_line_labels.csv"
)


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp"
}


# ============================================================
# SETUP
# ============================================================

def setup():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# REMOVE CURRENT AUTO CROPS
# ============================================================

def clean_old_line_crops():

    removed = 0

    for path in OUTPUT_DIR.glob(
        "real_train_line_*.png"
    ):

        path.unlink()
        removed += 1

    if LABELS_CSV.exists():

        LABELS_CSV.unlink()

    print(
        f"Old line crops removed: {removed}"
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(rows):

    columns = [
        "sample_id",
        "image_path",
        "source_id",
        "region_image",
        "line_number",
        "language",
        "ground_truth",
        "keep",
        "left",
        "top",
        "right",
        "bottom",
        "crop_width",
        "crop_height"
    ]

    with open(
        LABELS_CSV,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "MANUAL REAL SINHALA TEXT-LINE SELECTOR"
    )

    print(
        "=" * 72
    )

    setup()

    region_images = sorted([
        path
        for path in REGION_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ])

    print(
        f"\nHeadline regions found: "
        f"{len(region_images)}"
    )

    if not region_images:

        raise RuntimeError(
            "No headline regions found."
        )

    clean_old_line_crops()

    rows = []

    global_counter = 0

    for image_index, region_path in enumerate(
        region_images,
        start=1
    ):

        image = cv2.imread(
            str(region_path),
            cv2.IMREAD_COLOR
        )

        if image is None:

            print(
                f"Could not read: "
                f"{region_path.name}"
            )

            continue

        source_id = (
            region_path.stem
            .replace(
                "_headline",
                ""
            )
        )

        print(
            "\n" + "-" * 72
        )

        print(
            f"Region {image_index}/"
            f"{len(region_images)}"
        )

        print(
            region_path.name
        )

        print(
            "Select EACH text line separately."
        )

        print(
            "After drawing one box, press ENTER or SPACE."
        )

        print(
            "When all lines are selected, press ESC."
        )

        print(
            "Do NOT select logos, dates, or blank strips."
        )

        window_name = (
            f"Select text lines - "
            f"{region_path.name}"
        )

        rois = cv2.selectROIs(
            window_name,
            image,
            showCrosshair=True,
            fromCenter=False
        )

        cv2.destroyWindow(
            window_name
        )

        if rois is None or len(rois) == 0:

            print(
                "No lines selected."
            )

            continue

        # Sort top-to-bottom
        rois = sorted(
            rois,
            key=lambda roi:
                roi[1]
        )

        line_number = 0

        for roi in rois:

            x, y, width, height = [
                int(value)
                for value in roi
            ]

            if width <= 0 or height <= 0:

                continue

            crop = image[
                y:
                y + height,
                x:
                x + width
            ]

            if crop.size == 0:

                continue

            global_counter += 1
            line_number += 1

            sample_id = (
                f"real_train_line_"
                f"{global_counter:06d}"
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

            rows.append({

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
                    source_id,

                "region_image":
                    region_path.name,

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

        save_csv(
            rows
        )

        print(
            f"Saved text lines: "
            f"{line_number}"
        )

    cv2.destroyAllWindows()

    print(
        "\n" + "=" * 72
    )

    print(
        "MANUAL LINE SELECTION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nHeadline regions : "
        f"{len(region_images)}"
    )

    print(
        f"Selected lines   : "
        f"{len(rows)}"
    )

    print(
        "\nLine crops:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nLabels CSV:"
    )

    print(
        LABELS_CSV
    )