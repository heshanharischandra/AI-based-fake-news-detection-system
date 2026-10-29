from pathlib import Path
import csv
import os
import shutil

import cv2
import pandas as pd
import pytesseract


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "sinhala_ocr"
)

SOURCE_DIR = (
    OCR_ROOT
    / "real_training_sources"
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

DEBUG_DIR = (
    METADATA_DIR
    / "debug_overlays"
)

LABELS_CSV = (
    METADATA_DIR
    / "real_training_labels.csv"
)


# ============================================================
# 2. TESSERACT CONFIGURATION
# ============================================================

TESSERACT_EXE = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

TESSDATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "tessdata_best"
)


# ============================================================
# 3. SETTINGS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp"
}

MIN_WORD_WIDTH = 4

MIN_WORD_HEIGHT = 8

MIN_LINE_WIDTH = 25

MIN_LINE_HEIGHT = 12

PADDING_X = 12

PADDING_Y = 8

MAX_IMAGE_WIDTH = 1800


# ============================================================
# 4. SETUP
# ============================================================

def setup_environment():

    if not TESSERACT_EXE.exists():

        raise FileNotFoundError(
            f"Tesseract executable not found:\n"
            f"{TESSERACT_EXE}"
        )

    if not TESSDATA_DIR.exists():

        raise FileNotFoundError(
            f"Tessdata folder not found:\n"
            f"{TESSDATA_DIR}"
        )

    pytesseract.pytesseract.tesseract_cmd = str(
        TESSERACT_EXE
    )

    os.environ[
        "TESSDATA_PREFIX"
    ] = str(
        TESSDATA_DIR
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 5. CLEAN OLD GENERATED OUTPUT
# ============================================================

def clean_old_outputs():

    removed_lines = 0
    removed_debug = 0

    for path in OUTPUT_DIR.glob(
        "real_train_line_*.png"
    ):

        path.unlink()

        removed_lines += 1

    for path in DEBUG_DIR.glob(
        "*_overlay.png"
    ):

        path.unlink()

        removed_debug += 1

    if LABELS_CSV.exists():

        LABELS_CSV.unlink()

    print(
        f"Old generated line crops removed: "
        f"{removed_lines}"
    )

    print(
        f"Old debug overlays removed: "
        f"{removed_debug}"
    )


# ============================================================
# 6. LOAD IMAGE
# ============================================================

def load_image(
    image_path
):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            f"Could not read image:\n"
            f"{image_path}"
        )

    return image


# ============================================================
# 7. OPTIONAL RESIZE FOR DETECTION
# ============================================================

def resize_for_detection(
    image
):

    height, width = image.shape[:2]

    if width <= MAX_IMAGE_WIDTH:

        return image, 1.0

    scale = (
        MAX_IMAGE_WIDTH
        / width
    )

    resized_height = int(
        height * scale
    )

    resized = cv2.resize(
        image,
        (
            MAX_IMAGE_WIDTH,
            resized_height
        ),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


# ============================================================
# 8. DETECT WORD BOXES WITH TESSERACT
# ============================================================

def detect_words(
    image
):

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    config = (
        "--oem 1 "
        "--psm 11 "
        "-c preserve_interword_spaces=1"
    )

    data = pytesseract.image_to_data(
        rgb_image,
        lang="sin+eng",
        config=config,
        output_type=pytesseract.Output.DATAFRAME
    )

    if data is None or len(data) == 0:

        return pd.DataFrame()

    required_columns = [
        "block_num",
        "par_num",
        "line_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text"
    ]

    for column in required_columns:

        if column not in data.columns:

            raise ValueError(
                f"Tesseract output missing "
                f"column: {column}"
            )

    data = data.copy()

    data["text"] = (
        data["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    data = data[
        (data["width"] >= MIN_WORD_WIDTH)
        &
        (data["height"] >= MIN_WORD_HEIGHT)
        &
        (data["text"] != "")
    ].copy()

    return data


# ============================================================
# 9. GROUP WORDS INTO LINES
# ============================================================

def group_words_into_lines(
    data
):

    if data.empty:

        return []

    groups = data.groupby(
        [
            "block_num",
            "par_num",
            "line_num"
        ],
        sort=False
    )

    detected_lines = []

    for _, group in groups:

        if group.empty:

            continue

        left = int(
            group["left"].min()
        )

        top = int(
            group["top"].min()
        )

        right = int(
            (
                group["left"]
                + group["width"]
            ).max()
        )

        bottom = int(
            (
                group["top"]
                + group["height"]
            ).max()
        )

        width = (
            right - left
        )

        height = (
            bottom - top
        )

        if width < MIN_LINE_WIDTH:

            continue

        if height < MIN_LINE_HEIGHT:

            continue

        words = []

        group_sorted = group.sort_values(
            "left"
        )

        for text in group_sorted[
            "text"
        ].tolist():

            text = str(
                text
            ).strip()

            if text:

                words.append(
                    text
                )

        tesseract_guess = " ".join(
            words
        )

        detected_lines.append({

            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
            "width": width,
            "height": height,
            "tesseract_guess":
                tesseract_guess
        })

    # Top-to-bottom ordering
    detected_lines.sort(
        key=lambda item: (
            item["top"],
            item["left"]
        )
    )

    return detected_lines


# ============================================================
# 10. CONVERT COORDINATES BACK TO ORIGINAL IMAGE
# ============================================================

def restore_original_coordinates(
    line,
    scale
):

    if scale == 1.0:

        return line.copy()

    restored = line.copy()

    restored["left"] = int(
        line["left"]
        / scale
    )

    restored["top"] = int(
        line["top"]
        / scale
    )

    restored["right"] = int(
        line["right"]
        / scale
    )

    restored["bottom"] = int(
        line["bottom"]
        / scale
    )

    restored["width"] = (
        restored["right"]
        - restored["left"]
    )

    restored["height"] = (
        restored["bottom"]
        - restored["top"]
    )

    return restored


# ============================================================
# 11. CROP LINE
# ============================================================

def crop_line(
    image,
    line
):

    image_height, image_width = (
        image.shape[:2]
    )

    left = max(
        0,
        line["left"]
        - PADDING_X
    )

    top = max(
        0,
        line["top"]
        - PADDING_Y
    )

    right = min(
        image_width,
        line["right"]
        + PADDING_X
    )

    bottom = min(
        image_height,
        line["bottom"]
        + PADDING_Y
    )

    crop = image[
        top:bottom,
        left:right
    ]

    return (
        crop,
        left,
        top,
        right,
        bottom
    )


# ============================================================
# 12. PROCESS ONE SOURCE IMAGE
# ============================================================

def process_source_image(
    image_path,
    global_counter
):

    source_id = (
        image_path.stem
    )

    original = load_image(
        image_path
    )

    detection_image, scale = (
        resize_for_detection(
            original
        )
    )

    data = detect_words(
        detection_image
    )

    detected_lines = (
        group_words_into_lines(
            data
        )
    )

    # Debug image
    overlay = original.copy()

    rows = []

    source_line_number = 0

    for detected_line in detected_lines:

        line = (
            restore_original_coordinates(
                detected_line,
                scale
            )
        )

        (
            crop,
            left,
            top,
            right,
            bottom
        ) = crop_line(
            original,
            line
        )

        if crop.size == 0:

            continue

        crop_height, crop_width = (
            crop.shape[:2]
        )

        if (
            crop_width < MIN_LINE_WIDTH
            or crop_height < MIN_LINE_HEIGHT
        ):

            continue

        global_counter += 1
        source_line_number += 1

        sample_id = (
            f"real_train_line_"
            f"{global_counter:06d}"
        )

        output_name = (
            f"{sample_id}.png"
        )

        output_path = (
            OUTPUT_DIR
            / output_name
        )

        success = cv2.imwrite(
            str(output_path),
            crop
        )

        if not success:

            raise RuntimeError(
                f"Could not save crop:\n"
                f"{output_path}"
            )

        relative_path = (
            Path(
                "real_training_lines"
            )
            / output_name
        )

        rows.append({

            "sample_id":
                sample_id,

            "image_path":
                relative_path.as_posix(),

            "source_id":
                source_id,

            "source_image":
                image_path.name,

            "line_number":
                source_line_number,

            "language":
                "si",

            "detector":
                "tesseract_line_detection",

            "tesseract_guess":
                line[
                    "tesseract_guess"
                ],

            # IMPORTANT:
            # Fill this manually after inspection.
            "ground_truth":
                "",

            "left":
                left,

            "top":
                top,

            "right":
                right,

            "bottom":
                bottom,

            "crop_width":
                crop_width,

            "crop_height":
                crop_height
        })

        # -----------------------------------------------
        # Draw candidate rectangle on debug image
        # -----------------------------------------------

        cv2.rectangle(
            overlay,
            (
                left,
                top
            ),
            (
                right,
                bottom
            ),
            (
                0,
                255,
                0
            ),
            2
        )

        cv2.putText(
            overlay,
            str(
                source_line_number
            ),
            (
                left,
                max(
                    top - 5,
                    15
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                0,
                255,
                0
            ),
            2,
            cv2.LINE_AA
        )

    debug_path = (
        DEBUG_DIR
        / f"{source_id}_overlay.png"
    )

    cv2.imwrite(
        str(debug_path),
        overlay
    )

    return (
        rows,
        global_counter
    )


# ============================================================
# 13. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 72
    )

    print(
        "REAL SINHALA OCR TRAINING LINE PREPARATION"
    )

    print(
        "=" * 72
    )

    setup_environment()

    source_images = sorted([
        path
        for path in SOURCE_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ])

    print(
        f"\nSource images found: "
        f"{len(source_images)}"
    )

    if not source_images:

        raise RuntimeError(
            "No real training source images found."
        )

    clean_old_outputs()

    all_rows = []

    global_counter = 0

    for index, image_path in enumerate(
        source_images,
        start=1
    ):

        print(
            f"\nProcessing "
            f"{index}/{len(source_images)}: "
            f"{image_path.name}"
        )

        try:

            rows, global_counter = (
                process_source_image(
                    image_path,
                    global_counter
                )
            )

            all_rows.extend(
                rows
            )

            print(
                f"Candidate lines detected: "
                f"{len(rows)}"
            )

        except Exception as error:

            print(
                f"ERROR: {error}"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    columns = [
        "sample_id",
        "image_path",
        "source_id",
        "source_image",
        "line_number",
        "language",
        "detector",
        "tesseract_guess",
        "ground_truth",
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
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns
        )

        writer.writeheader()

        writer.writerows(
            all_rows
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "REAL TRAINING LINE EXTRACTION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nSource images : "
        f"{len(source_images)}"
    )

    print(
        f"Candidate lines: "
        f"{len(all_rows)}"
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

    print(
        "\nDebug overlays:"
    )

    print(
        DEBUG_DIR
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Tesseract guesses are NOT ground truth."
    )

    print(
        "Next step is manual visual review "
        "and correct transcription."
    )