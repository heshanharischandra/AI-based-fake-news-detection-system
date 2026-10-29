from pathlib import Path
from datetime import datetime

import json
import os
import re
import unicodedata

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


# ============================================================
# 1. CONFIGURATION
# ============================================================

TESSERACT_PATH = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

TESSDATA_DIR = (
    r"D:\FakeNewsResearch\datasets\tessdata_best"
)

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Use tessdata_best models
os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "ocr_results"
)

DEBUG_DIR = (
    PROJECT_ROOT
    / "results"
    / "ocr_debug"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEBUG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. TESSERACT CHECK
# ============================================================

def check_tesseract():

    if not Path(TESSERACT_PATH).exists():

        raise FileNotFoundError(
            f"Tesseract not found:\n"
            f"{TESSERACT_PATH}"
        )

    if not Path(TESSDATA_DIR).exists():

        raise FileNotFoundError(
            f"tessdata_best not found:\n"
            f"{TESSDATA_DIR}"
        )

    sin_file = (
        Path(TESSDATA_DIR)
        / "sin.traineddata"
    )

    eng_file = (
        Path(TESSDATA_DIR)
        / "eng.traineddata"
    )

    if not sin_file.exists():

        raise FileNotFoundError(
            f"sin.traineddata not found:\n"
            f"{sin_file}"
        )

    if not eng_file.exists():

        raise FileNotFoundError(
            f"eng.traineddata not found:\n"
            f"{eng_file}"
        )

    languages = pytesseract.get_languages(
        config=""
    )

    if "sin" not in languages:

        raise RuntimeError(
            "Sinhala OCR language could not be loaded."
        )

    if "eng" not in languages:

        raise RuntimeError(
            "English OCR language could not be loaded."
        )

    return languages


# ============================================================
# 3. CHARACTER HELPERS
# ============================================================

def is_sinhala(ch):

    return (
        "\u0D80"
        <= ch
        <= "\u0DFF"
    )


def is_english(ch):

    return (
        ch.isascii()
        and ch.isalpha()
    )


# ============================================================
# 4. CLEAN TEXT
# ============================================================

def clean_text(text):

    if not text:

        return ""

    text = unicodedata.normalize(
        "NFC",
        text
    )

    text = text.replace(
        "\u200b",
        ""
    )

    text = text.replace(
        "\ufeff",
        ""
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = text.strip()

    # Remove garbage at start/end
    text = text.strip(
        " |॥෴_~`"
    )

    return text.strip()


# ============================================================
# 5. LANGUAGE DETECTION
# ============================================================

def detect_language(text):

    sinhala_count = sum(
        1
        for ch in text
        if is_sinhala(ch)
    )

    english_count = sum(
        1
        for ch in text
        if is_english(ch)
    )

    total = (
        sinhala_count
        + english_count
    )

    if total == 0:

        return "Unknown"

    sinhala_ratio = (
        sinhala_count
        / total
    )

    english_ratio = (
        english_count
        / total
    )

    # Sinhala sentence containing A-9
    # must still be classified as Sinhala
    if sinhala_ratio >= 0.65:

        return "Sinhala"

    if english_ratio >= 0.80:

        return "English"

    return "Sinhala + English"


# ============================================================
# 6. LOAD IMAGE
# ============================================================

def load_image(image_path):

    path = Path(
        image_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{path}"
        )

    image = cv2.imread(
        str(path)
    )

    if image is None:

        raise ValueError(
            f"Could not read image:\n"
            f"{path}"
        )

    return image


# ============================================================
# 7. CREATE YELLOW + WHITE MASKS
# ============================================================

def create_text_masks(image):

    height, width = image.shape[:2]

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # Yellow text
    # --------------------------------------------------------

    yellow_lower = np.array(
        [14, 80, 100],
        dtype=np.uint8
    )

    yellow_upper = np.array(
        [48, 255, 255],
        dtype=np.uint8
    )

    yellow_mask = cv2.inRange(
        hsv,
        yellow_lower,
        yellow_upper
    )

    # --------------------------------------------------------
    # White text
    # --------------------------------------------------------

    white_lower = np.array(
        [0, 0, 165],
        dtype=np.uint8
    )

    white_upper = np.array(
        [180, 95, 255],
        dtype=np.uint8
    )

    white_mask = cv2.inRange(
        hsv,
        white_lower,
        white_upper
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Only inspect lower news-headline section.
    #
    # Removes:
    # logo
    # date
    # road
    # road markings
    # HIRU watermark
    # social media icons
    # --------------------------------------------------------

    roi_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    y1 = int(
        height * 0.52
    )

    y2 = int(
        height * 0.86
    )

    x1 = int(
        width * 0.04
    )

    x2 = int(
        width * 0.96
    )

    roi_mask[
        y1:y2,
        x1:x2
    ] = 255

    yellow_mask = cv2.bitwise_and(
        yellow_mask,
        roi_mask
    )

    white_mask = cv2.bitwise_and(
        white_mask,
        roi_mask
    )

    # Small morphology cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2, 2)
    )

    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return (
        yellow_mask,
        white_mask
    )


# ============================================================
# 8. FIND TEXT-LINE BOXES
# ============================================================

def find_line_boxes(
    mask,
    color_name
):

    height, width = mask.shape[:2]

    # Join letters belonging to the same line
    horizontal_kernel = (
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (31, 5)
        )
    )

    merged = cv2.dilate(
        mask,
        horizontal_kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        merged,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # Ignore quotes, icons, tiny objects
        if w < width * 0.28:

            continue

        if h < 12:

            continue

        if h > height * 0.15:

            continue

        boxes.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "color": color_name
        })

    # Reading order
    boxes.sort(
        key=lambda item:
            item["y"]
    )

    return boxes


# ============================================================
# 9. CROP ORIGINAL IMAGE LINE
# ============================================================

def crop_original_line(
    image,
    box
):

    height, width = image.shape[:2]

    x = box["x"]
    y = box["y"]
    w = box["w"]
    h = box["h"]

    pad_x = 8
    pad_y = 7

    x1 = max(
        0,
        x - pad_x
    )

    y1 = max(
        0,
        y - pad_y
    )

    x2 = min(
        width,
        x + w + pad_x
    )

    y2 = min(
        height,
        y + h + pad_y
    )

    crop = image[
        y1:y2,
        x1:x2
    ]

    return crop


# ============================================================
# 10. CREATE LINE OCR VARIANTS
# ============================================================

def create_line_variants(
    crop,
    text_color
):

    enlarged = cv2.resize(
        crop,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        0.8
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.6,
        blurred,
        -0.6,
        0
    )

    otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )[1]

    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        9
    )

    # --------------------------------------------------------
    # Color-specific binary version
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2HSV
    )

    if text_color == "yellow":

        lower = np.array(
            [14, 70, 90],
            dtype=np.uint8
        )

        upper = np.array(
            [50, 255, 255],
            dtype=np.uint8
        )

    else:

        lower = np.array(
            [0, 0, 150],
            dtype=np.uint8
        )

        upper = np.array(
            [180, 110, 255],
            dtype=np.uint8
        )

    color_mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    color_binary = cv2.bitwise_not(
        color_mask
    )

    variants = {
        "enhanced_gray":
            sharpened,

        "otsu":
            otsu,

        "adaptive":
            adaptive,

        "color_binary":
            color_binary
    }

    # Add white padding around every line
    padded_variants = {}

    for name, variant in variants.items():

        padded = cv2.copyMakeBorder(
            variant,
            30,
            30,
            30,
            30,
            cv2.BORDER_CONSTANT,
            value=255
        )

        padded_variants[
            name
        ] = padded

    return padded_variants


# ============================================================
# 11. OCR ONE LINE
# ============================================================

def run_single_line_ocr(
    image,
    language,
    psm
):

    config = (
        f"--oem 1 "
        f"--psm {psm} "
        f"-c preserve_interword_spaces=1"
    )

    data = pytesseract.image_to_data(
        image,
        lang=language,
        config=config,
        output_type=Output.DICT
    )

    words = []
    confidences = []

    for text, confidence in zip(
        data["text"],
        data["conf"]
    ):

        word = text.strip()

        if not word:

            continue

        try:

            confidence = float(
                confidence
            )

        except (
            ValueError,
            TypeError
        ):

            continue

        if confidence < 10:

            continue

        words.append(
            word
        )

        confidences.append(
            confidence
        )

    text = clean_text(
        " ".join(words)
    )

    mean_confidence = (
        sum(confidences)
        / len(confidences)
        if confidences
        else 0.0
    )

    return {
        "text": text,
        "confidence": round(
            mean_confidence,
            2
        )
    }


# ============================================================
# 12. SCORE ONE OCR LINE
# ============================================================

def score_line_candidate(
    result,
    color_name
):

    text = result["text"]

    if not text:

        return -1000

    sinhala_count = sum(
        1
        for ch in text
        if is_sinhala(ch)
    )

    english_count = sum(
        1
        for ch in text
        if is_english(ch)
    )

    digit_count = sum(
        1
        for ch in text
        if ch.isdigit()
    )

    non_space = [
        ch
        for ch in text
        if not ch.isspace()
    ]

    useful = sum(
        1
        for ch in non_space
        if (
            is_sinhala(ch)
            or is_english(ch)
            or ch.isdigit()
            or ch in "-–—()/."
        )
    )

    useful_ratio = (
        useful / len(non_space)
        if non_space
        else 0
    )

    letters = (
        sinhala_count
        + english_count
    )

    sinhala_ratio = (
        sinhala_count / letters
        if letters > 0
        else 0
    )

    # Keep A-9 correctly if detected
    has_a9 = bool(
        re.search(
            r"A\s*[-–]?\s*9",
            text,
            re.IGNORECASE
        )
    )

    a9_bonus = (
        15
        if has_a9
        else 0
    )

    # NEWS / HIRU / random Latin garbage penalty
    allowed_english = (
        1
        if has_a9
        else 0
    )

    english_noise = max(
        0,
        english_count
        - allowed_english
    )

    english_penalty = (
        english_noise
        * 2.5
    )

    garbage_penalty = (
        (1 - useful_ratio)
        * 90
    )

    length_bonus = min(
        sinhala_count,
        45
    ) * 0.65

    score = (
        result["confidence"] * 1.20
        + sinhala_ratio * 55
        + length_bonus
        + a9_bonus
        - english_penalty
        - garbage_penalty
    )

    # News headline line should contain Sinhala
    if sinhala_count < 4:

        score -= 50

    return round(
        score,
        2
    )


# ============================================================
# 13. OCR A DETECTED LINE
# ============================================================

def recognize_line(
    crop,
    color_name
):

    variants = create_line_variants(
        crop,
        color_name
    )

    candidates = []

    for variant_name, processed in variants.items():

        for language in [
            "sin",
            "sin+eng"
        ]:

            for psm in [
                7,
                13
            ]:

                result = run_single_line_ocr(
                    processed,
                    language,
                    psm
                )

                result[
                    "preprocessing"
                ] = variant_name

                result[
                    "language"
                ] = language

                result[
                    "psm"
                ] = psm

                result[
                    "score"
                ] = score_line_candidate(
                    result,
                    color_name
                )

                result[
                    "processed"
                ] = processed

                candidates.append(
                    result
                )

    best = max(
        candidates,
        key=lambda item:
            item["score"]
    )

    return (
        best,
        candidates
    )


# ============================================================
# 14. OCR COLOR HEADLINES
# ============================================================

def extract_colored_headline(
    image,
    timestamp
):

    yellow_mask, white_mask = (
        create_text_masks(
            image
        )
    )

    yellow_boxes = find_line_boxes(
        yellow_mask,
        "yellow"
    )

    white_boxes = find_line_boxes(
        white_mask,
        "white"
    )

    all_boxes = (
        yellow_boxes
        + white_boxes
    )

    all_boxes.sort(
        key=lambda item:
            item["y"]
    )

    if not all_boxes:

        raise ValueError(
            "No yellow/white headline lines detected."
        )

    debug_overlay = (
        image.copy()
    )

    extracted_lines = []

    line_debug_dir = (
        DEBUG_DIR
        / f"lines_{timestamp}"
    )

    line_debug_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, box in enumerate(
        all_boxes,
        start=1
    ):

        crop = crop_original_line(
            image,
            box
        )

        best_result, candidates = (
            recognize_line(
                crop,
                box["color"]
            )
        )

        # Ignore failed / garbage lines
        sinhala_chars = sum(
            1
            for ch in best_result["text"]
            if is_sinhala(ch)
        )

        if sinhala_chars < 4:

            continue

        crop_path = (
            line_debug_dir
            / (
                f"line_{index}_"
                f"{box['color']}.png"
            )
        )

        cv2.imwrite(
            str(crop_path),
            crop
        )

        extracted_lines.append({
            "line_number":
                len(extracted_lines) + 1,

            "color":
                box["color"],

            "y":
                box["y"],

            "text":
                best_result["text"],

            "confidence":
                best_result["confidence"],

            "score":
                best_result["score"],

            "ocr_language":
                best_result["language"],

            "preprocessing":
                best_result["preprocessing"],

            "psm":
                best_result["psm"],

            "debug_crop":
                str(crop_path)
        })

        # Debug rectangle
        x = box["x"]
        y = box["y"]
        w = box["w"]
        h = box["h"]

        cv2.rectangle(
            debug_overlay,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    extracted_lines.sort(
        key=lambda item:
            item["y"]
    )

    overlay_path = (
        DEBUG_DIR
        / (
            f"detected_lines_"
            f"{timestamp}.png"
        )
    )

    cv2.imwrite(
        str(overlay_path),
        debug_overlay
    )

    return (
        extracted_lines,
        overlay_path
    )


# ============================================================
# 15. QUALITY FLAG
# ============================================================

def calculate_quality(
    lines
):

    if not lines:

        return (
            0.0,
            "LOW"
        )

    confidence = sum(
        line["confidence"]
        for line in lines
    ) / len(lines)

    # For a news card like this,
    # 2-3 meaningful lines is much better than
    # lots of random text.
    if (
        confidence >= 80
        and len(lines) >= 2
    ):

        quality = "HIGH"

    elif (
        confidence >= 55
        and len(lines) >= 2
    ):

        quality = "MEDIUM"

    else:

        quality = "LOW"

    return (
        round(confidence, 2),
        quality
    )


# ============================================================
# 16. SAVE JSON
# ============================================================

def save_result(
    image_path,
    lines,
    overlay_path,
    timestamp
):

    final_text = "\n".join(
        line["text"]
        for line in lines
    )

    confidence, quality = (
        calculate_quality(
            lines
        )
    )

    detected_language = (
        detect_language(
            final_text
        )
    )

    json_data = {

        "timestamp":
            datetime.now().isoformat(),

        "input_type":
            "image",

        "image_path":
            str(
                Path(
                    image_path
                ).resolve()
            ),

        "detected_language":
            detected_language,

        "extracted_text":
            final_text,

        "ocr_confidence":
            confidence,

        "quality_flag":
            quality,

        "manual_verification_recommended":
            quality != "HIGH",

        "ocr_engine":
            "Tesseract",

        "traineddata":
            "tessdata_best",

        "ocr_method":
            "color_guided_line_by_line",

        "detected_line_count":
            len(lines),

        "lines":
            lines,

        "debug_overlay":
            str(
                overlay_path
            )
    }

    json_path = (
        RESULTS_DIR
        / (
            f"ocr_result_"
            f"{timestamp}.json"
        )
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    return (
        json_path,
        json_data
    )


# ============================================================
# 17. MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 65
    )

    print(
        "SINHALA / ENGLISH NEWS IMAGE OCR"
    )

    print(
        "Color-Guided Line-by-Line OCR"
    )

    print(
        "=" * 65
    )

    try:

        languages = check_tesseract()

        print(
            "\nTesseract setup: OK"
        )

        print(
            "Available languages:",
            languages
        )

        image_path = input(
            "\nEnter image path: "
        ).strip().strip('"')

        timestamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        image = load_image(
            image_path
        )

        lines, overlay_path = (
            extract_colored_headline(
                image,
                timestamp
            )
        )

        json_path, result = (
            save_result(
                image_path,
                lines,
                overlay_path,
                timestamp
            )
        )

        print(
            "\n" + "=" * 65
        )

        print(
            "FINAL EXTRACTED TEXT"
        )

        print(
            "=" * 65
        )

        print()

        if result["extracted_text"]:

            print(
                result[
                    "extracted_text"
                ]
            )

        else:

            print(
                "No reliable text detected."
            )

        print(
            "\nDetected Language:",
            result[
                "detected_language"
            ]
        )

        print(
            "OCR Confidence:",
            result[
                "ocr_confidence"
            ],
            "%"
        )

        print(
            "OCR Quality:",
            result[
                "quality_flag"
            ]
        )

        print(
            "Detected Lines:",
            result[
                "detected_line_count"
            ]
        )

        print(
            "\nOCR Method:",
            result[
                "ocr_method"
            ]
        )

        if result[
            "manual_verification_recommended"
        ]:

            print(
                "\nWARNING:"
            )

            print(
                "OCR result is not HIGH quality."
            )

            print(
                "Manual verification recommended."
            )

        print(
            "\nJSON saved:"
        )

        print(
            json_path
        )

        print(
            "\nDetected-line debug image:"
        )

        print(
            overlay_path
        )

        print(
            "\n" + "=" * 65
        )

    except Exception as error:

        print(
            "\nERROR:"
        )

        print(
            error
        )