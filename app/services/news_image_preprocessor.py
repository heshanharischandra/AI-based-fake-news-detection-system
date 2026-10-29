import json
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from app.services.ocr_service import configure_tesseract, select_ocr_language

from app.services.ocr_service import detect_language


BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "uploads" / "processed"


def analyze_news_image(image_path, min_confidence=20):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as source:
        original = ImageOps.exif_transpose(
            source
        ).convert("RGB")

    gray = ImageOps.autocontrast(
        ImageOps.grayscale(original)
    )

    ocr_image = ImageEnhance.Contrast(
        gray
    ).enhance(1.3)

    if configure_tesseract() is None:
        raise RuntimeError("Tesseract is not installed or discoverable. Set TESSERACT_CMD or install Tesseract OCR.")
    language = select_ocr_language()
    if language is None:
        raise RuntimeError("Tesseract needs eng.traineddata and/or sin.traineddata.")

    data = pytesseract.image_to_data(
        ocr_image,
        lang=language,
        config="--oem 3 --psm 11",
        output_type=Output.DICT,
    )

    groups = defaultdict(list)

    for i in range(len(data["text"])):
        word = str(data["text"][i]).strip()

        if not word:
            continue

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            continue

        if confidence < min_confidence:
            continue

        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])

        if width <= 0 or height <= 0:
            continue

        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
        )

        groups[key].append(
            (
                left,
                top,
                width,
                height,
                word,
                confidence,
            )
        )

    lines = []

    for words in groups.values():
        words.sort(key=lambda item: item[0])

        x1 = min(word[0] for word in words)
        y1 = min(word[1] for word in words)

        x2 = max(
            word[0] + word[2]
            for word in words
        )

        y2 = max(
            word[1] + word[3]
            for word in words
        )

        text = " ".join(
            word[4] for word in words
        )

        mean_confidence = (
            sum(word[5] for word in words)
            / len(words)
        )

        lines.append({
            "text": text,
            "language": detect_language(text),
            "confidence": round(
                mean_confidence,
                1,
            ),
            "bbox": [x1, y1, x2, y2],
        })

    lines.sort(
        key=lambda line: (
            line["bbox"][1],
            line["bbox"][0],
        )
    )

    output_dir = PROCESSED_DIR / (
        f"{image_path.stem}_{uuid4().hex[:8]}"
    )

    regions_dir = output_dir / "text_regions"

    regions_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview = original.copy()
    drawer = ImageDraw.Draw(preview)

    for index, line in enumerate(
        lines,
        start=1,
    ):
        x1, y1, x2, y2 = line["bbox"]

        padding = 6

        crop_box = (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(original.width, x2 + padding),
            min(original.height, y2 + padding),
        )

        crop_path = regions_dir / (
            f"line_{index:03d}_{line['language']}.png"
        )

        original.crop(crop_box).save(
            crop_path,
            format="PNG",
        )

        drawer.rectangle(
            crop_box,
            outline="red",
            width=2,
        )

        line["id"] = index
        line["crop_path"] = str(crop_path)

    preview_path = (
        output_dir / "text_regions_preview.png"
    )

    preview.save(
        preview_path,
        format="PNG",
    )

    headline_candidate = ""

    if lines:
        candidate = max(
            lines,
            key=lambda line: (
                line["bbox"][3] - line["bbox"][1],
                line["confidence"],
            ),
        )

        headline_candidate = candidate["text"]

    full_text = "\n".join(
        line["text"] for line in lines
    )

    result = {
        "status": (
            "text_detected"
            if lines
            else "no_text_detected"
        ),
        "source_image": str(
            image_path.resolve()
        ),
        "language": detect_language(full_text),
        "full_text": full_text,
        "headline_candidate": headline_candidate,
        "headline_status": (
            "needs_review"
            if lines
            else "unavailable"
        ),
        "text_regions": lines,
        "preview_path": str(preview_path),
        "output_directory": str(output_dir),
        "note": (
            "OCR and headline candidates must be reviewed; "
            "no watermark removal or image manipulation "
            "is performed."
        ),
    }

    manifest_path = output_dir / "analysis.json"

    result["manifest_path"] = str(manifest_path)

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return result
