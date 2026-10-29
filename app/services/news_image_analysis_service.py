
from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import pytesseract

# Use the existing Windows installation even when tesseract.exe is not on PATH.
if not shutil.which('tesseract'):
    candidate = Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    if candidate.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(candidate)
from pytesseract import Output
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
from app.services.ocr_service import configure_tesseract, select_ocr_language, available_languages, tessdata_config
from app.services.language_service import detect_language, normalize_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "uploads" / "processed"


def language_of(text: str) -> str:
    return detect_language(text)


def validate_bbox(bbox, size):
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox must be four pixel coordinates: x1,y1,x2,y2")
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width, height = size
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"bbox {bbox} is outside image {size}")
    return x1, y1, x2, y2


def roi_dialog(image_path, title):
    """Human confirmation of a meaningful region; returns None if cancelled."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("The interactive selector needs opencv-python and numpy") from exc
    with Image.open(image_path) as src:
        im = ImageOps.exif_transpose(src).convert('RGB')
    scale = min(1.0, 1100 / max(im.size))
    resized = im.resize((round(im.width * scale), round(im.height * scale)))
    preview = cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2BGR)
    print(f"[{title}] Drag around the area, press Enter; Esc cancels.", flush=True)
    try:
        x, y, w, h = cv2.selectROI(title, preview, showCrosshair=True, fromCenter=False)
    finally:
        cv2.destroyAllWindows()
    if w <= 0 or h <= 0:
        return None
    return validate_bbox((round(x/scale), round(y/scale),
                          min(im.width, round((x+w)/scale)),
                          min(im.height, round((y+h)/scale))), im.size)


def _ocr_variants(crop):
    """Return independent OCR readings; don't infer accuracy from model confidence."""
    gray = ImageOps.autocontrast(ImageOps.grayscale(crop))
    larger = ImageEnhance.Contrast(gray).enhance(1.3)
    larger = larger.resize((max(1, gray.width * 2), max(1, gray.height * 2)))
    results = []
    tests = [
        ("sin+eng", 6, crop),
        ("sin", 6, larger),
        ("sin+eng", 11, larger),
        ("eng", 7, larger),
    ]
    available = available_languages()
    for lang, psm, image in tests:
        if not set(lang.split('+')).issubset(available):
            continue
        txt = pytesseract.image_to_string(
            image, lang=lang, config=f"{tessdata_config()} --oem 3 --psm {psm}"
        )
        txt = normalize_text(txt)
        if txt:
            results.append({"text": txt, "language": language_of(txt),
                            "ocr_languages": lang, "psm": psm})
    return results


def _text_regions(image, min_confidence=30):
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    language = select_ocr_language()
    if language is None:
        return []
    data = pytesseract.image_to_data(gray, lang=language,
                                    config=f'{tessdata_config()} --oem 3 --psm 11',
                                    output_type=Output.DICT)
    grouped = defaultdict(list)
    for i, word in enumerate(data['text']):
        text = normalize_text(word)
        if not text:
            continue
        try:
            conf = float(data['conf'][i])
            x, y, w, h = [int(data[k][i]) for k in ('left','top','width','height')]
        except (ValueError, TypeError):
            continue
        if conf < min_confidence or w <= 0 or h <= 0:
            continue
        if not (0 <= x < x+w <= image.width and 0 <= y < y+h <= image.height):
            continue
        key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        grouped[key].append((x,y,x+w,y+h,text,conf))
    lines = []
    for values in grouped.values():
        values.sort(key=lambda v:v[0])
        raw_text = ' '.join(x[4] for x in values)
        bbox = [min(v[0] for v in values), min(v[1] for v in values),
                max(v[2] for v in values), max(v[3] for v in values)]
        watermark_candidate = bool(re.search(
            r'©|www\.|@\w+|youtube|subscribe|ada\s*derana|copyright',
            raw_text, flags=re.I
        ))
        lines.append({"text":raw_text, "language":language_of(raw_text),
                      "bbox":bbox, "ocr_confidence":round(sum(v[5] for v in values)/len(values),1),
                      "possible_text_overlay":watermark_candidate,
                      "review_status":"unreviewed"})
    return sorted(lines, key=lambda v:(v['bbox'][1],v['bbox'][0]))


def analyze_news_image(image_path, *, headline_boxes=None, photo_boxes=None,
                       watermark_boxes=None, min_confidence=30):
    """Save all crops and JSON manifest. Only supplied boxes count as reviewed ROIs.

    Coordinates are pixel corners in the ORIGINAL displayed orientation.
    headline_boxes/photo_boxes/watermark_boxes are lists of (x1,y1,x2,y2).
    Unprovided regions remain unknown rather than inferred as verified.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    if configure_tesseract() is None:
        raise RuntimeError("Tesseract is not installed or discoverable. Set TESSERACT_CMD or install Tesseract OCR.")
    if select_ocr_language() is None:
        raise RuntimeError("Tesseract needs eng.traineddata and/or sin.traineddata.")
    output_dir = OUTPUT_ROOT / f"{image_path.stem}_{uuid4().hex[:8]}"
    line_dir = output_dir / 'text_regions'
    photo_dir = output_dir / 'photo_regions'
    marks_dir = output_dir / 'watermark_regions'
    for directory in (line_dir, photo_dir, marks_dir):
        directory.mkdir(parents=True, exist_ok=True)
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    lines = _text_regions(image, min_confidence=min_confidence)
    for i, line in enumerate(lines,1):
        bbox=line['bbox']
        x1,y1,x2,y2 = bbox
        crop = image.crop((max(x1-5,0),max(y1-5,0),min(x2+5,image.width),min(y2+5,image.height)))
        dest=line_dir/f"line_{i:03d}_{line['language']}.png"
        crop.save(dest)
        line['crop_path']=str(dest.resolve())
        draw.rectangle(bbox,outline='#e02424',width=3)
    headline_lines=[]
    for i, bbox in enumerate(headline_boxes or [],1):
        coords=validate_bbox(bbox,image.size)
        crop=image.crop(coords)
        dest=line_dir/f'headline_selected_{i:02d}.png'
        crop.save(dest)
        variants=_ocr_variants(crop)
        headline_lines.append({"bbox":list(coords),"crop_path":str(dest.resolve()),
                               "ocr_variants":variants,"review_status":"needs_human_confirmation"})
        draw.rectangle(coords,outline='#20aa20',width=4)
    photos=[]
    for i,bbox in enumerate(photo_boxes or [],1):
        coords=validate_bbox(bbox,image.size)
        dest=photo_dir/f'photo_{i:02d}.png'
        image.crop(coords).save(dest)
        photos.append({"bbox":list(coords),"path":str(dest.resolve()),
                       "review_status":"user_selected_not_factually_verified"})
        draw.rectangle(coords,outline='#2563eb',width=4)
    marks=[]
    for i,bbox in enumerate(watermark_boxes or [],1):
        coords=validate_bbox(bbox,image.size)
        dest=marks_dir/f'overlay_{i:02d}.png'
        image.crop(coords).save(dest)
        marks.append({"bbox":list(coords),"path":str(dest.resolve()),
                      "status":"suspected_overlay_unverified"})
        draw.rectangle(coords,outline='#fb923c',width=3)
    preview_path=output_dir/'regions_preview.png'
    preview.save(preview_path)
    manifest={
        "status":"region_analysis_complete",
        "input_image":str(image_path.resolve()),
        "original_image_modified":False,
        "image_size":list(image.size),
        "text_regions":lines,
        "detected_full_text":'\n'.join(line['text'] for line in lines),
        "detected_language":language_of('\n'.join(line['text'] for line in lines)),
        "headline_regions":headline_lines,
        "headline_review_status":"needs_human_confirmation" if headline_lines else "not_selected",
        "photo_regions":photos,
        "visual_analysis_status":"ready_for_model" if photos else "no_photo_region_selected",
        "suspected_overlay_regions":marks,
        "watermark_handling":"detect_and_preserve_only; no inpainting or removal",
        "preview_path":str(preview_path.resolve()),
        "output_directory":str(output_dir.resolve()),
        "notes":["OCR results can be incorrect even with high confidence",
                 "BLIP may hallucinate place names; predictions require independent evidence",
                 "Photo regions in collages must be evaluated separately",
                 "An overlay is not proof of manipulation or misinformation"]
    }
    dest=output_dir/'analysis.json'
    manifest['manifest_path']=str(dest.resolve())
    with dest.open('w',encoding='utf-8') as f:
        json.dump(manifest,f,indent=2,ensure_ascii=False)
    return manifest
