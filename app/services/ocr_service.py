
from __future__ import annotations

import os
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pytesseract import Output
from app.services.language_service import detect_language, normalize_text

_WINDOWS_TESSERACT_PATHS = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
)
_DEFAULT_PSM_MODES = (6, 11)


def _selection_config() -> dict:
    selection = Path(__file__).resolve().parents[2] / "models/sinhala_ocr/selection.json"
    if not selection.is_file():
        return {}
    try:
        value = json.loads(selection.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def configure_tesseract() -> str | None:
    """Find Tesseract from configuration, PATH, or common Windows locations."""
    configured = os.environ.get("TESSERACT_CMD")
    candidates = [Path(configured)] if configured else []
    path_command = shutil.which("tesseract")
    if path_command:
        candidates.append(Path(path_command))
    candidates.extend(_WINDOWS_TESSERACT_PATHS)
    for candidate in candidates:
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return str(candidate)
    return None


def tessdata_config() -> str:
    
    directory = os.environ.get("SINHALA_TESSDATA_DIR")
    if not directory:
        directory = _selection_config().get("tessdata_dir")
    if directory and (Path(directory) / "sin.traineddata").is_file() and (Path(directory) / "eng.traineddata").is_file():
        directory = Path(directory).resolve().as_posix()
        if os.name == "nt" and any(char.isspace() for char in directory):
            # pytesseract's Windows shlex mode preserves literal quotes. Use
            # the OS-provided short path where available, never shell escaping.
            import ctypes
            buffer = ctypes.create_unicode_buffer(32768)
            length = ctypes.windll.kernel32.GetShortPathNameW(directory, buffer, len(buffer))
            if not length or length >= len(buffer) or any(c.isspace() for c in buffer.value):
                raise ValueError("Put SINHALA_TESSDATA_DIR in a path without spaces on this Windows system")
            directory = Path(buffer.value).as_posix()
        if os.name == "nt":
            return f"--tessdata-dir {directory}"
        import shlex
        return f"--tessdata-dir {shlex.quote(directory)}"
    return ""


def runtime_psm_modes() -> tuple[int, ...]:
    configured = _selection_config().get("runtime_psm_modes", _DEFAULT_PSM_MODES)
    if (isinstance(configured, list) and configured
            and all(isinstance(mode, int) and 0 <= mode <= 13 for mode in configured)):
        return tuple(dict.fromkeys(configured))
    return _DEFAULT_PSM_MODES


def available_languages() -> set[str]:
    if configure_tesseract() is None:
        return set()
    try:
        return set(pytesseract.get_languages(config=tessdata_config()))
    except (OSError, pytesseract.TesseractNotFoundError):
        return set()


def select_ocr_language(languages: set[str] | None = None) -> str | None:
    installed = languages if languages is not None else available_languages()
    if {"sin", "eng"}.issubset(installed):
        return "sin+eng"
    if "sin" in installed:
        return "sin"
    if "eng" in installed:
        return "eng"
    return None


def _clean_text(text: str) -> str:
    lines = []
    for line in normalize_text(text).splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _image_variants(image: Image.Image) -> list[tuple[str, Image.Image]]:
    gray = ImageOps.grayscale(image)
    scale = 2 if max(gray.size) < 1800 else 1
    if scale > 1:
        gray = gray.resize((gray.width * scale, gray.height * scale), Image.Resampling.LANCZOS)
    enhanced = ImageEnhance.Contrast(ImageOps.autocontrast(gray)).enhance(1.35)
    sharp = enhanced.filter(ImageFilter.SHARPEN)
    threshold = sharp.point(lambda pixel: 255 if pixel > 175 else 0)
    return [("grayscale", gray), ("enhanced", sharp), ("threshold", threshold)]


def _read_pass_details(image: Image.Image, language: str, psm: int) -> tuple[list[dict], float]:
    data = pytesseract.image_to_data(
        image, lang=language, config=f"{tessdata_config()} --oem 3 --psm {psm}", output_type=Output.DICT
    )
    lines = defaultdict(list)
    for index, raw in enumerate(data.get("text", [])):
        word = str(raw or "").strip()
        try:
            confidence = float(data["conf"][index])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if word and math.isfinite(confidence) and confidence >= 0:
            def value(field):
                values = data.get(field, [])
                try:
                    return int(values[index])
                except (IndexError, TypeError, ValueError):
                    return 0

            key = tuple(value(field) for field in ("block_num", "par_num", "line_num"))
            lines[key].append({
                "text": word,
                "confidence": confidence,
                "left": value("left"),
                "top": value("top"),
                "width": value("width"),
                "height": value("height"),
            })

    records = []
    for key, words in lines.items():
        words.sort(key=lambda word: word["left"])
        chars = sum(len(word["text"]) for word in words)
        confidence = (sum(word["confidence"] * len(word["text"]) for word in words) / chars
                      if chars else 0.0)
        records.append({
            "key": key,
            "text": _clean_text(" ".join(word["text"] for word in words)),
            "confidence": confidence,
            "left": min(word["left"] for word in words),
            "top": min(word["top"] for word in words),
            "right": max(word["left"] + word["width"] for word in words),
            "bottom": max(word["top"] + word["height"] for word in words),
        })
    records.sort(key=lambda line: (line["top"], line["left"]))
    char_count = sum(len(word["text"]) for words in lines.values() for word in words)
    overall_confidence = (sum(word["confidence"] * len(word["text"])
                              for words in lines.values() for word in words) / char_count
                          if char_count else 0.0)
    return records, overall_confidence


def _read_pass(image: Image.Image, language: str, psm: int) -> tuple[str, float]:
    """Backward-compatible full-text helper used by the OCR validation tools."""
    lines, confidence = _read_pass_details(image, language, psm)
    return _clean_text("\n".join(line["text"] for line in lines)), confidence


def _merge_line_fragments(lines):
    """Join neighbouring OCR blocks on the same baseline before headline ranking.

    Sparse-text segmentation can split one bold Sinhala title into several
    blocks. Ranking those fragments independently favours a short, confident
    tail over the complete headline. Do not join columns, badges or other rows.
    """
    groups = []
    for line in sorted(lines, key=lambda row: (row["left"], row["top"])):
        line_height = line["bottom"] - line["top"]
        target = None
        nearest_gap = float("inf")
        for group in groups:
            group_height = group["bottom"] - group["top"]
            if min(line_height, group_height) <= 0:
                continue
            overlap = min(line["bottom"], group["bottom"]) - max(line["top"], group["top"])
            gap = line["left"] - group["right"]
            if (max(line_height, group_height) / min(line_height, group_height) <= 1.5
                    and overlap >= 0.7 * min(line_height, group_height)
                    and 0 <= gap <= 1.8 * min(line_height, group_height)
                    and gap < nearest_gap):
                target, nearest_gap = group, gap
        if target is None:
            groups.append(dict(line))
            continue
        left_weight, right_weight = len(target["text"]), len(line["text"])
        target["confidence"] = ((target["confidence"] * left_weight + line["confidence"] * right_weight)
                                / max(left_weight + right_weight, 1))
        target["text"] = _clean_text(target["text"] + " " + line["text"])
        target["top"] = min(target["top"], line["top"])
        target["bottom"] = max(target["bottom"], line["bottom"])
        target["right"] = max(target["right"], line["right"])
    return sorted(groups, key=lambda row: (row["top"], row["left"]))


def _headline_candidates(lines, image_size):
    """Rank short, prominent OCR line groups as headline candidates."""
    if not lines:
        return []
    lines = _merge_line_fragments(lines)
    image_width, image_height = image_size
    heights = [max(line["bottom"] - line["top"], 1) for line in lines]
    typical_height = sorted(heights)[len(heights) // 2]
    results = []
    excluded = re.compile(r"(?:https?://|www\.|\bsubscribe\b|\bfollow us\b|@[\w.]+)", re.I)

    for start, first in enumerate(lines):
        group = [first]
        for candidate in lines[start + 1:start + 3]:
            previous = group[-1]
            gap = candidate["top"] - previous["bottom"]
            previous_height = max(previous["bottom"] - previous["top"], 1)
            candidate_height = max(candidate["bottom"] - candidate["top"], 1)
            similar_font_size = max(previous_height, candidate_height) / min(previous_height, candidate_height) <= 1.5
            aligned_left = abs(candidate["left"] - first["left"]) <= max(image_width * 0.18, 24)
            if gap > max(typical_height * 1.15, 8) or not similar_font_size or not aligned_left:
                break
            group.append(candidate)

        for count in range(1, len(group) + 1):
            selected = group[:count]
            text = _clean_text(" ".join(line["text"] for line in selected))
            letters = sum(char.isalpha() for char in text)
            words = text.split()
            if letters < 5 or len(words) > 18 or excluded.search(text):
                continue
            height = max(line["bottom"] for line in selected) - min(line["top"] for line in selected)
            ratio = height / max(typical_height * count, 1)
            confidence = sum(line["confidence"] for line in selected) / count
            size_score = min(ratio, 2.0) / 2.0 * 35
            position_score = max(0.0, 1.0 - min(selected[0]["top"] / max(image_height, 1), 1.0)) * 10
            length_score = 10 if 2 <= len(words) <= 14 else 4
            score = confidence * 0.45 + size_score + position_score + length_score
            results.append({"text": text, "confidence": confidence, "score": score,
                            "top": selected[0]["top"], "left": selected[0]["left"],
                            "width": image_width})
    return results


def extract_text(image_path: str | Path, language_hint: str | None = None,
                 *, psm_modes: tuple[int, ...] | None = None) -> dict:
    modes = runtime_psm_modes() if psm_modes is None else psm_modes
    if not modes or any(type(mode) is not int or not 0 <= mode <= 13 for mode in modes):
        raise ValueError("psm_modes must contain integers from 0 through 13")
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    executable = configure_tesseract()
    if executable is None:
        return {"extracted_text": "", "language": "unknown", "ocr_engine": "Tesseract",
                "status": "ocr_unavailable",
                "error": "Tesseract is not installed or discoverable. Set TESSERACT_CMD or install Tesseract OCR with eng/sin language data."}
    installed = available_languages()
    language = select_ocr_language(installed)
    if language_hint is not None:
        requested = {"si": "sin", "en": "eng", "si+en": "sin+eng"}.get(language_hint)
        if requested is None:
            raise ValueError("language_hint must be si, en or si+en")
        language = requested if set(requested.split("+")).issubset(installed) else None
    if language is None:
        return {"extracted_text": "", "language": "unknown", "ocr_engine": "Tesseract",
                "status": "ocr_language_data_missing",
                "error": "No supported Tesseract language data found. Install eng and/or sin traineddata.",
                "available_languages": sorted(installed)}
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    results = []
    for variant_name, variant in _image_variants(image):
        for psm in dict.fromkeys(modes):
            try:
                lines, confidence = _read_pass_details(variant, language, psm)
            except (OSError, pytesseract.TesseractError) as error:
                return {"extracted_text": "", "language": "unknown", "ocr_engine": "Tesseract",
                        "status": "ocr_error", "error": str(error), "ocr_languages": language}
            # Coordinates are measured on the preprocessed variant (often 2x
            # upscaled), so normalize headline position using that same size.
            for candidate in _headline_candidates(lines, variant.size):
                results.append((candidate["score"], candidate["confidence"],
                                candidate["text"], variant_name, psm, candidate))
    if not results:
        return {"extracted_text": "", "language": "unknown", "ocr_engine": "Tesseract",
                "status": "no_text_found", "ocr_languages": language}
    results.sort(key=lambda item: item[0], reverse=True)
    _, confidence, text, variant_name, psm, candidate = results[0]
    return {"extracted_text": text, "language": detect_language(text), "ocr_engine": "Tesseract",
            "ocr_languages": language, "confidence": round(confidence, 1),
            "requires_review": confidence < 80 or "\ufffd" in text,
            "selected_pass": {"variant": variant_name, "psm": psm,
                              "headline_rank_score": round(candidate["score"], 1)},
            "status": "success"}
