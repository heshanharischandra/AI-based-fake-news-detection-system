"""Read-only dataset inspector for Component 4 PP1 (22 October 2026).

Run from the ImageVerificationSystem project root:
    python inspect_datasets.py

Optional:
    python inspect_datasets.py --dataset-dir "dataset/news_image_context_verification_dataset" --samples 3

Python standard library only. This does not train models, download images,
contact websites, modify CSV files, or verify that remote URLs are online.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


DATASET_NAME = "news_image_context_verification_dataset"
FILES = [
    ("News / master", "news/master_news_images.csv"),
    ("Cross-modal / all pairs", "cross_modal/all_pairs.csv"),
    ("Cross-modal / train", "cross_modal/train.csv"),
    ("Cross-modal / validation", "cross_modal/val.csv"),
    ("Cross-modal / test", "cross_modal/test.csv"),
    ("OCR / annotations", "ocr/annotations.csv"),
    ("Authenticity / labels", "authenticity/labels.csv"),
    ("Image reuse / pairs", "image_reuse/pairs.csv"),
    ("Visual / annotations", "visual/annotations.csv"),
    ("Context / reference events", "context/reference_events.csv"),
    ("Context / annotations", "context/annotations.csv"),
]


def separator(title: str) -> None:
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        fields = list(reader.fieldnames)
        rows = []
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items() if k is not None})
    return fields, rows


def looks_like_http_url(value: str) -> bool:
    try:
        result = urlsplit(value)
        return result.scheme in {"http", "https"} and bool(result.netloc)
    except ValueError:
        return False


def sample_text(value: str, limit: int = 145) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."


def print_samples(rows: list[dict[str, str]], count: int) -> None:
    for i, r in enumerate(rows[:count], 1):
        print(f"\n  Example {i}")
        for key in ("record_id", "headline", "pair_label", "language", "image_url", "article_url"):
            val = r.get(key, "")
            if val:
                print(f"    {key:16} : {sample_text(val)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect actual Component 4 dataset CSVs in the terminal")
    parser.add_argument(
        "--dataset-dir", default=str(Path("dataset") / DATASET_NAME),
        help="Directory containing news/, cross_modal/, ocr/, etc. (relative to current working directory)",
    )
    parser.add_argument("--samples", type=int, default=3, help="Number of master records to display (0 to 10)")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    root = Path(args.dataset_dir).expanduser().resolve()
    separator("COMPONENT 4 - PP1 DATASET INSPECTION")
    print("Project    : Context-Aware Image Verification & Cross-Modal Evidence Verification")