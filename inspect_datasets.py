import csv
from pathlib import Path
from collections import Counter

# Project dataset location
BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    BASE_DIR
    / "dataset"
    / "news_image_context_verification_dataset"
)

DATASETS = {
    "Master News Images": "news/master_news_images.csv",
    "Cross Modal All Pairs": "cross_modal/all_pairs.csv",
    "Cross Modal Training": "cross_modal/train.csv",
    "Cross Modal Validation": "cross_modal/val.csv",
    "Cross Modal Testing": "cross_modal/test.csv",
    "OCR Annotations": "ocr/annotations.csv",
    "Authenticity Labels": "authenticity/labels.csv",
    "Image Reuse Pairs": "image_reuse/pairs.csv",
    "Visual Annotations": "visual/annotations.csv",
    "Context Reference Events": "context/reference_events.csv",
    "Context Annotations": "context/annotations.csv",
}

print("=" * 65)
print("COMPONENT 4 - NEWS IMAGE VERIFICATION DATASET INSPECTION")
print("=" * 65)

if not DATASET_DIR.exists():
    print("ERROR: Dataset folder not found!")
    print(DATASET_DIR)
    raise SystemExit(1)

total_records = 0

for dataset_name, relative_path in DATASETS.items():

    file_path = DATASET_DIR / relative_path

    print("\n" + "-" * 65)
    print("Dataset:", dataset_name)

    if not file_path.exists():
        print("Status: FILE NOT FOUND")
        continue

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    print("Status: FOUND")
    print("Records:", len(rows))

    total_records += len(rows)

    if rows:
        print("Columns:", ", ".join(rows[0].keys()))

        for label_col in (
            "pair_label",
            "label",
            "verite_label"
        ):
            if label_col in rows[0]:
                counts = Counter(
                    row.get(label_col, "")
                    for row in rows
                )
                print("Labels:", dict(counts))
                break

        if dataset_name == "Master News Images":
            print("\nSample News Records:")

            for row in rows[:3]:
                print("\nRecord ID:", row.get("record_id", "N/A"))

                for key in (
                    "headline",
                    "caption",
                    "claim_text",
                    "image_url",
                    "article_url"
                ):
                    if row.get(key):
                        print(f"{key}: {row[key]}")

print("\n" + "=" * 65)
print("DATASET INSPECTION COMPLETED")
print("Total CSV rows across files:", total_records)
print("NOTE: These are not all unique images.")
print("NOTE: Online image URLs were not checked.")
print("=" * 65)