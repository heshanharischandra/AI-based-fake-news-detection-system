"""Add an independently verified local image reference for context scoring."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.reference_registry import register_verified_reference


def main():
    parser = argparse.ArgumentParser(
        description="Copy a reference image and register verified source metadata."
    )
    parser.add_argument("--image", required=True, help="Local reference image path")
    parser.add_argument("--source-name", required=True, help="Publisher or fact-checker name")
    parser.add_argument("--source-url", required=True, help="Canonical HTTP(S) source URL")
    parser.add_argument("--publication-date", required=True, help="Source publication date: YYYY-MM-DD")
    parser.add_argument("--event-date", required=True, help="Original event date: YYYY-MM-DD")
    parser.add_argument("--event-name", required=True, help="Original event name")
    parser.add_argument("--location", required=True, help="Original event location")
    parser.add_argument("--original-claim", required=True, help="Claim documented by the source")
    parser.add_argument("--filename", help="Optional filename in data/reference_images")
    args = parser.parse_args()

    record = register_verified_reference(
        args.image,
        source_name=args.source_name,
        source_url=args.source_url,
        publication_date=args.publication_date,
        event_date=args.event_date,
        event_name=args.event_name,
        location=args.location,
        original_claim=args.original_claim,
        filename=args.filename,
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
