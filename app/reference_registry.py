"""Register human-verified local image references for context checks."""
from __future__ import annotations

import json
import shutil
import tempfile
from datetime import date
from pathlib import Path
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = BASE_DIR / "data" / "reference_images"
METADATA_PATH = REFERENCE_DIR / "reference_metadata.json"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
REQUIRED_FIELDS = (
    "source_name", "source_url", "publication_date", "event_date",
    "event_name", "location", "original_claim",
)


def _valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"https", "http"} and bool(parsed.hostname)


def _valid_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError):
        return False
    return True


def _load_records(metadata_path: Path) -> list[dict]:
    if not metadata_path.exists():
        return []
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("Reference metadata must be a JSON list of objects.")
    return records


def _write_records(metadata_path: Path, records: list[dict]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", prefix=".metadata-",
        dir=metadata_path.parent, delete=False,
    ) as temporary:
        json.dump(records, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    temporary_path.replace(metadata_path)


def register_verified_reference(
    image_path: str | Path,
    *,
    source_name: str,
    source_url: str,
    publication_date: str,
    event_date: str,
    event_name: str,
    location: str,
    original_claim: str,
    filename: str | None = None,
    reference_dir: Path = REFERENCE_DIR,
    metadata_path: Path = METADATA_PATH,
) -> dict:
    """Copy an image and store source-attested metadata for evidence checks.

    The caller is responsible for supplying an independently verified source.
    This function validates completeness but cannot verify a publisher's facts.
    """
    source = Path(image_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Reference image path does not exist or is not a file.")
    if source.suffix.casefold() not in SUPPORTED_SUFFIXES:
        raise ValueError("Reference image must be PNG, JPEG, or WebP.")

    values = {
        "source_name": str(source_name).strip(),
        "source_url": str(source_url).strip(),
        "publication_date": str(publication_date).strip(),
        "event_date": str(event_date).strip(),
        "event_name": str(event_name).strip(),
        "location": str(location).strip(),
        "original_claim": str(original_claim).strip(),
    }
    missing = [key for key in REQUIRED_FIELDS if not values[key]]
    if missing:
        raise ValueError("Missing required verified evidence fields: " + ", ".join(missing))
    if not _valid_http_url(values["source_url"]):
        raise ValueError("source_url must be a valid HTTP(S) URL.")
    for key in ("publication_date", "event_date"):
        if not _valid_date(values[key]):
            raise ValueError(f"{key} must use YYYY-MM-DD format.")

    requested_name = filename or source.name
    target_name = Path(requested_name).name
    if target_name != requested_name or target_name in {"", ".", ".."}:
        raise ValueError("filename must be a simple file name, not a path.")
    if Path(target_name).suffix.casefold() not in SUPPORTED_SUFFIXES:
        raise ValueError("filename must use a PNG, JPEG, or WebP extension.")

    reference_dir = Path(reference_dir).resolve()
    metadata_path = Path(metadata_path).resolve()
    if metadata_path.parent != reference_dir:
        raise ValueError("metadata_path must be inside reference_dir.")
    target = reference_dir / target_name
    records = _load_records(metadata_path)
    if target.exists() or any(str(record.get("filename", "")).casefold() == target_name.casefold() for record in records):
        raise ValueError(f"A reference named '{target_name}' already exists.")

    reference_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    record = {
        "filename": target_name,
        **values,
        "evidence_status": "verified",
        "verification_method": "manual_source_attestation",
    }
    try:
        _write_records(metadata_path, [*records, record])
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return record
