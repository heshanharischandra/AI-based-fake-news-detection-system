import json
import re
from datetime import date
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
METADATA_PATH = BASE_DIR / "data" / "reference_images" / "reference_metadata.json"


def load_reference_metadata():
    if not METADATA_PATH.exists():
        return []

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Reference metadata must be a JSON list.")
    return [record for record in records if isinstance(record, dict)]


def find_reference_metadata(filename):
    if not filename:
        return None

    target = Path(filename).name.casefold()
    return next(
        (
            record
            for record in load_reference_metadata()
            if str(record.get("filename") or "").casefold() == target
        ),
        None,
    )


def parse_event_date(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _base_result():
    return {
        "status": "insufficient_evidence",
        "event_check": "not_assessed",
        "time_check": "not_assessed",
        "location_check": "not_assessed",
        "claim_check": "not_assessed",
        "image_reuse": "unknown",
        "matched_reference": None,
        "nearest_reference_candidate": None,
        "source_name": None,
        "source_url": None,
        "publication_date": None,
        "event_date": None,
        "event_name": None,
        "location": None,
        "evidence_status": "missing",
        "missing_evidence": [],
        "explanations": [],
    }


def _explicit_claim_year(claim_text):
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", claim_text or ""))
    return int(next(iter(years))) if len(years) == 1 else None


def _assess_verified_metadata(result, claim_text, reference):
    """Assess only explicit, checkable details in a verified local record."""
    event_name = str(reference.get("event_name") or "").strip()
    location = str(reference.get("location") or "").strip()
    original_claim = str(reference.get("original_claim") or "").strip()
    event_date = parse_event_date(reference.get("event_date"))
    claim_lower = (claim_text or "").casefold()

    if event_name:
        event_terms = [
            word.casefold()
            for word in re.findall(r"[\w'-]+", event_name)
            if len(word) > 2
        ]
        if event_name.casefold() in claim_lower:
            result["event_check"] = "event_name_appears_in_claim"
        elif event_terms and sum(term in claim_lower for term in event_terms) / len(event_terms) >= 0.75:
            result["event_check"] = "possible_event_name_match_review_needed"
        else:
            result["event_check"] = "not_assessable_from_claim_text"
    else:
        result["missing_evidence"].append("event_name")
        result["event_check"] = "missing_reference_event_name"

    claimed_year = _explicit_claim_year(claim_text)
    if event_date and claimed_year is not None:
        result["time_check"] = (
            "year_appears_consistent"
            if event_date.year == claimed_year
            else "possible_time_mismatch"
        )
    elif event_date:
        result["time_check"] = "event_date_available_claim_has_no_explicit_year"
    else:
        result["missing_evidence"].append("valid_event_date")
        result["time_check"] = "missing_reference_event_date"

    if location:
        result["location_check"] = (
            "location_name_appears_in_claim"
            if location.casefold() in claim_lower
            else "location_not_stated_or_not_matched"
        )
    else:
        result["missing_evidence"].append("location")
        result["location_check"] = "missing_reference_location"

    if original_claim:
        # Exact phrase inclusion is a transparent lexical check, not semantic
        # entailment or independent fact-checking.
        result["claim_check"] = (
            "reference_claim_phrase_appears"
            if original_claim.casefold() in claim_lower
            else "claim_requires_independent_fact_check"
        )
    else:
        result["missing_evidence"].append("original_claim")
        result["claim_check"] = "missing_reference_claim"


def verify_context(claim_text, reuse_result):
    result = _base_result()
    claim_text = str(claim_text or "").strip()

    if not claim_text:
        result["missing_evidence"].append("usable_claim_text")
        result["explanations"].append(
            "No usable claim text was provided, so claim context could not be assessed."
        )

    if not isinstance(reuse_result, dict):
        result["missing_evidence"].append("image_reuse_result")
        result["explanations"].append(
            "Image reuse analysis was unavailable; no reference could be linked to this image."
        )
        return result

    reuse_status = reuse_result.get("status", "unknown")

    if reuse_result.get("possible_reuse") is False:
        result["image_reuse"] = "no_close_local_match"
        result["status"] = "insufficient_evidence_no_local_image_match"
        result["missing_evidence"].append("matched_reference_with_verified_source")
        nearest_candidate = reuse_result.get("nearest_reference_candidate") or reuse_result.get("matched_reference")
        if nearest_candidate:
            result["nearest_reference_candidate"] = nearest_candidate
        result["explanations"].append(
            "No close image match was found in the local reference collection. "
            "This does not rule out reuse elsewhere; event, date, location, and claim checks need a sourced reference."
        )
        return result

    if reuse_result.get("possible_reuse") is not True:
        result["image_reuse"] = reuse_status
        result["status"] = "insufficient_evidence_ambiguous_image_match"
        result["missing_evidence"].append("confirmed_image_reference")
        result["explanations"].append(
            "The image-reference match is unavailable or ambiguous, so it cannot anchor a context check."
        )
        return result

    filename = reuse_result.get("matched_reference")
    result["matched_reference"] = filename
    result["image_reuse"] = reuse_status
    reference = find_reference_metadata(filename)
    if reference is None:
        result["status"] = "reference_metadata_missing"
        result["evidence_status"] = "missing"
        result["missing_evidence"].extend(
            ["verified_source_name", "verified_source_url", "event_details"]
        )
        result["explanations"].append(
            "A similar local image was found, but no metadata record is available for that reference."
        )
        return result

    result["source_name"] = reference.get("source_name")
    result["source_url"] = reference.get("source_url")
    result["publication_date"] = reference.get("publication_date")
    result["event_date"] = reference.get("event_date")
    result["event_name"] = reference.get("event_name")
    result["location"] = reference.get("location")

    source_verified = (
        str(reference.get("evidence_status") or "").casefold() == "verified"
        and bool(reference.get("source_name"))
        and isinstance(reference.get("source_url"), str)
        and reference["source_url"].startswith(("https://", "http://"))
    )
    if not source_verified:
        result["status"] = "reference_source_unverified"
        result["evidence_status"] = "unverified"
        result["event_check"] = "not_assessed_unverified_source"
        result["time_check"] = "not_assessed_unverified_source"
        result["location_check"] = "not_assessed_unverified_source"
        result["claim_check"] = "not_assessed_unverified_source"
        if not reference.get("source_name"):
            result["missing_evidence"].append("source_name")
        if not reference.get("source_url"):
            result["missing_evidence"].append("source_url")
        result["missing_evidence"].append("independently_verified_reference_source")
        result["explanations"].append(
            "A visually similar image was found, but its source is missing or not marked verified. "
            "Context consistency is not reported as a factual result."
        )
        return result

    result["evidence_status"] = "verified"
    _assess_verified_metadata(result, claim_text, reference)

    if result["time_check"] == "possible_time_mismatch":
        result["status"] = "possible_context_mismatch"
        result["explanations"].append(
            "The claim's explicit year conflicts with the verified reference event date; review the source and claim."
        )
    elif result["missing_evidence"]:
        result["status"] = "partial_verified_context"
        result["explanations"].append(
            "Some verified reference metadata is available, but required event, date, location, or claim details are missing."
        )
    else:
        result["status"] = "reference_context_checks_available"
        result["explanations"].append(
            "Reference details were compared with explicit claim text. These checks do not establish the full claim as true."
        )

    return result
