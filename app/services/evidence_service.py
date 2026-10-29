
import json
from pathlib import Path
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parents[2]

REFERENCE_METADATA_PATH = (
    BASE_DIR / "data" / "reference_images"
    / "reference_metadata.json"
)

NEWS_EVIDENCE_PATH = (
    BASE_DIR / "data" / "evidence" / "evidence.json"
)


def load_json_records(file_path):
    if not file_path.exists():
        return []

    with open(file_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            f"Expected a JSON list in {file_path}"
        )

    return records


def valid_source_url(url):
    if not isinstance(url, str):
        return False

    parsed = urlparse(url)

    return (
        parsed.scheme in ("http", "https")
        and bool(parsed.hostname)
    )


def is_eligible_evidence(record):
    return (
        record.get("evidence_status") == "verified"
        and valid_source_url(
            record.get("source_url")
        )
        and bool(record.get("source_name"))
    )


def get_matched_image_evidence(reuse_result):
    if not reuse_result:
        return []

    if reuse_result.get("possible_reuse") is not True:
        return []

    matched_filename = reuse_result.get(
        "matched_reference"
    )

    if not matched_filename:
        return []

    records = load_json_records(
        REFERENCE_METADATA_PATH
    )

    for record in records:
        if record.get("filename") != matched_filename:
            continue

        if not is_eligible_evidence(record):
            return []

        return [{
            "title": (
                record.get("original_claim")
                or record.get("event_name")
                or "Matched image reference"
            ),
            "summary": (
                f"Event: {record.get('event_name') or 'Unknown'}. "
                f"Location: {record.get('location') or 'Unknown'}."
            ),
            "source_name": record["source_name"],
            "source_url": record["source_url"],
            "publication_date": record.get(
                "publication_date"
            ),
            "event_date": record.get("event_date"),
            "evidence_type": "matched_image_reference",
            "matched_reference": matched_filename,
            "text_relevance": None,
            "supports_claim": "not_determined",
            "match_reason": "local_image_hash_match"
        }]

    return []


def get_news_evidence():
    records = load_json_records(
        NEWS_EVIDENCE_PATH
    )

    eligible = []

    for record in records:
        if not is_eligible_evidence(record):
            continue

        eligible.append({
            "title": record.get("title", ""),
            "summary": record.get("summary", ""),
            "source_name": record["source_name"],
            "source_url": record["source_url"],
            "publication_date": record.get(
                "publication_date"
            ),
            "event_date": record.get("event_date"),
            "evidence_type": "news_reference",
            "text_relevance": None,
            "supports_claim": "not_determined",
            "match_reason": "text_relevance"
        })

    return eligible


def rank_news_evidence(claim_text, records, top_k):
    if not claim_text or not claim_text.strip():
        return []

    if not records:
        return []

    documents = []

    for record in records:
        document = (
            f"{record.get('title', '')} "
            f"{record.get('summary', '')}"
        )
        documents.append(document)

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        vectors = vectorizer.fit_transform(
            [claim_text] + documents
        )

        similarities = cosine_similarity(
            vectors[0:1],
            vectors[1:]
        )[0]

    except ValueError:
        return []

    ranked = []

    for record, similarity in zip(
        records,
        similarities
    ):
        score = float(similarity)

        if score < 0.05:
            continue

        item = dict(record)
        item["text_relevance"] = round(score, 4)

        ranked.append(item)

    ranked.sort(
        key=lambda item: item["text_relevance"],
        reverse=True
    )

    return ranked[:top_k]


def retrieve_evidence(
    claim_text,
    reuse_result=None,
    top_k=3
):
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    image_evidence = get_matched_image_evidence(
        reuse_result
    )

    news_records = get_news_evidence()

    ranked_news = rank_news_evidence(
        claim_text,
        news_records,
        top_k
    )

    results = image_evidence + ranked_news

    unique_results = []
    seen_urls = set()

    for item in results:
        url = item["source_url"]

        if url in seen_urls:
            continue

        seen_urls.add(url)
        unique_results.append(item)

        if len(unique_results) >= top_k:
            break

    status = (
        "evidence_retrieved"
        if unique_results
        else "no_verified_evidence_available"
    )

    return {
        "status": status,
        "claim": claim_text,
        "evidence_count": len(unique_results),
        "results": unique_results,
        "method": "local_reference_metadata_and_tfidf",
        "note": (
            "Sources are selected from locally recorded "
            "evidence marked as verified. URLs are not "
            "automatically checked live. Text relevance "
            "does not establish whether a claim is true."
        )
    }
