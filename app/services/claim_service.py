
import math

from app.services.language_service import detect_language, normalize_text


def prepare_news_claim(ocr_result, translator=None):
    original = normalize_text(ocr_result.get("headline") or ocr_result.get("extracted_text") or "")
    language = detect_language(original)
    quality = ocr_result.get("headline_status") or ("needs_review" if original else "no_claim_detected")
    result = {"original_claim": original, "english_claim": "", "language": language,
              "claim_source": "ocr_unverified" if original else "no_claim_detected",
              "headline_status": quality, "translation_status": "not_attempted",
              "translation_verified": False}
    try:
        confidence = float(ocr_result.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0
    usable = len(original.split()) >= 2 and sum(c.isalpha() for c in original) >= 8 and "\ufffd" not in original
    reviewed = ocr_result.get("transcription_verified") is True
    accepted = quality == "candidate_accepted_for_demo" and not ocr_result.get("requires_review", False)
    confident = (ocr_result.get("status") == "success" and math.isfinite(confidence)
                 and confidence >= 80 and not ocr_result.get("requires_review", False))
    if not usable or not (reviewed or accepted or confident):
        return result
    result["headline_status"] = "reviewed_transcription" if reviewed else "high_confidence_ocr_candidate" if confident else quality
    result["claim_source"] = "reviewed_transcription" if reviewed else "ocr_candidate"
    if language == "en":
        result["english_claim"] = original
        result["translation_status"] = "not_needed"
    elif language in ("si", "si+en"):
        try:
            if translator is None:
                from app.services.translation_service import translate_to_english
                translator = translate_to_english
            translated = normalize_text(translator(original, language))
            if detect_language(translated) != "en" or "\ufffd" in translated:
                raise ValueError("Translation did not produce usable English text")
            result["english_claim"] = translated
            result["translation_status"] = "unverified_translation_candidate"
            result["claim_source"] = "translated_ocr_candidate"
        except Exception as error:
            result["translation_status"] = "translation_failed"
            result["translation_error"] = str(error)
    else:
        result["claim_source"] = "unknown_language"
    return result
