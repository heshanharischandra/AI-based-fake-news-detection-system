from pathlib import Path
from tkinter import Tk, filedialog
from uuid import uuid4
from datetime import datetime, timezone
import math
import shutil
import json

from PIL import Image

from app.services.ocr_service import extract_text, detect_language
from app.services.claim_service import prepare_news_claim
from app.services.language_service import configure_unicode_console
from app.services.visual_service import describe_image
from app.services.similarity_service import compare_image_text
from app.services.authenticity_service import check_ai_generation
from app.services.reuse_service import check_image_reuse
from app.services.context_service import verify_context
from app.services.scoring_service import calculate_verification_score
from app.services.explanation_service import generate_explanation
from app.services.evidence_service import retrieve_evidence
from app.config import Settings
from app.database import MongoDatabase


# =============================================================
# IMAGE SELECTION
# =============================================================

def select_image():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        image_path = filedialog.askopenfilename(
            title="Select a News Image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.webp"),
                ("All Files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    return image_path


# =============================================================
# HEADING
# =============================================================

def heading(title):
    print()
    print("=" * 65)
    print(title)
    print("=" * 65)


# =============================================================
# HEADLINE VALIDATION
# =============================================================

def is_usable_headline(text):
    text = text.strip()

    word_count = len(text.split())
    letter_count = sum(char.isalpha() for char in text)

    return word_count >= 2 and letter_count >= 8


# =============================================================
# STEP 2: NEWS HEADLINE / CLAIM
# =============================================================

def get_news_claim(ocr_result):
    heading("STEP 2: NEWS HEADLINE / CLAIM")
    result = prepare_news_claim(ocr_result)
    print("Headline Candidate:", result["original_claim"] or "Not available")
    print("Detected Language:", result["language"])
    if result["claim_source"] == "ocr_unverified":
        print("OCR text requires review before it can be used as a claim.")
        print("Translation and CLIP are skipped to avoid comparing corrupted text.")
    if result["language"] in ("si", "si+en"):
        print("Translation Status:", result["translation_status"])
        if result.get("translation_error"):
            print("Translation Error:", result["translation_error"])
        if result["english_claim"]:
            print("English translation is a model candidate; Sinhala context accuracy has not been validated.")
    print("\nOriginal Claim:", result["original_claim"] or "Not available")
    print("English Claim:", result["english_claim"] or "Not available")
    print("Claim Source:", result["claim_source"])
    return result


def make_mongodb_safe(value):
    """Convert model outputs into values accepted by BSON."""
    if isinstance(value, dict):
        return {str(key): make_mongodb_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_mongodb_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return make_mongodb_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def persist_demo_analysis(analysis, database=None):
    """Persist a completed CLI analysis without making MongoDB a hard dependency."""
    owns_database = database is None
    if database is None:
        settings = Settings.from_env()
        database = MongoDatabase(settings, collection_name=settings.mongodb_demo_collection)
    try:
        if not database.connect():
            print("MongoDB Save: unavailable -", database.health().get("error", "not configured"))
            return False
        database.save_analysis(make_mongodb_safe(analysis))
        print("MongoDB Save: completed")
        print("MongoDB Record ID:", analysis["id"])
        return True
    except RuntimeError as error:
        print("MongoDB Save: failed -", error)
        return False
    finally:
        if owns_database:
            database.close()


# =============================================================
# MAIN PIPELINE
# =============================================================

def main():
    started_at = datetime.now(timezone.utc).isoformat()
    configure_unicode_console()
    heading(
        "CONTEXT-AWARE IMAGE-BASED CROSS-MODAL VERIFICATION"
    )

    print("\nPlease select a news image...")

    selected_file = select_image()

    if not selected_file:
        print("No image selected.")
        return

    source = Path(selected_file)

    # ---------------------------------------------------------
    # VALIDATE SELECTED IMAGE
    # ---------------------------------------------------------

    try:
        with Image.open(source) as image:
            image.verify()

    except Exception as error:
        print("Invalid image:", error)
        return

    # ---------------------------------------------------------
    # SAVE IMAGE TO UPLOADS DIRECTORY
    # ---------------------------------------------------------

    uploads = Path("uploads")
    uploads.mkdir(parents=True, exist_ok=True)

    analysis_id = uuid4().hex
    destination = uploads / (
        f"{analysis_id[:8]}_{source.name}"
    )

    try:
        shutil.copy2(source, destination)

    except Exception as error:
        print("Unable to save image:", error)
        return

    print("\nSelected Image:", source.name)
    print("Saved Image:", destination)

    # ---------------------------------------------------------
    # STEP 1: OCR TEXT EXTRACTION
    # ---------------------------------------------------------

    heading("STEP 1: OCR TEXT EXTRACTION")

    try:
        ocr = extract_text(destination)

    except Exception as error:
        print("OCR Error:", error)

        ocr = {
            "extracted_text": "",
            "language": "unknown",
            "status": "error",
        }

    extracted_headline = str(ocr.get("extracted_text") or "").strip()
    headline_language = detect_language(extracted_headline)
    language_labels = {
        "si": "Sinhala (si)",
        "en": "English (en)",
        "si+en": "Sinhala + English (si+en)",
        "unknown": "Unknown",
    }
    print("OCR Status:", ocr.get("status") or "unknown")
    print("Detected Language:", language_labels.get(headline_language, "Unknown"))
    print("\nExtracted Text:")
    print(extracted_headline or "No main headline detected.")

    claim_result = get_news_claim(ocr)
    claim = claim_result["english_claim"]
    # Keep a readable Unicode copy even if the terminal lacks Sinhala shaping.
    transcript_path = destination.with_suffix(destination.suffix + ".text.json")
    try:
        transcript_path.write_text(json.dumps({"ocr": ocr, "claim": claim_result},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as error:
        print("Unable to save UTF-8 text result:", error)

    visual_path = str(destination)
    screenshot_has_overlays = bool(ocr.get("extracted_text"))

    visual = None
    consistency = None
    authenticity = None
    reuse_result = None
    context_result = None
    score_result = None
    xai_result = None
    evidence_result = None

    # ---------------------------------------------------------
    # STEP 3: VISUAL IMAGE ANALYSIS
    # ---------------------------------------------------------

    heading("STEP 3: VISUAL IMAGE ANALYSIS")

    try:
        if not visual_path:
            raise ValueError("No valid visual candidate was extracted")
        visual = describe_image(visual_path)

        print("Visual Status:", visual.get("status", "unknown"))
        print("Model:", visual.get("model", "unknown"))
        print("Device:", visual.get("device", "unknown"))

        print("\nModel Visual Description (unverified; location not determined):")
        print(
            visual.get("caption")
            or "No caption available"
        )

    except Exception as error:
        visual = None

        print("Visual Analysis Error:", error)
        print("Visual Analysis Status: unavailable")

    # ---------------------------------------------------------
    # STEP 4: IMAGE-TEXT CONSISTENCY
    # ---------------------------------------------------------

    heading("STEP 4: IMAGE-TEXT CONSISTENCY")

    if claim:
        try:
            consistency = compare_image_text(
                str(visual_path),
                claim,
            )

            print(
                "Headline:",
                consistency.get("headline", claim),
            )

            print(
                "CLIP Similarity:",
                consistency.get(
                    "similarity_score",
                    "unavailable",
                ),
            )

            print(
                "Status:",
                consistency.get("status", "unknown"),
            )

            print(
                "Model:",
                consistency.get("model", "unknown"),
            )

            print(
                "\nNote: CLIP similarity is not a probability "
                "that the news is true."
            )

        except Exception as error:
            consistency = None

            print("CLIP Error:", error)
            print(
                "Image-text verification status: unavailable"
            )

    else:
        print(
            "Image-text verification skipped: "
            "No usable English claim available."
        )

    # ---------------------------------------------------------
    # STEP 5: IMAGE AUTHENTICITY VERIFICATION
    # ---------------------------------------------------------

    heading("STEP 5: IMAGE AUTHENTICITY VERIFICATION")

    try:
        if not visual_path:
            raise ValueError("No visual image is available for authenticity analysis")
        authenticity = check_ai_generation(
            str(visual_path),
            input_contains_text_overlay=screenshot_has_overlays,
        )

        print(
            "Model:",
            authenticity.get("model", "unknown"),
        )

        print(
            "Device:",
            authenticity.get("device", "unknown"),
        )

        print("\nAI-Generation Analysis:")

        print("Detector:", authenticity.get("model", "unknown"))
        model_score = authenticity.get("model_score")
        print(
            "Top Class Model Score:",
            f"{model_score * 100:.2f}%" if model_score is not None else "unavailable",
        )

        real_score = authenticity.get("real_score")
        ai_score = authenticity.get("ai_generated_score")

        if real_score is not None:
            print(
                "Real Class Model Score (uncalibrated):",
                f"{real_score * 100:.2f}%",
            )
        else:
            print("Real Class Model Score: unavailable")

        if ai_score is not None:
            print(
                "AI Class Model Score (uncalibrated):",
                f"{ai_score * 100:.2f}%",
            )
        else:
            print("AI Class Model Score: unavailable")

        print(
            "Decision Thresholds:",
            authenticity.get("decision_threshold", "unavailable"),
        )
        print(
            "Augmentation Disagreement:",
            authenticity.get("augmentation_disagreement", "unavailable"),
        )
        print(
            "Augmentation Views:",
            authenticity.get("augmentation_count", "unavailable"),
        )

        authenticity_status = authenticity.get(
            "status",
            "unknown",
        )

        print("\nDetection Status:", authenticity_status)
        for reason in authenticity.get("review_reasons", []):
            print("Review:", reason)
        if authenticity.get("error"):
            print("Authenticity Analysis Error:", authenticity["error"])

    except Exception as error:
        authenticity = None

        print("Authenticity Analysis Error:", error)
        print("Authenticity Status: unavailable")

    # ---------------------------------------------------------
    # STEP 6: IMAGE REUSE DETECTION
    # ---------------------------------------------------------

    heading("STEP 6: IMAGE REUSE DETECTION")

    try:
        reuse_result = check_image_reuse(
            str(visual_path or destination)
        )

        print(
            "Status:",
            reuse_result.get("status", "unknown"),
        )

        print(
            "References Checked:",
            reuse_result.get("reference_count", 0),
        )

        possible_reuse = reuse_result.get(
            "possible_reuse"
        )

        print("Possible Reuse:", possible_reuse)

        if possible_reuse is True:
            print("\nPossible Matching Image Found")

            print(
                "Matched Reference:",
                reuse_result.get(
                    "matched_reference",
                    "unavailable",
                ),
            )

            print(
                "Hash Distance:",
                reuse_result.get(
                    "hash_distance",
                    "unavailable",
                ),
            )

            print(
                "Match Strength:",
                reuse_result.get(
                    "match_strength",
                    "unavailable",
                ),
            )

            print("\nExplanation:")

            print(
                "A visually similar image was found in the "
                "local reference collection."
            )

            print(
                "Further context verification is needed to "
                "determine whether the image was reused for "
                "another event, time, or location."
            )

        elif possible_reuse is False:
            print("\nNo Close Image Match Detected")

            print(
                "Closest Reference:",
                reuse_result.get(
                    "nearest_reference_candidate",
                    "unavailable",
                ),
            )

            print(
                "Hash Distance:",
                reuse_result.get(
                    "hash_distance",
                    "unavailable",
                ),
            )

            print("\nExplanation:")

            print(
                "No close image match was found in the "
                "available local reference collection."
            )

        elif reuse_result.get("status") == "ambiguous_near_duplicate":
            print("\nSimilar image found; the match needs review.")
            print("Candidate Reference:", reuse_result.get("nearest_reference_candidate"))
        else:
            print("\nImage reuse verification unavailable.")

            print(
                "No usable reference-image comparison "
                "was available."
            )

        if reuse_result.get("failed_references"):
            print("Incomplete comparisons:", len(reuse_result["failed_references"]))

    except Exception as error:
        reuse_result = None

        print("Image Reuse Error:", error)
        print("Reuse Verification Status: unavailable")

    # ---------------------------------------------------------
    # STEP 7: CONTEXT VERIFICATION
    # ---------------------------------------------------------

    heading("STEP 7: CONTEXT VERIFICATION")

    context_claim = (
        claim_result["english_claim"]
        or claim_result["original_claim"]
    )

    try:
        context_result = verify_context(
            claim_text=context_claim,
            reuse_result=reuse_result,
        )

        print(
            "Context Status:",
            context_result.get("status", "unknown"),
        )

        print("\nREFERENCE IMAGE INFORMATION")

        print(
            "Matched Reference:",
            context_result.get("matched_reference")
            or "Not available",
        )

        nearest_candidate = context_result.get(
            "nearest_reference_candidate"
        )
        if nearest_candidate:
            print(
                "Nearest Local Candidate (not a match):",
                nearest_candidate,
            )

        print(
            "Original Source:",
            context_result.get("source_name")
            or "Not verified",
        )

        print(
            "Source URL:",
            context_result.get("source_url")
            or "Not available",
        )

        print(
            "Publication Date:",
            context_result.get("publication_date")
            or "Not verified",
        )

        print(
            "Original Event Date:",
            context_result.get("event_date")
            or "Not verified",
        )

        print(
            "Original Event:",
            context_result.get("event_name")
            or "Not verified",
        )

        print(
            "Original Location:",
            context_result.get("location")
            or "Not verified",
        )

        print("\nCONTEXT CONSISTENCY CHECKS")

        for label, key in [
            ("Event Consistency", "event_check"),
            ("Time Consistency", "time_check"),
            ("Location Consistency", "location_check"),
            ("Claim Consistency", "claim_check"),
            ("Image Reuse", "image_reuse"),
        ]:
            print(
                f"{label}:",
                context_result.get(
                    key,
                    "unavailable",
                ),
            )

        print(
            "\nEvidence Status:",
            context_result.get(
                "evidence_status",
                "unknown",
            ),
        )

        missing_context_evidence = context_result.get(
            "missing_evidence",
            [],
        )
        if missing_context_evidence:
            print(
                "Missing Evidence Needed:",
                ", ".join(missing_context_evidence),
            )

        print("\nCONTEXT VERIFICATION RESULT")

        print(
            "Overall Status:",
            context_result.get(
                "status",
                "unknown",
            ),
        )

    except Exception as error:
        context_result = None

        print("Context Verification Error:", error)
        print("Context Verification Status: unavailable")

    # ---------------------------------------------------------
    # STEP 8: VERIFICATION SCORING
    # ---------------------------------------------------------

    heading("STEP 8: VERIFICATION SCORING")

    try:
        score_result = calculate_verification_score(
            context_result=context_result,
            consistency_result=consistency,
            authenticity_result=authenticity,
        )

        print("\nSOURCE-BASED VERIFICATION SCORE")

        verification_score = score_result.get(
            "verification_score"
        )

        if verification_score is not None:
            print(
                "Verification Score:",
                f"{verification_score:.2f}/100",
            )
        else:
            print("Verification Score: Pending sufficient verified context evidence")

        assessed_count = score_result.get("assessed_factor_count", 0)
        total_count = score_result.get("total_factor_count", 4)
        print(f"Context Factors Assessed: {assessed_count}/{total_count}")
        partial_agreement = score_result.get("assessed_factor_agreement_percent")
        if partial_agreement is not None and verification_score is None:
            print(
                "Agreement Among Assessed Factors Only:",
                f"{partial_agreement:.2f}% (not an overall verification score)",
            )
        print(
            "Score Status:",
            score_result.get(
                "score_status",
                "unknown",
            ),
        )

        print(
            "Risk Category:",
            score_result.get(
                "risk_category",
                "undetermined",
            ),
        )

        print("\nEVIDENCE COVERAGE")

        coverage = score_result.get(
            "context_evidence_coverage_percent"
        )

        if coverage is not None:
            print(
                "Context Evidence Coverage:",
                f"{coverage:.2f}%",
            )
        else:
            print(
                "Context Evidence Coverage: NOT AVAILABLE"
            )

        print(
            "Supporting Factors:",
            score_result.get(
                "supporting_factors",
                [],
            ),
        )

        print(
            "Conflicting Factors:",
            score_result.get(
                "conflicting_factors",
                [],
            ),
        )

        print(
            "Missing Factors:",
            score_result.get(
                "missing_factors",
                [],
            ),
        )

        print("\nMODEL INDICATORS")

        if consistency is not None:
            print(
                "CLIP Similarity:",
                consistency.get(
                    "similarity_score",
                    "unavailable",
                ),
            )
        else:
            print("CLIP Similarity: unavailable")

        if authenticity is not None:
            ai_score = authenticity.get(
                "ai_generated_score"
            )

            if ai_score is not None:
                print(
                    "AI-Generation Indicator:",
                    f"{ai_score * 100:.2f}%",
                )
            else:
                print(
                    "AI-Generation Indicator: unavailable"
                )
        else:
            print(
                "AI-Generation Indicator: unavailable"
            )

    except Exception as error:
        score_result = None

        print("Scoring Error:", error)
        print("Verification Score: NOT CALCULATED")
        print("Score Status: unavailable")

    # ---------------------------------------------------------
    # STEP 9: EXPLAINABLE AI (XAI) - OUTPUT DISABLED
    # The original implementation is retained below as comments so it can be
    # re-enabled later. No XAI explanation is generated or printed in this run.
    # ---------------------------------------------------------
    xai_result = None
    # heading("STEP 9: EXPLAINABLE AI (XAI)")
    # xai_result = generate_explanation(
    #     claim_result=claim_result,
    #     visual_result=visual,
    #     consistency_result=consistency,
    #     authenticity_result=authenticity,
    #     reuse_result=reuse_result,
    #     context_result=context_result,
    #     score_result=score_result,
    # )
    # print("XAI Status:", xai_result.get("status", "unknown"))
    # print("Method:", xai_result.get("method", "unknown"))
    # print("FINAL EXPLANATION SUMMARY")
    # print(xai_result.get("summary", "No summary available."))
    # print("WHY DID THE SYSTEM PRODUCE THIS RESULT?")
    # for number, reason in enumerate(xai_result.get("reasons", []), start=1):
    #     print(f"{number}. {reason}")
    # print("CONTEXT STATUS:", xai_result.get("context_status", "unavailable"))
    # print("IMPORTANT NOTE: Explanations are not independent proof.")

    # ---------------------------------------------------------
    # STEP 10: EVIDENCE AND REFERENCE RETRIEVAL - OUTPUT DISABLED
    # The original retrieval and display implementation is retained below as
    # comments for future use. Evidence is not retrieved in this run.
    # ---------------------------------------------------------
    evidence_result = None
    # heading("STEP 10: EVIDENCE AND REFERENCE RETRIEVAL")
    # evidence_result = retrieve_evidence(
    #     claim_text=context_claim,
    #     reuse_result=reuse_result,
    #     top_k=3,
    # )
    # print("Evidence Status:", evidence_result.get("status", "unknown"))
    # print("Retrieval Method:", evidence_result.get("method", "unknown"))
    # print("Evidence Records Found:", evidence_result.get("evidence_count", 0))
    # for evidence in evidence_result.get("results", []):
    #     print("Source:", evidence.get("source_name"), evidence.get("source_url"))
    # print("No verified evidence available.")

    # ---------------------------------------------------------
    # COMPLETION SUMMARY
    # ---------------------------------------------------------

    heading("CURRENT VERIFICATION COMPLETED")

    persist_demo_analysis({
        "id": analysis_id,
        "record_type": "cli_demo_analysis",
        "filename": source.name,
        "created_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "image": {
            "saved_path": str(destination),
            "transcript_path": str(transcript_path),
        },
        "results": {
            "ocr": ocr,
            "claim": claim_result,
            "visual": visual,
            "consistency": consistency,
            "authenticity": authenticity,
            "reuse": reuse_result,
            "context": context_result,
            "score": score_result,
            "explanation": xai_result,
            "evidence": evidence_result,
        },
    })



# =============================================================
# APPLICATION ENTRY POINT
# =============================================================

if __name__ == "__main__":
    main()
