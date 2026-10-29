from urllib.parse import urlparse

SUPPORTING_RESULTS = {
    "supported",
    "consistent",
    "verified_match",
    "date_appears_consistent",
    "year_appears_consistent",
    "event_name_appears_in_claim",
    "location_name_appears_in_claim",
    "reference_claim_phrase_appears",
}

CONFLICTING_RESULTS = {
    "contradicted",
    "inconsistent",
    "possible_time_mismatch",
    "possible_mismatch"
}

CONTEXT_FACTORS = {
    "event": "event_check",
    "time": "time_check",
    "location": "location_check",
    "claim": "claim_check"
}


def calculate_verification_score(
    context_result,
    consistency_result=None,
    authenticity_result=None
):
    context_result = context_result or {}
    consistency_result = consistency_result or {}
    authenticity_result = authenticity_result or {}

    source_url = context_result.get("source_url")
    try:
        parsed_source = urlparse(source_url if isinstance(source_url, str) else "")
        valid_source = parsed_source.scheme in ("https", "http") and bool(parsed_source.hostname)
    except ValueError:
        valid_source = False
    evidence_verified = (
        context_result.get("evidence_status") == "verified"
        and valid_source
    )

    evaluated_factors = {}
    missing_factors = []

    supporting_factors = []
    conflicting_factors = []
    factor_details = {}

    for factor, field in CONTEXT_FACTORS.items():
        status = context_result.get(
            field,
            "insufficient_evidence"
        )
        factor_details[factor] = {
            "context_status": status,
            "assessment": "not_evaluated",
            "reason": "No verified source with a valid HTTP(S) URL is available.",
        }

        if not evidence_verified:
            missing_factors.append(factor)
            continue

        if status in SUPPORTING_RESULTS:
            evaluated_factors[factor] = status
            supporting_factors.append(factor)
            factor_details[factor].update(assessment="supporting", reason="Supporting context check with a locally verified source.")

        elif status in CONFLICTING_RESULTS:
            evaluated_factors[factor] = status
            conflicting_factors.append(factor)
            factor_details[factor].update(assessment="conflicting", reason="Context check reports a conflict requiring review.")

        else:
            missing_factors.append(factor)
            factor_details[factor]["reason"] = (
                "The context check has no conclusive assessment; phrase overlap alone is not factual support."
            )

    total_factors = len(CONTEXT_FACTORS)

    evidence_coverage = round(
        len(evaluated_factors) / total_factors * 100,
        2
    )

    verification_score = None
    score_status = "insufficient_verified_evidence"
    risk_category = "undetermined"
    # Report partial progress separately; it must never masquerade as an
    # overall score when some of the four factors remain unassessed.
    assessed_factor_agreement = (
        round(len(supporting_factors) / len(evaluated_factors) * 100, 2)
        if evaluated_factors else None
    )

    if conflicting_factors:
        risk_category = "context_review_required"

    if evidence_verified and evaluated_factors:
        score_status = "partial_context_evidence"

    if (
        evidence_verified
        and len(evaluated_factors) == total_factors
    ):
        verification_score = round(
            len(supporting_factors) / total_factors * 100,
            2
        )

        score_status = "preliminary_context_alignment_score"

        if conflicting_factors:
            risk_category = "context_review_required"
        else:
            risk_category = "no_context_conflict_identified"

    explanations = []

    if not evidence_verified:
        explanations.append(
            "Verified reference evidence is not available."
        )

    if missing_factors:
        explanations.append(
            "Additional evidence is required for: "
            + ", ".join(missing_factors)
            + "."
        )

    if conflicting_factors:
        explanations.append(
            "Possible contextual inconsistencies were "
            "identified in: "
            + ", ".join(conflicting_factors)
            + "."
        )

    if verification_score is None:
        explanations.append(
            "A numerical verification score has been "
            "withheld because the available evidence "
            "is incomplete."
        )

    return {
        "verification_score": verification_score,
        "scoring_type": "evidence_gated_rules",
        "training_status": "not_a_trained_model",
        "accuracy_status": "not_measured_without_labeled_holdout",
        "assessed_factor_agreement_percent": assessed_factor_agreement,
        "assessed_factor_count": len(evaluated_factors),
        "total_factor_count": total_factors,
        "factor_details": factor_details,
        "source_eligible": bool(evidence_verified),
        "calibration_status": "rule_based_not_accuracy_calibrated",
        "trained_context": consistency_result.get("trained_context", {"status": "unavailable"}),
        "score_status": score_status,
        "risk_category": risk_category,
        "context_evidence_coverage_percent": evidence_coverage,
        "evaluated_factors": evaluated_factors,
        "missing_factors": missing_factors,
        "supporting_factors": supporting_factors,
        "conflicting_factors": conflicting_factors,
        "context_status": context_result.get(
            "status",
            "unavailable"
        ),
        "semantic_similarity": consistency_result.get(
            "similarity_score"
        ),
        "ai_generation_indicator": authenticity_result.get(
            "ai_generated_score"
        ),
        "explanations": explanations,
        "method": "evidence_gated_factor_assessment_v2",
        "note": (
            "The numerical score, if available, "
            "measures preliminary context-factor agreement. "
            "It is not a calibrated news-truth probability. "
            "CLIP and AI-generation scores are displayed "
            "separately and are not treated as truth evidence."
        )
    }
