
def generate_explanation(
    claim_result=None,
    visual_result=None,
    consistency_result=None,
    authenticity_result=None,
    reuse_result=None,
    context_result=None,
    score_result=None
):

    claim_result = claim_result or {}
    visual_result = visual_result or {}
    consistency_result = consistency_result or {}
    authenticity_result = authenticity_result or {}
    reuse_result = reuse_result or {}
    context_result = context_result or {}
    score_result = score_result or {}

    explanations = []

    # 1. NEWS CLAIM SOURCE

    claim_source = claim_result.get("claim_source")

    if claim_source == "automatic_ocr":
        explanations.append(
            "The news text was automatically extracted "
            "using OCR. Recognition errors may affect "
            "the subsequent verification."
        )

    elif claim_source == "manual_fallback":
        explanations.append(
            "The headline was manually provided because "
            "OCR did not find usable news text."
        )

    elif claim_source == "no_claim_detected":
        explanations.append(
            "No usable news claim was available."
        )

    # 2. VISUAL IMAGE ANALYSIS

    visual_caption = visual_result.get("caption")

    if visual_caption:
        explanations.append(
            "The visual model described the image as: "
            f"'{visual_caption}'. This description "
            "is not independent factual evidence."
        )

    # 3. IMAGE-TEXT CONSISTENCY

    similarity = consistency_result.get(
        "similarity_score"
    )

    if similarity is not None:
        explanations.append(
            "CLIP calculated an image-text cosine "
            f"similarity of {similarity:.4f}. "
            "This measures semantic association, "
            "not whether the news claim is true."
        )

    else:
        explanations.append(
            "Image-text consistency could not be "
            "evaluated because a valid result "
            "was unavailable."
        )

    # 4. AI-GENERATED IMAGE INDICATOR

    ai_score = authenticity_result.get(
        "ai_generated_score"
    )

    if ai_score is not None:

        percentage = ai_score * 100

        explanations.append(
            "The pretrained AI-image classifier "
            f"assigned {percentage:.2f}% to its "
            "AI-generated class. This is a model "
            "indicator, not proof that the image "
            "is synthetic or manipulated."
        )

    else:
        explanations.append(
            "AI-generation analysis was unavailable."
        )

    # 5. IMAGE REUSE

    possible_reuse = reuse_result.get(
        "possible_reuse"
    )

    if possible_reuse is True:

        reference = reuse_result.get(
            "matched_reference",
            "unknown"
        )

        distance = reuse_result.get(
            "hash_distance"
        )

        explanations.append(
            "A visually similar image was found "
            f"in the local reference collection: "
            f"{reference}. Perceptual hash distance: "
            f"{distance}. This does not establish "
            "historical or misleading reuse."
        )

    elif possible_reuse is False:

        explanations.append(
            "No close image match was found "
            "in the available local reference "
            "collection. This does not rule out "
            "reuse elsewhere."
        )

    else:

        explanations.append(
            "Image reuse could not be determined "
            "from the available reference collection."
        )

    # 6. CONTEXT VERIFICATION

    context_status = context_result.get(
        "status",
        "unavailable"
    )

    if context_status == "reference_unverified":

        explanations.append(
            "A matching reference was found, "
            "but its original source information "
            "has not been verified. Event, time, "
            "location, and claim conclusions "
            "cannot yet be established."
        )

    elif context_status == "possible_context_mismatch":

        explanations.append(
            "The available context analysis "
            "identified a possible mismatch. "
            "The supporting evidence should "
            "be checked before reaching a verdict."
        )

    elif context_status == "requires_additional_evidence":

        explanations.append(
            "Available reference information "
            "is insufficient for a complete "
            "contextual verification."
        )

    else:

        explanations.append(
            "Context verification status: "
            f"{context_status}."
        )

    # 7. VERIFICATION SCORE

    verification_score = score_result.get(
        "verification_score"
    )

    if verification_score is None:

        explanations.append(
            "The numerical verification score "
            "was withheld because sufficient "
            "verified context evidence "
            "was not available."
        )

    else:

        explanations.append(
            "The preliminary context-factor "
            "agreement score was "
            f"{verification_score:.2f}/100. "
            "This is not a calibrated probability "
            "that the news is true."
        )

    # 8. FINAL SUMMARY

    if context_status == "possible_context_mismatch":

        summary = (
            "Possible contextual inconsistency "
            "detected. Further evidence review "
            "is recommended."
        )

    elif context_status == "reference_unverified":

        summary = (
            "A visually similar reference image "
            "was found, but its original context "
            "has not been verified."
        )

    else:

        summary = (
            "Available verification signals "
            "have been analyzed, but further "
            "evidence may be required."
        )

    return {
        "method": "rule_based_factor_explanation",
        "summary": summary,
        "reasons": explanations,
        "context_status": context_status,
        "status": "explanation_generated"
    }
