
from importlib import import_module
from time import monotonic

STAGES = (
    ("ocr", "Headline extraction"), ("claim", "News headline / claim"),
    ("visual", "Visual image analysis"), ("consistency", "Image–text consistency"),
    ("authenticity", "AI-generation analysis"), ("reuse", "Image reuse detection"),
    ("context", "Context verification"), ("score", "Verification scoring"),
    ("explanation", "Explanation"), ("evidence", "Evidence & references"),
)
SERVICE_FUNCTIONS = {
    "ocr": ("ocr_service", "extract_text"), "claim": ("claim_service", "prepare_news_claim"),
    "visual": ("visual_service", "describe_image"), "consistency": ("similarity_service", "compare_image_text"),
    "authenticity": ("authenticity_service", "check_ai_generation"), "reuse": ("reuse_service", "check_image_reuse"),
    "context": ("context_service", "verify_context"), "score": ("scoring_service", "calculate_verification_score"),
    "explanation": ("explanation_service", "generate_explanation"), "evidence": ("evidence_service", "retrieve_evidence"),
}


def run_pipeline(image_path, options, emit, services=None):
    """Report independent stage failures and preserve uncertainty in downstream checks."""
    results = {}

    def invoke(key, *args, **kwargs):
        # Import inside the stage: a missing optional model cannot prevent UI startup.
        if services is not None:
            return services[key](*args, **kwargs)
        module, name = SERVICE_FUNCTIONS[key]
        return getattr(import_module("app.services." + module), name)(*args, **kwargs)

    def stage(key, action):
        emit(key, "running", None, 0)
        started = monotonic()
        try:
            data = action()
            if not isinstance(data, dict):
                raise ValueError("The service did not return a result object")
            state = "skipped" if data.get("status") == "skipped" else "error" if data.get("error") else "completed"
        except Exception as error:
            data, state = {"status": "error", "error": str(error)[:1200]}, "error"
        results[key] = data
        emit(key, state, data, round(monotonic() - started, 2))
        return data

    ocr = stage("ocr", lambda: invoke("ocr", image_path))

    def claim_action():
        candidate = dict(ocr)
        headline = options.get("headline", "").strip()
        comparison = options.get("comparison_text", "").strip()
        if headline:
            # Review attests to transcription only, never factual truth.
            candidate.update(headline=headline, transcription_verified=True)
        if comparison:
            value = invoke("claim", candidate, translator=lambda *_: comparison)
            if value.get("original_claim"):
                value.update(english_claim=comparison, comparison_source="user_supplied", translation_verified=False)
        else:
            value = invoke("claim", candidate)
        value["headline_input"] = "user_supplied" if headline else "ocr"
        return value

    claim = stage("claim", claim_action)
    stage("visual", lambda: invoke("visual", image_path))
    english = claim.get("english_claim", "")
    stage("consistency", lambda: invoke("consistency", image_path, english) if english else {
        "status": "skipped", "reason": "No usable English comparison text. Review the headline or supply comparison text.",
        "similarity_score": None,
    })
    stage("authenticity", lambda: invoke("authenticity", image_path,
          input_contains_text_overlay=bool(ocr.get("extracted_text"))))
    reuse = stage("reuse", lambda: invoke("reuse", image_path))
    context_claim = english or claim.get("original_claim", "")
    context = stage("context", lambda: invoke("context", claim_text=context_claim, reuse_result=reuse))
    score = stage("score", lambda: invoke("score", context_result=context,
                  consistency_result=results["consistency"], authenticity_result=results["authenticity"]))
    if options.get("include_details"):
        stage("explanation", lambda: invoke("explanation", claim_result=claim, visual_result=results["visual"],
              consistency_result=results["consistency"], authenticity_result=results["authenticity"],
              reuse_result=reuse, context_result=context, score_result=score))
        stage("evidence", lambda: invoke("evidence", claim_text=context_claim, reuse_result=reuse))
    return results
