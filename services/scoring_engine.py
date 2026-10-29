
from services.credibility_analyzer import INDICATOR_WEIGHTS


MIN_EVIDENCE_COVERAGE = 60


def calculate_trust_score(
    credibility_result,
    cross_platform_result=None
):
    indicators = (
        credibility_result.get("indicators")
        if isinstance(credibility_result, dict)
        else None
    )

    if not isinstance(indicators, dict):
        raise ValueError(
            "Credibility indicators are missing"
        )

    positive = 0
    negative = 0
    unknown = 0
    breakdown = {}

    for name, weight in INDICATOR_WEIGHTS.items():
        item = indicators.get(name)

        if (
            not isinstance(item, dict)
            or item.get("status") not in {
                "found",
                "unknown",
                "not_found"
            }
        ):
            raise ValueError(
                f"Invalid indicator: {name}"
            )

        status = item["status"]

        if status == "found":
            positive += weight

        elif status == "not_found":
            negative += weight

        else:
            unknown += weight

        breakdown[name] = {
            "weight": weight,
            **item,
            "awarded_points": (
                weight
                if status == "found"
                else 0
            )
        }

    total = sum(
        INDICATOR_WEIGHTS.values()
    )

    coverage = round(
        100 * (positive + negative) / total,
        2
    )

    evidence_score = round(
        100 * positive / total,
        2
    )

    sufficient = (
        coverage >= MIN_EVIDENCE_COVERAGE
    )

    provisional = (
        evidence_score
        if sufficient
        else None
    )

    if not sufficient:
        status = "insufficient_evidence"
        level = "not_rated"

    elif provisional >= 80:
        status = "provisional"
        level = "high_observed_transparency"

    elif provisional >= 50:
        status = "provisional"
        level = "moderate_observed_transparency"

    else:
        status = "provisional"
        level = "low_observed_transparency"

    en = (
        f"Supported evidence: {evidence_score}/100. "
        f"Assessed indicator coverage: {coverage}%. "
    )

    si = (
        f"තහවුරු වූ සාක්ෂි ලකුණු: {evidence_score}/100. "
        f"පරීක්ෂා කළ දර්ශක ආවරණය: {coverage}%. "
    )

    if not sufficient:
        en += (
            "Not enough evidence for a provisional publisher "
            "transparency rating. Unknown indicators are not "
            "negative evidence."
        )

        si += (
            "තාවකාලික විනිවිදභාවය මට්ටමක් තීරණය කිරීමට "
            "ප්‍රමාණවත් සාක්ෂි නොමැත. නොදන්නා දර්ශක "
            "ඍණාත්මක නොවේ."
        )

    else:
        en += (
            f"Provisional evidence-supported transparency "
            f"score: {provisional}/100. "
            "Unknown indicators remain unassessed. "
            "This does not measure the probability "
            "that news is true."
        )

        si += (
            f"සාක්ෂි මත පදනම් වූ තාවකාලික විනිවිදභාවය: "
            f"{provisional}/100. "
            "නොදන්නා දර්ශක තහවුරු වී නොමැත. "
            "මෙය පුවත් සත්‍ය වීමේ සම්භාවිතාවක් නොවේ."
        )

    return {
        "score_type": "research_prototype_transparency_v2",
        "evidence_score": evidence_score,
        "credibility_score": provisional,
        "evidence_coverage_percent": coverage,
        "minimum_required_coverage": MIN_EVIDENCE_COVERAGE,
        "score_status": status,
        "trust_level": level,
        "verified_truth_probability": None,
        "supported_weight": positive,
        "negative_weight": negative,
        "unknown_weight": unknown,
        "possible_evidence_score_range": {
            "min": evidence_score,
            "max": round(
                100 * (positive + unknown) / total,
                2
            )
        },
        "indicator_breakdown": breakdown,
        "identity_verification": (
            cross_platform_result or {}
        ).get(
            "identity_verification",
            "not_checked"
        ),
        "explanation_en": en,
        "explanation_si": si,
        "analysis_status": "scoring_completed"
    }
