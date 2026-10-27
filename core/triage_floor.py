# ==============================================================================
#  core/triage_floor.py — Safety Triage Floor
#
#  Pure function ensuring that SELF_CARE is never assigned when evidence is
#  absent, uncertain, or conflicting.
# ==============================================================================

HIGH_SEVERITY_LEVELS = {"HIGH", "EMERGENCY"}


import math


def _coerce_confidence(val: int | float | str | None) -> float | None:
    """
    Safely coerce a confidence value to float.
    Accepts int, float, or numeric strings (stripping leading/trailing whitespace and '%').
    Returns None if val is None, boolean, unparseable, NaN, or infinite.
    """
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        try:
            f = float(val)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except (ValueError, TypeError, OverflowError):
            return None
    if isinstance(val, str):
        cleaned = val.strip().rstrip('%').strip()
        try:
            f = float(cleaned)
            return None if (math.isnan(f) or math.isinf(f)) else f
        except (ValueError, TypeError, OverflowError):
            return None
    return None


def apply_triage_floor(
    triage: str | None,
    differential: list | None,
    top_condition: str | None,
    confidence: int | float | str | None,
    severity_level: str | None,
    conditions_agree: bool | None = True,
) -> tuple[str, str | None]:
    """
    Pure safety floor function for Care-AI triage decisions.

    Ensures that a triage verdict of 'SELF_CARE' is escalated to 'SEE_DOCTOR'
    if evidence is missing, uncertain, or indicates high symptom severity.
    Never downgrades any other triage level.

    Args:
        triage: Proposed triage level ('SELF_CARE', 'SEE_DOCTOR', etc.).
        differential: List of differential diagnosis dicts.
        top_condition: Name of top hypothesized condition.
        confidence: Diagnostic confidence score (0-100), numeric str ('85%'), or None.
        severity_level: Severity string from symptom scorer (e.g. 'LOW', 'HIGH') or None.
        conditions_agree: True if dual reasoning passes agreed or single pass, False if disagreed.

    Returns:
        (final_triage, reason_code_or_None)
    """
    if triage != "SELF_CARE":
        return (triage, None)

    if not differential:
        return ("SEE_DOCTOR", "EMPTY_DIFFERENTIAL")

    if not top_condition or not str(top_condition).strip():
        return ("SEE_DOCTOR", "EMPTY_TOP_CONDITION")

    if severity_level and str(severity_level).strip().upper() in HIGH_SEVERITY_LEVELS:
        return ("SEE_DOCTOR", "HIGH_SEVERITY")

    conf_num = _coerce_confidence(confidence)
    if conf_num is None:
        return ("SEE_DOCTOR", "MISSING_CONFIDENCE")

    if conf_num < 50:
        return ("SEE_DOCTOR", "LOW_CONFIDENCE")

    if conditions_agree is False:
        return ("SEE_DOCTOR", "CONDITION_DISAGREEMENT")

    return ("SELF_CARE", None)
