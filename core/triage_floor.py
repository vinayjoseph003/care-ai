# ==============================================================================
#  core/triage_floor.py — Safety Triage Floor
#
#  Pure function ensuring that SELF_CARE is never assigned when evidence is
#  absent, uncertain, or conflicting.
# ==============================================================================

HIGH_SEVERITY_LEVELS = {"HIGH", "EMERGENCY"}


import math
import re


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


def is_severe_pain(text: str | None) -> bool:
    """
    Determine if user-reported severity text represents severe pain.

    TRUE:
      - "8/10", "7/10", "9 out of 10", "10", "8"
      - "severe", "unbearable", "worst pain"
    FALSE:
      - "6/10", "3", "mild", "", None, "moderate"
      - "11/10", "100" (out of range -> False, do not guess)
      - Ambiguous sentences with no clear score
      - Bare-number rule: only if the whole string is a single integer 1-10.
    """
    if text is None or isinstance(text, bool):
        return False
    if not isinstance(text, (str, int, float)):
        return False
    cleaned = str(text).strip()
    if not cleaned:
        return False

    # 1. Bare-number rule: only if the whole string is a single integer 1-10
    if re.fullmatch(r"[+-]?\d+", cleaned):
        try:
            val = int(cleaned)
            if 1 <= val <= 10:
                return val >= 7
            return False
        except ValueError:
            return False

    # 2. X/10 or X out of 10 pattern
    match = re.search(r"\b(\d+)\s*(?:/|\bout\s+of\b)\s*10\b", cleaned, re.IGNORECASE)
    if match:
        try:
            val = int(match.group(1))
            if 1 <= val <= 10:
                return val >= 7
            return False
        except ValueError:
            pass

    # 3. Verbal descriptors (severe, unbearable, worst pain)
    lower = cleaned.lower()
    if re.search(r"\b(?:not|no)\s+(?:severe|unbearable|worst\s+pain)\b", lower):
        return False
    if "moderately severe" in lower:
        return False
    if re.search(r"\b(?:severe|unbearable|worst\s+pain)\b", lower):
        return True

    return False


def apply_triage_floor(
    triage: str | None,
    differential: list | None,
    top_condition: str | None,
    confidence: int | float | str | None,
    severity_level: str | None,
    conditions_agree: bool | None = True,
    user_severity: str | None = None,
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
        user_severity: User-reported severity string from SOCRATES extraction, or None.

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

    if is_severe_pain(user_severity):
        return ("SEE_DOCTOR", "SEVERE_PAIN_SCORE")

    return ("SELF_CARE", None)
