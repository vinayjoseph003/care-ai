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


ENGLISH_PRE_NEGATORS = {
    "no", "not", "without", "never", "denies", "dont", "cant", "cannot", "free",
    "don't", "can't", "dont have", "don't have", "denied",
}
HINDI_TELUGU_PRE_NEGATORS = {"bina", "बिना", "lekunda", "లేకుండా"}
HINDI_TELUGU_POST_NEGATORS = {"nahi", "nahin", "नहीं", "ledu", "లేదు"}
CONTRASTIVE_CONJUNCTIONS = {
    "but", "however", "yet", "still", "although", "though", "except",
    "lekin", "magar", "par", "परन्तु", "किन्तु", "लेकिन", "मगर", "కానీ", "కాని",
}

# Explicit allowlist of approved clinical intake fields from SOCRATES
APPROVED_CLINICAL_FIELDS = {
    "symptom_or_topic",
    "site",
    "onset",
    "character",
    "radiation",
    "associations",
    "time_course",
    "exacerbating",
    "relieving",
    "severity",
    "trigger_activity",
    "other_details",
}

TRIVIAL_INTAKE_VALUES = {
    "", "...", "null", "none", "unknown", "n/a", "na", "not sure",
    "idk", "i don't know", "i dont know", "no idea", "dunno", "nothing",
    "nope", "nil", "undefined",
    "yes", "yeah", "yep", "no", "ok", "okay", "fine", "good", "better",
    "worse", "same", "maybe", "probably",
}

# Curated clinical symptom vocabulary (English, Hindi Latin/Devanagari, Telugu Latin/Script)
# Broad conversational tokens (e.g. "back", "feeling", "sleep", "voice") are excluded as standalone seeds.
COMMON_SYMPTOM_SEEDS = {
    # English single-word clinical symptoms
    "pain", "ache", "aching", "hurt", "hurting", "sore", "soreness",
    "fever", "cough", "coughing", "migraine", "headache", "nausea",
    "vomiting", "diarrhea", "diarrhoea", "constipation", "cramp", "cramping",
    "rash", "rashes", "itch", "itching", "swelling", "swollen",
    "dizzy", "dizziness", "vertigo", "sprain", "strain", "stiffness",
    "numbness", "breathless", "breathlessness", "fatigue",
    # English multi-word phrases to avoid ambiguity of standalone words
    "sore throat", "throat pain", "stomach ache", "stomach pain",
    "abdominal pain", "chest pain", "back pain", "neck pain", "joint pain",
    "runny nose", "blocked nose", "loss of taste", "loss of smell",
    "shortness of breath", "difficulty breathing", "common cold", "have a cold",
    "had a cold", "got a cold", "feeling sick", "feeling unwell",
    "can't breathe", "cannot breathe",
    # Hindi Latin
    "bukhar", "dard", "khasi", "khaasi", "zukam", "jukaam", "jukham",
    "sardard", "ulti", "chakkar", "kamzori",
    # Hindi Devanagari
    "दर्द", "बुखार", "खांसी", "जुकाम", "सिरदर्द", "उल्टी", "चक्कर",
    "कमजोरी", "बदन दर्द", "गले में खराश",
    # Telugu Latin
    "noppi", "jwaram", "daggu", "jalubu", "talanopi", "talanoppi",
    "vanti", "vantulu", "neerasam",
    # Telugu Script
    "నొప్పి", "జ్వరం", "దగ్గు", "జలుబు", "తలనొప్పి", "వాంతులు", "నీరసం",
}


def is_indic(s: str) -> bool:
    """Return True if string contains Devanagari or Telugu characters."""
    return any(('ऀ' <= ch <= 'ॿ') or ('ఀ' <= ch <= '౿') for ch in s)


def _tokenize_words(s: str) -> list[str]:
    puncts = ".,;!?\"'()[]{}"
    return [t.strip(puncts) for t in s.lower().split() if t.strip(puncts)]


def _is_span_negated(text: str, start: int, end: int) -> bool:
    """
    Check if a matched symptom span is negated in its clause context.
    Conservative: if negated or pre-empted by negation in the same clause, returns True.
    """
    pre_text = text[:start]
    last_sep = max([pre_text.rfind(c) for c in ('.', ';', '!', '?', chr(10))] + [-1])
    pre_clause = pre_text[last_sep + 1:]
    pre_tokens = _tokenize_words(pre_clause)

    for i, tok in enumerate(reversed(pre_tokens)):
        if i >= 5:
            break
        if tok in CONTRASTIVE_CONJUNCTIONS:
            break
        if tok in ENGLISH_PRE_NEGATORS or tok in HINDI_TELUGU_PRE_NEGATORS:
            return True

    post_text = text[end:]
    first_sep_cands = [post_text.find(c) for c in ('.', ';', '!', '?', chr(10)) if post_text.find(c) != -1]
    first_sep = min(first_sep_cands) if first_sep_cands else len(post_text)
    post_clause = post_text[:first_sep]
    post_tokens = _tokenize_words(post_clause)

    for i, tok in enumerate(post_tokens[:3]):
        if tok in CONTRASTIVE_CONJUNCTIONS:
            break
        if tok in HINDI_TELUGU_POST_NEGATORS:
            return True

    return False


def has_affirmative_symptom(text: str, seeds: set | None = None) -> bool:
    """Check if text contains an affirmative, non-negated symptom term."""
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    active_seeds = seeds or COMMON_SYMPTOM_SEEDS
    for term in active_seeds:
        if is_indic(term):
            start = 0
            while True:
                pos = text_lower.find(term, start)
                if pos == -1:
                    break
                if not _is_span_negated(text, pos, pos + len(term)):
                    return True
                start = pos + 1
        else:
            pat = rf'(?:^|[\s\.,;:!?\(\)\[\]])({re.escape(term)})(?:$|[\s\.,;:!?\(\)\[\]])'
            for m in re.finditer(pat, text_lower):
                sp_start = m.start(1)
                sp_end = m.end(1)
                if not _is_span_negated(text, sp_start, sp_end):
                    return True
    return False


def has_adequate_clinical_info(
    collected_info: dict | None = None,
    user_text: str | None = None,
    symptom_seeds: set | None = None,
) -> bool:
    """
    Determine whether clinical intake contains substantive symptom information.

    Requires:
      1. Meaningful primary symptom/topic in collected_info["symptom_or_topic"],
         OR substantive affirmative symptom description across approved clinical fields.
         Generic values (e.g. 'yes' in other_details, 'ok' in character) do NOT qualify.
      2. OR user_text contains an affirmative, non-negated clinical symptom term.

    Returns:
        True if valid clinical information is found, False otherwise.
    """
    if collected_info and isinstance(collected_info, dict):
        primary = collected_info.get("symptom_or_topic")
        if primary and not isinstance(primary, bool):
            cleaned = str(primary).strip().lower()
            if cleaned and cleaned not in TRIVIAL_INTAKE_VALUES and len(cleaned) >= 2:
                if not cleaned.isdigit():
                    return True

        for k, v in collected_info.items():
            if k not in APPROVED_CLINICAL_FIELDS or v is None or isinstance(v, bool):
                continue
            cleaned = str(v).strip().lower()
            if not cleaned or cleaned in TRIVIAL_INTAKE_VALUES:
                continue
            if has_affirmative_symptom(str(v), symptom_seeds):
                return True

    if user_text and isinstance(user_text, str):
        if has_affirmative_symptom(user_text, symptom_seeds):
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
    has_sufficient_info: bool = True,
    no_new_info_count: int = 0,
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

    if not has_sufficient_info:
        return ("SEE_DOCTOR", "INSUFFICIENT_INFORMATION")

    return ("SELF_CARE", None)
