# ==============================================================================
#  core/red_flags.py — Care-AI Deterministic Emergency Pre-check
#
#  Multilingual, negation-aware emergency red flag detection.
#  Runs locally and deterministically with zero API calls.
#
#  NOTE ON CLINICAL / LINGUISTIC VALIDATION:
#  The seed rules and phrases defined below are authored as initial heuristics
#  and are NOT native-speaker or clinician reviewed.
#  They require formal clinical triage protocol validation and review by native
#  Hindi and Telugu speakers before production clinical reliance.
#
#  TODO: Phrases and scripts needing native-speaker / clinician review:
#    - Telugu Latin-script stroke phrases (e.g., "matta jaragadam ledu", "maata muddaga ravadam")
#    - Native Telugu stroke terms (e.g., పక్షవాతం / pakshavatham, మూతి వంకర)
#    - Native Hindi stroke terms (e.g., फेशियल ड्रूपिंग, एक तरफ कमजोरी, लकवा)
#    - Hindi native and Latin-script terms for bleeding, poisoning, anaphylaxis
#    - Telugu native and Latin-script terms for bleeding, poisoning, anaphylaxis
#    - Regional dialect variations of "saans nahi aa rahi" (e.g., dam ghutna)
#    - Regional Telugu variations for breathing difficulty (e.g., aayasamga undi)
#    - Telugu Latin-script transliteration variants (chati noppi vs chaati noppi)
# ==============================================================================

import re
import unicodedata

# Seed rules: id -> {category, source, phrases: {en, hi, hi_latn, te, te_latn}}
RULES = {
    "cardiac": {
        "category": "cardiac",
        "source": "seed_unreviewed",
        "phrases": {
            "en": ["chest pain", "chest tightness", "heart attack"],
            "hi": ["सीने में दर्द"],
            "hi_latn": ["seene mein dard"],
            "te": ["ఛాతీ నొప్పి"],
            "te_latn": ["chaati noppi", "gunde noppi"],
        },
    },
    "breathing": {
        "category": "breathing",
        "source": "seed_unreviewed",
        "phrases": {
            "en": ["can't breathe", "cannot breathe", "difficulty breathing", "not breathing"],
            "hi": ["सांस नहीं आ रही", "सांस लेने में तकलीफ"],
            "hi_latn": ["saans nahi aa rahi", "saans lene mein takleef"],
            "te": ["ఊపిరి ఆడటం లేదు", "శ్వాస తీసుకోవడం కష్టం"],
            "te_latn": ["oopiri aadatledu"],
        },
    },
    "stroke": {
        "category": "stroke",
        "source": "seed_unreviewed",
        "phrases": {
            "en": ["stroke", "face drooping", "slurred speech", "weakness on one side"],
            "hi": [],
            "hi_latn": ["chehra tedha"],
            "te": [],
            "te_latn": [],  # TODO: "matta jaragadam ledu" unverified; native review needed
        },
    },
    "consciousness_seizure": {
        "category": "consciousness_seizure",
        "source": "seed_unreviewed",
        "phrases": {
            "en": ["seizure", "unconscious", "passed out"],
            "hi": ["बेहोश", "दौरा"],
            "hi_latn": ["behosh", "daura"],
            "te": ["స్పృహ తప్పి"],
            "te_latn": ["spruha tappindi"],
        },
    },
    "bleeding_poisoning_anaphylaxis": {
        "category": "bleeding_poisoning_anaphylaxis",
        "source": "seed_unreviewed",
        "phrases": {
            "en": ["severe bleeding", "overdose", "poisoning", "throat swelling", "anaphylaxis"],
            "hi": [],
            "hi_latn": [],
            "te": [],
            "te_latn": [],
        },
    },
    "self_harm": {
        "category": "self_harm",
        "source": "seed_unreviewed",
        "phrases": {
            "en": ["suicidal", "want to die", "kill myself", "self-harm"],
            "hi": ["आत्महत्या", "मरना चाहता"],
            "hi_latn": ["marna chahta", "atmahatya"],
            "te": ["ఆత్మహత్య", "చనిపోవాలని"],
            "te_latn": ["atmahatya", "chanipovalani"],
        },
    },
}

# Aliases for dictionary lookups when callers query with slashes
RULES["consciousness/seizure"] = RULES["consciousness_seizure"]
RULES["bleeding/poisoning/anaphylaxis"] = RULES["bleeding_poisoning_anaphylaxis"]

CANONICAL_RULE_IDS = [
    "cardiac",
    "breathing",
    "stroke",
    "consciousness_seizure",
    "bleeding_poisoning_anaphylaxis",
    "self_harm",
]

# Symptom nouns indicating that a negator applies to a distinct symptom
SYMPTOM_NOUNS = {
    # English
    "fever", "cough", "cold", "headache", "nausea", "vomiting", "pain",
    "rash", "chills", "fatigue", "weakness", "dizziness", "ache", "swelling",
    "diarrhea", "sore", "congestion", "infection", "bleed", "bleeding",
    # Hindi (Latin & Devanagari)
    "bukhar", "khaasi", "khasi", "jukaam", "jukham", "dard", "sardard",
    "ulti", "chakkar", "kamzori", "बुखार", "खांसी", "जुकाम", "दर्द",
    "सिरदर्द", "उल्टी", "चक्कर", "कमजोरी",
    # Telugu (Latin & Telugu script)
    "jwaram", "daggu", "jalubu", "talanopi", "talanoppi", "vanti", "vantulu",
    "neerasam", "noppi", "జ్వరం", "దగ్గు", "జలుబు", "తలనొప్పి", "వాంతులు",
    "నీరసం", "నొప్పి",
}

# Negator groups categorized by language and positional rules:
# English negators: suppress ONLY if BEFORE the phrase, within 4 tokens, in the same clause
ENGLISH_PRE_NEGATORS = {"no", "not", "without", "never", "denies", "dont have", "dont"}

# Hindi/Telugu post-negators: suppress ONLY if AFTER the phrase, within 2 tokens, in the same clause,
# with no other symptom noun between phrase and negator
HINDI_TELUGU_POST_NEGATORS = {"nahi", "nahin", "नहीं", "ledu", "లేదు"}

# Hindi/Telugu pre-negators: suppress ONLY if BEFORE the phrase
HINDI_TELUGU_PRE_NEGATORS = {"bina", "बिना", "lekunda", "లేకుండా"}

# Contrastive conjunctions that stop negation propagation within a clause
CONTRASTIVE_CONJUNCTIONS = {
    # English
    "but", "however", "yet", "still", "although", "though", "except", "nevertheless", "while",
    # Hindi
    "lekin", "magar", "par", "kintu", "parantu", "परन्तु", "किन्तु", "लेकिन", "मगर", "पर",
    # Telugu
    "kaani", "kani", "ayina", "ayithe", "కానీ", "కాని", "అయినా",
}


def is_indic(s: str) -> bool:
    """Return True if string contains Devanagari or Telugu characters."""
    return any(('\u0900' <= ch <= '\u097F') or ('\u0C00' <= ch <= '\u0C7F') for ch in s)


def split_clauses(text: str) -> list:
    """
    Split text into clauses on . , ; ! ? and newlines BEFORE stripping punctuation.
    Negation never crosses a clause boundary.
    """
    if not text:
        return []
    raw_clauses = re.split(r'[\.,;!\?\n\r]+', text)
    return [c.strip() for c in raw_clauses if c.strip()]


def normalize_text(text: str) -> str:
    """
    Normalize text:
    1. Unicode NFKC
    2. Lowercase
    3. Strip punctuation: removes internal apostrophes to join contractions (e.g. can't -> cant),
       replaces all other punctuation/symbols with space while preserving
       Unicode letters and combining marks (e.g. Indic matras).
    4. Collapse whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"['’`]", "", text)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith(('P', 'S')):
            out.append(' ')
        else:
            out.append(ch)
    cleaned = "".join(out)
    return re.sub(r"\s+", " ", cleaned).strip()


def check_red_flags(text: str) -> dict:
    """
    Deterministic, multilingual, negation-aware emergency pre-check.
    Never prints or logs user text.

    Args:
        text (str): Raw user message.

    Returns:
        dict: {"emergency": bool, "category": str|None, "rule_ids": list}
    """
    if not text or not isinstance(text, str):
        return {"emergency": False, "category": None, "rule_ids": []}

    # Split text into clauses on . , ; ! ? and newlines BEFORE stripping punctuation;
    # negation never crosses a clause boundary.
    clauses = split_clauses(text)
    if not clauses:
        return {"emergency": False, "category": None, "rule_ids": []}

    valid_matches = []  # list of (rule_id, category, phrase)

    for clause in clauses:
        norm_clause = normalize_text(clause)
        if not norm_clause:
            continue

        token_spans = []
        for m in re.finditer(r'\S+', norm_clause):
            token_spans.append((m.group(), m.start(), m.end()))

        if not token_spans:
            continue

        tokens = [t[0] for t in token_spans]

        for rule_id in CANONICAL_RULE_IDS:
            rule = RULES[rule_id]
            category = rule["category"]
            phrases_by_lang = rule.get("phrases", {})

            for lang, phrase_list in phrases_by_lang.items():
                for raw_phrase in phrase_list:
                    norm_phrase = normalize_text(raw_phrase)
                    if not norm_phrase:
                        continue

                    indic_phrase = is_indic(norm_phrase) or (lang in ("hi", "te"))

                    match_spans = []
                    if indic_phrase:
                        # Indic-script phrases: substring match
                        start = 0
                        while True:
                            pos = norm_clause.find(norm_phrase, start)
                            if pos == -1:
                                break
                            match_spans.append((pos, pos + len(norm_phrase)))
                            start = pos + 1
                    else:
                        # Latin-script phrases: match on word boundaries
                        pat = re.compile(r'\b' + re.escape(norm_phrase) + r'\b')
                        for m in pat.finditer(norm_clause):
                            match_spans.append((m.start(), m.end()))

                    for m_start, m_end in match_spans:
                        matched_indices = [
                            idx for idx, (tok, t_start, t_end) in enumerate(token_spans)
                            if t_start < m_end and t_end > m_start
                        ]
                        if not matched_indices:
                            continue

                        phrase_start_tok = min(matched_indices)
                        phrase_end_tok = max(matched_indices)

                        # Determine if this match is negated within the current clause
                        is_negated = False

                        # 1. English negators & Indic pre-negators (bina, lekunda):
                        # suppress ONLY if BEFORE the phrase, within 4 tokens, in the same clause.
                        before_start = max(0, phrase_start_tok - 4)
                        before_indices = list(range(before_start, phrase_start_tok))

                        for idx in before_indices:
                            tok = tokens[idx]
                            neg_found = False
                            neg_len = 1

                            if tok in ENGLISH_PRE_NEGATORS or tok in HINDI_TELUGU_PRE_NEGATORS:
                                neg_found = True
                            elif idx + 1 < phrase_start_tok and (tok, tokens[idx + 1]) == ("dont", "have"):
                                neg_found = True
                                neg_len = 2

                            if neg_found:
                                between_tokens = tokens[idx + neg_len : phrase_start_tok]

                                # If separated by contrastive conjunction, negation does not apply
                                has_contrast = any(t in CONTRASTIVE_CONJUNCTIONS for t in between_tokens)
                                if has_contrast:
                                    continue

                                # If another symptom noun appears between negator and phrase (e.g. "no fever but..."),
                                # the negator belongs to that other symptom.
                                has_other_symptom = any(t in SYMPTOM_NOUNS for t in between_tokens)
                                if has_other_symptom:
                                    continue

                                is_negated = True
                                break

                        # 2. Hindi/Telugu negators (nahi, nahin, नहीं, ledu, లేదు):
                        # suppress ONLY if AFTER the phrase, within 2 tokens, in the same clause,
                        # with NO other symptom noun between phrase and negator.
                        if not is_negated:
                            after_end = min(len(tokens), phrase_end_tok + 3)
                            after_indices = list(range(phrase_end_tok + 1, after_end))

                            for idx in after_indices:
                                tok = tokens[idx]
                                if tok in HINDI_TELUGU_POST_NEGATORS:
                                    between_tokens = tokens[phrase_end_tok + 1 : idx]

                                    # Within 2 tokens distance: len(between_tokens) <= 2
                                    if len(between_tokens) > 2:
                                        continue

                                    # Conjunction between phrase and negator
                                    has_contrast = any(t in CONTRASTIVE_CONJUNCTIONS for t in between_tokens)
                                    if has_contrast:
                                        continue

                                    # Another symptom noun between phrase and negator
                                    has_other_symptom = any(t in SYMPTOM_NOUNS for t in between_tokens)
                                    if has_other_symptom:
                                        continue

                                    is_negated = True
                                    break

                        if not is_negated:
                            valid_matches.append((rule_id, category, raw_phrase))

    if not valid_matches:
        return {"emergency": False, "category": None, "rule_ids": []}

    seen = set()
    rule_ids = []
    for r_id, cat, phr in valid_matches:
        if r_id not in seen:
            seen.add(r_id)
            rule_ids.append(r_id)

    first_category = valid_matches[0][1]
    return {
        "emergency": True,
        "category": first_category,
        "rule_ids": rule_ids,
    }
