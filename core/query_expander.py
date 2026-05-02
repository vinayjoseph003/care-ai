# ================================================================
#  core/query_expander.py — Synonym-based query expansion
#  Expands user queries before TF-IDF vectorisation.
#  Covers Indian-English colloquialisms + medical synonyms.
# ================================================================

SYNONYMS: dict[str, list[str]] = {
    # Pain variants
    "pain":             ["ache", "aching", "discomfort", "soreness", "hurt"],
    "stomach pain":     ["abdominal pain", "belly ache", "tummy pain", "abdomen pain",
                         "gut pain", "gastric pain"],
    "chest pain":       ["chest tightness", "chest pressure", "chest discomfort",
                         "precordial pain", "angina"],
    "back pain":        ["backache", "lumbar pain", "spine pain", "dorsal pain"],
    "headache":         ["head pain", "cephalalgia", "migraine", "cranial pain"],
    "joint pain":       ["arthralgia", "joint ache", "bone pain"],
    "muscle pain":      ["myalgia", "muscle ache", "muscle soreness", "DOMS"],

    # Respiratory
    "breathless":       ["dyspnea", "shortness of breath", "difficulty breathing",
                         "breathlessness", "breathe hard"],
    "cough":            ["coughing", "dry cough", "wet cough", "persistent cough"],
    "cold":             ["common cold", "rhinitis", "runny nose", "nasal congestion",
                         "sneezing", "blocked nose"],

    # GI
    "loose motion":     ["diarrhea", "diarrhoea", "watery stool", "loose stool",
                         "frequent stools", "stomach upset"],
    "vomiting":         ["nausea and vomiting", "throwing up", "emesis", "nausea"],
    "acidity":          ["acid reflux", "heartburn", "GERD", "indigestion", "gastritis"],
    "constipation":     ["hard stool", "difficulty passing stool", "no bowel movement"],
    "gas":              ["bloating", "flatulence", "abdominal gas", "belching"],

    # Fever
    "fever":            ["pyrexia", "high temperature", "febrile", "high grade fever",
                         "temperature", "hyperthermia"],
    "chills":           ["rigors", "shivering", "chills and fever"],

    # Urinary
    "burning urine":    ["dysuria", "burning urination", "painful urination", "UTI symptoms"],
    "frequent urination": ["polyuria", "frequent micturition", "urinary frequency"],

    # Neuro
    "fits":             ["seizure", "convulsion", "epilepsy", "epileptic episode"],
    "dizzy":            ["dizziness", "vertigo", "lightheadedness", "giddiness"],
    "numbness":         ["tingling", "paresthesia", "pins and needles", "numb"],
    "fainting":         ["syncope", "loss of consciousness", "blackout", "passing out"],

    # Skin
    "rash":             ["skin rash", "eruption", "hives", "urticaria", "redness"],
    "itching":          ["pruritus", "itch", "skin itching"],
    "swelling":         ["edema", "oedema", "puffiness", "swollen"],

    # General
    "tiredness":        ["fatigue", "weakness", "lethargy", "exhaustion", "asthenia"],
    "weight loss":      ["unintentional weight loss", "losing weight", "weight reduction"],
    "yellow eyes":      ["jaundice", "icterus", "yellowing of skin", "yellowish urine"],
    "eye pain":         ["ocular pain", "eye ache", "eye discomfort"],

    # Indian-English colloquialisms
    "motion":           ["diarrhea", "loose stool", "bowel movement"],
    "sugar":            ["diabetes", "diabetes mellitus", "high blood sugar", "hyperglycemia"],
    "bp":               ["blood pressure", "hypertension", "high BP"],
    "urine infection":  ["UTI", "urinary tract infection", "cystitis"],
    "periods":          ["menstruation", "menstrual cycle", "menstrual pain", "dysmenorrhea"],
    "periods pain":     ["dysmenorrhea", "menstrual cramps", "period cramps"],
    "white discharge":  ["leucorrhoea", "vaginal discharge"],
    "piles":            ["hemorrhoids", "haemorrhoids", "rectal bleeding"],
    "stones":           ["kidney stones", "renal calculi", "gallstones", "urolithiasis"],
}

# Reverse map: longer phrases should match before shorter ones
# Sort by key length descending so "stomach pain" matches before "pain"
_SORTED_KEYS = sorted(SYNONYMS.keys(), key=len, reverse=True)


def expand_query(query: str) -> str:
    """
    Expand query with synonyms before TF-IDF vectorisation.
    Appends synonym terms without modifying the original query.

    Example:
        expand_query("I have loose motion and fever")
        → "I have loose motion and fever diarrhea diarrhoea watery stool ... pyrexia ..."
    """
    q_lower = query.lower()
    extras: list[str] = []
    matched_positions: set[int] = set()

    for key in _SORTED_KEYS:
        idx = q_lower.find(key)
        if idx == -1:
            continue
        # Avoid double-expanding overlapping matches
        positions = set(range(idx, idx + len(key)))
        if positions & matched_positions:
            continue
        matched_positions |= positions
        extras.extend(SYNONYMS[key])

    if not extras:
        return query

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_extras: list[str] = []
    for term in extras:
        if term.lower() not in q_lower and term not in seen:
            seen.add(term)
            unique_extras.append(term)

    return query + " " + " ".join(unique_extras)