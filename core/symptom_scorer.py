# ================================================================
#  core/symptom_scorer.py
#  Scores user-reported symptoms using Symptom-severity.csv
#  Unique advantage of your dataset — real severity weights.
# ================================================================

import re
from core.red_flags import (
    ENGLISH_PRE_NEGATORS,
    HINDI_TELUGU_PRE_NEGATORS,
    HINDI_TELUGU_POST_NEGATORS,
    CONTRASTIVE_CONJUNCTIONS,
)

AFFIRMATIVE_TRIGGERS = {
    "have", "has", "had", "having", "with", "present", "presents",
    "experiencing", "developed", "noticed", "feeling", "got", "suffer", "suffering",
}
NEGATION_TOKENS = set(ENGLISH_PRE_NEGATORS) | set(HINDI_TELUGU_PRE_NEGATORS) | {
    "cant", "cannot", "none", "zero", "denied", "free",
}

# Pre-negation scan stoppers (conjunctions/transitions where negation stops propagating)
# "while" and "though" are in CONTRASTIVE_CONJUNCTIONS; "and", "plus", "also", "along" are added here.
NEGATION_SCAN_STOPPERS = {
    "and", "plus", "also", "along", "while", "though",
}


SEVERITY_THRESHOLDS = {
    "LOW":       (0,  6),
    "MEDIUM":    (7,  12),
    "HIGH":      (13, 20),
    "EMERGENCY": (21, 999),
}

# Symptoms that immediately trigger HIGH regardless of score
HIGH_RISK_SYMPTOMS = {
    "chest pain", "difficulty breathing", "breathlessness",
    "loss of consciousness", "altered sensorium", "swollen extremeties",
    "sudden high fever", "stiff neck", "blurred vision",
    "weakness of one body side", "slurred speech",
}


class SymptomScorer:
    """
    Scores a list of user symptoms using severity weights from your dataset.
    Returns severity level, individual scores, and high-risk flags.
    """

    def __init__(self, severity_map: dict):
        """
        severity_map: dict from get_symptom_severity_map()
        e.g. {"chest pain": 5, "itching": 1, ...}
        """
        self.severity_map = severity_map

    def _normalize(self, symptom: str) -> str:
        """Normalize symptom string for lookup."""
        return re.sub(r'\s+', ' ', symptom.lower().strip().replace('_', ' '))

    def score(self, symptoms: list) -> dict:
        """
        Score a list of symptoms.

        Returns:
            {
                "symptom_scores": {"chest pain": 5, "fever": 3},
                "total_score": 8,
                "max_score": 5,
                "severity_level": "MEDIUM",
                "high_risk_symptoms": ["chest pain"],
                "unrecognized_symptoms": ["xyz"],
                "recommendation": "..."
            }
        """
        scores = {}
        unrecognized = []

        for sym in symptoms:
            key = self._normalize(sym)
            if key in self.severity_map:
                scores[sym] = self.severity_map[key]
            else:
                # Fuzzy match — check if any known symptom contains this word
                matched = False
                for known_sym, weight in self.severity_map.items():
                    if key in known_sym or known_sym in key:
                        scores[sym] = weight
                        matched = True
                        break
                if not matched:
                    unrecognized.append(sym)

        total  = sum(scores.values())
        max_sc = max(scores.values()) if scores else 0

        # Check for high-risk symptoms
        high_risk = [
            s for s in symptoms
            if self._normalize(s) in HIGH_RISK_SYMPTOMS or scores.get(s, 0) >= 5
        ]

        # Determine severity
        if high_risk or max_sc >= 5:
            level = "HIGH"
        elif total >= 13:
            level = "HIGH"
        elif total >= 7:
            level = "MEDIUM"
        else:
            level = "LOW"

        # Human-readable recommendation
        if level == "HIGH":
            rec = "⚠️ Some of your symptoms are serious. Please see a doctor soon or go to an emergency room if symptoms are severe."
        elif level == "MEDIUM":
            rec = "Your symptoms are moderate. Monitor closely and consult a doctor if they worsen or persist."
        else:
            rec = "Your symptoms appear mild. Rest, stay hydrated, and consult a doctor if they don't improve in a few days."

        return {
            "symptom_scores":       scores,
            "total_score":          total,
            "max_score":            max_sc,
            "severity_level":       level,
            "high_risk_symptoms":   high_risk,
            "unrecognized_symptoms": unrecognized,
            "recommendation":       rec,
        }

    def _is_match_negated(self, text: str, start: int, end: int) -> bool:
        """
        Check if a matched symptom span is negated in its clause/sentence context.
        Looks backward up to 6 tokens for pre-negators (stopped by contrastive or affirmative tokens).
        Looks forward up to 3 tokens for post-negators (Hindi/Telugu).
        """
        pre_text = text[:start]
        last_sep = max(pre_text.rfind(c) for c in ('.', ';', '!', '?', '\n'))
        pre_clause = pre_text[last_sep + 1:] if last_sep != -1 else pre_text
        pre_tokens = re.findall(r"[a-zA-Z0-9'’]+", pre_clause.lower())

        negated = False
        for i, tok in enumerate(reversed(pre_tokens)):
            if i >= 6:
                break
            if (
                tok in AFFIRMATIVE_TRIGGERS
                or tok in CONTRASTIVE_CONJUNCTIONS
                or tok in NEGATION_SCAN_STOPPERS
            ):
                break
            if tok in NEGATION_TOKENS:
                negated = True
                break

        if negated:
            return True

        post_text = text[end:]
        first_sep_candidates = [post_text.find(c) for c in ('.', ';', '!', '?', '\n') if post_text.find(c) != -1]
        first_sep = min(first_sep_candidates) if first_sep_candidates else len(post_text)
        post_clause = post_text[:first_sep]
        post_tokens = re.findall(r"[a-zA-Z0-9\u0900-\u097F\u0C00-\u0C7F]+", post_clause.lower())

        for i, tok in enumerate(post_tokens[:3]):
            if tok in CONTRASTIVE_CONJUNCTIONS:
                break
            if tok in HINDI_TELUGU_POST_NEGATORS:
                return True

        return False

    def extract_symptoms_from_text(self, text: str) -> list:
        """
        Extract known symptoms mentioned in free text, filtering out negated symptoms.
        Useful for parsing user messages directly.
        """
        text_normalized = self._normalize(text)
        found = []
        for known_sym in self.severity_map:
            norm_sym = self._normalize(known_sym)
            if not norm_sym:
                continue
            pattern = re.compile(r'\b' + re.escape(norm_sym) + r'\b', re.IGNORECASE)
            matches = list(pattern.finditer(text_normalized))
            if not matches:
                continue
            if any(not self._is_match_negated(text_normalized, m.start(), m.end()) for m in matches):
                found.append(known_sym)
        return found
