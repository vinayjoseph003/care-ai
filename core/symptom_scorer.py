# ================================================================
#  core/symptom_scorer.py
#  Scores user-reported symptoms using Symptom-severity.csv
#  Unique advantage of your dataset — real severity weights.
# ================================================================

import re


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

    def extract_symptoms_from_text(self, text: str) -> list:
        """
        Try to extract known symptoms mentioned in free text.
        Useful for parsing user messages directly.
        """
        text_normalized = self._normalize(text)
        found = []
        for known_sym in self.severity_map:
            if known_sym in text_normalized:
                found.append(known_sym)
        return found