"""
tests/test_severity_floor.py — Safety triage floor tests for user-reported severe pain.

Tests:
1. is_severe_pain helper:
   - TRUE: "8/10", "7/10", "9 out of 10", "10", "8", "severe", "unbearable", "worst pain"
   - FALSE: "6/10", "3", "mild", "", None, "moderate", "11/10", "100", ambiguous sentences
   - Bare number rule: only single integers 1-10 (>=7 -> True, <7 -> False, outside 1-10 -> False)
2. apply_triage_floor with user_severity:
   - SELF_CARE + severe pain -> ("SEE_DOCTOR", "SEVERE_PAIN_SCORE")
   - Non-SELF_CARE triages unchanged
   - Existing floor reasons keep priority order over severe pain
3. Pipeline integration with CareAI:
   - Wire collected_info.get("severity") into apply_triage_floor
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

for _mod in ("scipy", "scipy.sparse"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

from core.triage_floor import apply_triage_floor, is_severe_pain
from bot.careai import CareAI


class TestIsSeverePain(unittest.TestCase):

    def test_true_cases(self):
        true_cases = [
            "8/10",
            "7/10",
            "9 out of 10",
            "10",
            "8",
            "severe",
            "unbearable",
            "worst pain",
        ]
        for tc in true_cases:
            with self.subTest(case=tc):
                self.assertTrue(is_severe_pain(tc), f"Expected True for '{tc}'")

    def test_false_cases(self):
        false_cases = [
            "6/10",
            "3",
            "mild",
            "",
            None,
            "moderate",
            "11/10",
            "100",
            "I walked 10 miles and had pain for 8 days after lifting",
        ]
        for fc in false_cases:
            with self.subTest(case=fc):
                self.assertFalse(is_severe_pain(fc), f"Expected False for '{fc}'")

    def test_bare_number_rule(self):
        # Single integer in 7..10 -> True
        self.assertTrue(is_severe_pain("7"))
        self.assertTrue(is_severe_pain("8"))
        self.assertTrue(is_severe_pain("9"))
        self.assertTrue(is_severe_pain("10"))

        # Single integer in 1..6 -> False
        self.assertFalse(is_severe_pain("1"))
        self.assertFalse(is_severe_pain("3"))
        self.assertFalse(is_severe_pain("6"))

        # Single integer out of 1..10 range -> False
        self.assertFalse(is_severe_pain("0"))
        self.assertFalse(is_severe_pain("11"))
        self.assertFalse(is_severe_pain("100"))
        self.assertFalse(is_severe_pain("-5"))

        # Numbers embedded in sentences without /10 -> False (bare-number rule only applies to whole string)
        self.assertFalse(is_severe_pain("pain level 8"))
        self.assertFalse(is_severe_pain("started 8 days ago"))
        self.assertFalse(is_severe_pain("carried 10 boxes"))


class TestApplyTriageFloorSeverity(unittest.TestCase):

    def setUp(self):
        self.clean_kwargs = {
            "triage": "SELF_CARE",
            "differential": [{"condition": "Tension Headache"}],
            "top_condition": "Tension Headache",
            "confidence": 80,
            "severity_level": "LOW",
            "conditions_agree": True,
        }

    def test_self_care_with_severe_pain_escalates(self):
        triage, reason = apply_triage_floor(
            **self.clean_kwargs,
            user_severity="8/10",
        )
        self.assertEqual(triage, "SEE_DOCTOR")
        self.assertEqual(reason, "SEVERE_PAIN_SCORE")

        triage, reason = apply_triage_floor(
            **self.clean_kwargs,
            user_severity="severe",
        )
        self.assertEqual(triage, "SEE_DOCTOR")
        self.assertEqual(reason, "SEVERE_PAIN_SCORE")

    def test_self_care_with_non_severe_pain_stays_self_care(self):
        for sev in ("6/10", "moderate", "mild", "3", None, ""):
            with self.subTest(user_severity=sev):
                triage, reason = apply_triage_floor(
                    **self.clean_kwargs,
                    user_severity=sev,
                )
                self.assertEqual(triage, "SELF_CARE")
                self.assertIsNone(reason)

    def test_non_self_care_triages_remain_unchanged(self):
        for level in ("SEE_DOCTOR", "URGENT_CARE", "EMERGENCY"):
            with self.subTest(level=level):
                kwargs = dict(self.clean_kwargs)
                kwargs["triage"] = level
                triage, reason = apply_triage_floor(
                    **kwargs,
                    user_severity="10/10",
                )
                self.assertEqual(triage, level)
                self.assertIsNone(reason)

    def test_existing_floor_reasons_keep_priority_over_user_severity(self):
        # 1. EMPTY_DIFFERENTIAL
        kwargs = dict(self.clean_kwargs, differential=[])
        t, r = apply_triage_floor(**kwargs, user_severity="8/10")
        self.assertEqual(t, "SEE_DOCTOR")
        self.assertEqual(r, "EMPTY_DIFFERENTIAL")

        # 2. EMPTY_TOP_CONDITION
        kwargs = dict(self.clean_kwargs, top_condition="")
        t, r = apply_triage_floor(**kwargs, user_severity="8/10")
        self.assertEqual(t, "SEE_DOCTOR")
        self.assertEqual(r, "EMPTY_TOP_CONDITION")

        # 3. HIGH_SEVERITY
        kwargs = dict(self.clean_kwargs, severity_level="HIGH")
        t, r = apply_triage_floor(**kwargs, user_severity="8/10")
        self.assertEqual(t, "SEE_DOCTOR")
        self.assertEqual(r, "HIGH_SEVERITY")

        # 4. MISSING_CONFIDENCE
        kwargs = dict(self.clean_kwargs, confidence=None)
        t, r = apply_triage_floor(**kwargs, user_severity="8/10")
        self.assertEqual(t, "SEE_DOCTOR")
        self.assertEqual(r, "MISSING_CONFIDENCE")

        # 5. LOW_CONFIDENCE
        kwargs = dict(self.clean_kwargs, confidence=40)
        t, r = apply_triage_floor(**kwargs, user_severity="8/10")
        self.assertEqual(t, "SEE_DOCTOR")
        self.assertEqual(r, "LOW_CONFIDENCE")

        # 6. CONDITION_DISAGREEMENT
        kwargs = dict(self.clean_kwargs, conditions_agree=False)
        t, r = apply_triage_floor(**kwargs, user_severity="8/10")
        self.assertEqual(t, "SEE_DOCTOR")
        self.assertEqual(r, "CONDITION_DISAGREEMENT")


class TestCareAISeverityFloorWiring(unittest.TestCase):

    def test_pipeline_wires_collected_severity_to_floor(self):
        mock_router = MagicMock()
        mock_vs = MagicMock()
        mock_vs.search.return_value = []
        mock_vs.search_by_symptoms.return_value = []
        mock_scorer = MagicMock()
        mock_scorer.extract_symptoms_from_text.return_value = []
        mock_scorer.score.return_value = {}

        bot = CareAI(
            vector_store=mock_vs,
            symptom_scorer=mock_scorer,
            router=mock_router,
        )
        bot._detect_language = MagicMock(return_value={"language": "English", "code": "en"})
        bot._check_followup = MagicMock(return_value={
            "has_enough_info": True,
            "diagnostic_confidence": 75,
            "collected_info": {"severity": "8/10"},
            "followup_question": None,
        })
        bot._classify_query = MagicMock(return_value="SYMPTOM")
        bot._retrieve = MagicMock(return_value=[])
        bot._reason_consistent = MagicMock(return_value={
            "triage_level": "SELF_CARE",
            "differential_diagnosis": [
                {"rank": 1, "condition": "Tension Headache", "probability": "HIGH"}
            ],
            "specialist": "General Practitioner",
            "confidence_score": 85,
            "conditions_agree": True,
        })
        bot._get_selfcare_guidance = MagicMock()
        bot._get_treatment_guidance = MagicMock(return_value={
            "home_care": ["Rest"],
            "escalation_timeline": "See doctor if worse",
        })
        bot._get_red_flags = MagicMock(return_value={"red_flags": []})
        bot._answer_streamed = MagicMock(return_value="Advice")
        bot._factcheck = MagicMock(return_value={
            "claim_checks": [],
            "hallucination_risk": "LOW",
            "verdict": "VERIFIED",
        })
        bot._xai_explain_streamed = MagicMock(return_value="Explanation")

        result = bot.chat("I have a severe headache 8/10")

        self.assertEqual(bot.last_triage, "SEE_DOCTOR")
        self.assertEqual(result.get("triage"), "SEE_DOCTOR")
        bot._get_selfcare_guidance.assert_not_called()
        self.assertEqual(bot.xai_logs[-1].get("triage_floor_reason"), "SEVERE_PAIN_SCORE")


if __name__ == "__main__":
    unittest.main()

