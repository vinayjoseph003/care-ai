# ==============================================================================
#  tests/test_triage_floor.py — Unit tests for Safety Triage Floor
#
#  Offline, deterministic test suite using mocks only.
#  Covers:
#    - Pure function apply_triage_floor:
#        - Empty differential escalates to SEE_DOCTOR
#        - Empty top condition escalates to SEE_DOCTOR
#        - High severity from scorer escalates to SEE_DOCTOR
#        - Emergency severity from scorer escalates to SEE_DOCTOR
#        - Low confidence (< 50) escalates to SEE_DOCTOR
#        - Disagreeing conditions across passes (conditions_agree=False) escalates
#        - All clean inputs stay SELF_CARE (returns None reason)
#        - Non-SELF_CARE levels (SEE_DOCTOR, URGENT_CARE, EMERGENCY) pass through untouched
#    - Pipeline-level CareAI integration test:
#        - Mocked reasoning returning SELF_CARE with empty differential gets floored
#        - last_triage == "SEE_DOCTOR"
#        - _get_selfcare_guidance is not called
#        - triage_floor_reason recorded in xai_logs
# ==============================================================================

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Ensure offline mock-only tests can run without optional heavy dependencies (e.g. scipy)
for _mod in ("scipy", "scipy.sparse"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

from core.triage_floor import apply_triage_floor
from bot.careai import CareAI


class TestTriageFloorPureFunction(unittest.TestCase):

    def test_empty_differential_escalates(self):
        """Rule A1: Empty differential escalates SELF_CARE to SEE_DOCTOR."""
        triage, reason = apply_triage_floor(
            triage="SELF_CARE",
            differential=[],
            top_condition="Tension Headache",
            confidence=80,
            severity_level="LOW",
            conditions_agree=True,
        )
        self.assertEqual(triage, "SEE_DOCTOR")
        self.assertEqual(reason, "EMPTY_DIFFERENTIAL")

    def test_empty_top_condition_escalates(self):
        """Rule A2: Empty top_condition string escalates SELF_CARE to SEE_DOCTOR."""
        triage, reason = apply_triage_floor(
            triage="SELF_CARE",
            differential=[{"condition": ""}],
            top_condition="",
            confidence=80,
            severity_level="LOW",
            conditions_agree=True,
        )
        self.assertEqual(triage, "SEE_DOCTOR")
        self.assertEqual(reason, "EMPTY_TOP_CONDITION")

    def test_high_severity_escalates(self):
        """Rule B1: SymptomScorer HIGH severity escalates SELF_CARE to SEE_DOCTOR."""
        triage, reason = apply_triage_floor(
            triage="SELF_CARE",
            differential=[{"condition": "Common Cold"}],
            top_condition="Common Cold",
            confidence=80,
            severity_level="HIGH",
            conditions_agree=True,
        )
        self.assertEqual(triage, "SEE_DOCTOR")
        self.assertEqual(reason, "HIGH_SEVERITY")

    def test_emergency_severity_escalates(self):
        """Rule B2: SymptomScorer EMERGENCY severity escalates SELF_CARE to SEE_DOCTOR."""
        triage, reason = apply_triage_floor(
            triage="SELF_CARE",
            differential=[{"condition": "Chest Discomfort"}],
            top_condition="Chest Discomfort",
            confidence=80,
            severity_level="EMERGENCY",
            conditions_agree=True,
        )
        self.assertEqual(triage, "SEE_DOCTOR")
        self.assertEqual(reason, "HIGH_SEVERITY")

    def test_confidence_none_escalates_to_missing_confidence(self):
        """SELF_CARE with confidence=None escalates to SEE_DOCTOR with MISSING_CONFIDENCE."""
        triage, reason = apply_triage_floor(
            triage="SELF_CARE",
            differential=[{"condition": "Common Cold"}],
            top_condition="Common Cold",
            confidence=None,
            severity_level="LOW",
            conditions_agree=True,
        )
        self.assertEqual(triage, "SEE_DOCTOR")
        self.assertEqual(reason, "MISSING_CONFIDENCE")

    def test_confidence_non_numeric_string_escalates_to_missing_confidence(self):
        """SELF_CARE with non-numeric string confidence ('high') escalates with MISSING_CONFIDENCE."""
        triage, reason = apply_triage_floor(
            triage="SELF_CARE",
            differential=[{"condition": "Common Cold"}],
            top_condition="Common Cold",
            confidence="high",
            severity_level="LOW",
            conditions_agree=True,
        )
        self.assertEqual(triage, "SEE_DOCTOR")
        self.assertEqual(reason, "MISSING_CONFIDENCE")

    def test_confidence_percentage_string_stays_self_care(self):
        """Clean case with numeric string confidence ('85%') stays SELF_CARE."""
        triage, reason = apply_triage_floor(
            triage="SELF_CARE",
            differential=[{"condition": "Common Cold"}],
            top_condition="Common Cold",
            confidence="85%",
            severity_level="LOW",
            conditions_agree=True,
        )
        self.assertEqual(triage, "SELF_CARE")
        self.assertIsNone(reason)

    def test_confidence_float_stays_self_care(self):
        """Clean case with float confidence (85.0) stays SELF_CARE."""
        triage, reason = apply_triage_floor(
            triage="SELF_CARE",
            differential=[{"condition": "Common Cold"}],
            top_condition="Common Cold",
            confidence=85.0,
            severity_level="LOW",
            conditions_agree=True,
        )
        self.assertEqual(triage, "SELF_CARE")
        self.assertIsNone(reason)

    def test_low_confidence_escalates(self):
        """Rule C: Confidence score < 50 escalates SELF_CARE to SEE_DOCTOR."""
        triage, reason = apply_triage_floor(
            triage="SELF_CARE",
            differential=[{"condition": "Common Cold"}],
            top_condition="Common Cold",
            confidence=45,
            severity_level="LOW",
            conditions_agree=True,
        )
        self.assertEqual(triage, "SEE_DOCTOR")
        self.assertEqual(reason, "LOW_CONFIDENCE")

    def test_conditions_disagree_escalates(self):
        """Rule D: conditions_agree=False escalates SELF_CARE to SEE_DOCTOR."""
        triage, reason = apply_triage_floor(
            triage="SELF_CARE",
            differential=[{"condition": "Common Cold"}],
            top_condition="Common Cold",
            confidence=80,
            severity_level="LOW",
            conditions_agree=False,
        )
        self.assertEqual(triage, "SEE_DOCTOR")
        self.assertEqual(reason, "CONDITION_DISAGREEMENT")

    def test_all_clean_inputs_stay_self_care(self):
        """All-clean inputs with solid evidence stay SELF_CARE with no reason code."""
        triage, reason = apply_triage_floor(
            triage="SELF_CARE",
            differential=[{"condition": "Common Cold"}],
            top_condition="Common Cold",
            confidence=75,
            severity_level="LOW",
            conditions_agree=True,
        )
        self.assertEqual(triage, "SELF_CARE")
        self.assertIsNone(reason)

    def test_non_self_care_triages_pass_through_unchanged(self):
        """Higher triage levels are never downgraded even if differential is empty or confidence low."""
        for level in ("SEE_DOCTOR", "URGENT_CARE", "EMERGENCY"):
            with self.subTest(level=level):
                triage, reason = apply_triage_floor(
                    triage=level,
                    differential=[],
                    top_condition="",
                    confidence=10,
                    severity_level="HIGH",
                    conditions_agree=False,
                )
                self.assertEqual(triage, level)
                self.assertIsNone(reason)


class TestTriageFloorPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.mock_router = MagicMock()
        self.mock_vs = MagicMock()
        self.mock_vs.search.return_value = []
        self.mock_vs.search_by_symptoms.return_value = []
        self.mock_scorer = MagicMock()
        self.mock_scorer.extract_symptoms_from_text.return_value = []
        self.mock_scorer.score.return_value = {}

        self.mock_router.complete.return_value = '{"language": "English", "code": "en"}'

        self.bot = CareAI(
            vector_store=self.mock_vs,
            symptom_scorer=self.mock_scorer,
            router=self.mock_router,
        )
        self.bot._detect_language = MagicMock(return_value={"language": "English", "code": "en"})

    def test_pipeline_escalates_empty_differential_and_skips_selfcare_guidance(self):
        """
        Pipeline-level test:
        When reasoning returns SELF_CARE and an empty differential:
          - last_triage is floored to 'SEE_DOCTOR'
          - _get_selfcare_guidance is NOT called
          - triage_floor_reason recorded in xai_logs
        """
        # Mock inquiry step to bypass followups
        self.bot._check_followup = MagicMock(return_value={
            "has_enough_info": True,
            "diagnostic_confidence": 75,
            "collected_info": {},
            "followup_question": None,
        })
        self.bot._classify_query = MagicMock(return_value="SYMPTOM")
        self.bot._retrieve = MagicMock(return_value=[])

        # Reasoning returns SELF_CARE with empty differential
        self.bot._reason_consistent = MagicMock(return_value={
            "triage_level": "SELF_CARE",
            "differential_diagnosis": [],
            "specialist": "General Practitioner",
            "confidence_score": 85,
            "conditions_agree": True,
        })

        self.bot._get_selfcare_guidance = MagicMock()
        self.bot._get_treatment_guidance = MagicMock(return_value={
            "home_care": ["Rest"],
            "escalation_timeline": "See doctor if worse",
        })
        self.bot._get_red_flags = MagicMock(return_value={"red_flags": []})
        self.bot._answer_streamed = MagicMock(return_value="Clinical advice")
        self.bot._factcheck = MagicMock(return_value={
            "claim_checks": [],
            "hallucination_risk": "LOW",
            "verdict": "VERIFIED",
        })
        self.bot._xai_explain_streamed = MagicMock(return_value="XAI explanation")

        result = self.bot.chat("I have a minor issue")

        # 1. last_triage must be escalated to SEE_DOCTOR
        self.assertEqual(self.bot.last_triage, "SEE_DOCTOR")
        self.assertEqual(result.get("triage"), "SEE_DOCTOR")

        # 2. _get_selfcare_guidance must NOT be called
        self.bot._get_selfcare_guidance.assert_not_called()
        self.bot._get_treatment_guidance.assert_called_once()

        # 3. triage_floor_reason recorded in xai_logs
        last_log = self.bot.xai_logs[-1]
        self.assertEqual(last_log.get("triage"), "SEE_DOCTOR")
        self.assertEqual(last_log.get("triage_floor_reason"), "EMPTY_DIFFERENTIAL")

    def test_pipeline_with_conditions_agree_set_by_reason_consistent(self):
        """
        Pipeline-level test:
        Verify that _reason_consistent sets 'conditions_agree' on result_a,
        which the pipeline consumes:
          - Pass 1 & Pass 2 disagree -> _reason_consistent sets conditions_agree=False
          - pipeline applies floor: last_triage escalates to SEE_DOCTOR
          - triage_floor_reason is 'CONDITION_DISAGREEMENT'
          - _get_selfcare_guidance is not called
        """
        self.bot._check_followup = MagicMock(return_value={
            "has_enough_info": True,
            "diagnostic_confidence": 75,
            "collected_info": {},
            "followup_question": None,
        })
        self.bot._classify_query = MagicMock(return_value="SYMPTOM")
        self.bot._retrieve = MagicMock(return_value=[])

        # Do NOT mock _reason_consistent; let it run and set conditions_agree
        # Pass 1 returns SELF_CARE with 1 differential (triggers pass 2)
        # Confidence is 75; pass 2 disagreement will reduce by 15 to 60 (>= 50),
        # isolating CONDITION_DISAGREEMENT as the floor reason.
        self.bot._reason = MagicMock(return_value={
            "triage_level": "SELF_CARE",
            "differential_diagnosis": [
                {
                    "rank": 1,
                    "condition": "Common Cold",
                    "probability": "HIGH",
                    "supporting_symptoms": ["cough"],
                }
            ],
            "specialist": "General Practitioner",
            "confidence_score": 75,
            "uncertainty_zones": [],
        })

        # Mock second LLM call for pass 2 to return a conflicting condition
        self.bot._llm = MagicMock(return_value="""{
            "triage_level": "SELF_CARE",
            "differential_diagnosis": [
                {
                    "rank": 1,
                    "condition": "Bacterial Sinusitis",
                    "probability": "HIGH",
                    "supporting_symptoms": ["cough"]
                }
            ],
            "specialist": "General Practitioner",
            "confidence_score": 75,
            "triage_reason": "Mild symptoms"
        }""")

        self.bot._get_selfcare_guidance = MagicMock()
        self.bot._get_treatment_guidance = MagicMock(return_value={
            "home_care": ["Rest"],
            "escalation_timeline": "See doctor if worse",
        })
        self.bot._get_red_flags = MagicMock(return_value={"red_flags": []})
        self.bot._answer_streamed = MagicMock(return_value="Clinical advice")
        self.bot._factcheck = MagicMock(return_value={
            "claim_checks": [],
            "hallucination_risk": "LOW",
            "verdict": "VERIFIED",
        })
        self.bot._xai_explain_streamed = MagicMock(return_value="XAI explanation")

        result = self.bot.chat("I have a persistent cough")

        # Floor must escalate due to conditions_agree=False
        self.assertEqual(self.bot.last_triage, "SEE_DOCTOR")
        self.assertEqual(result.get("triage"), "SEE_DOCTOR")
        self.bot._get_selfcare_guidance.assert_not_called()

        last_log = self.bot.xai_logs[-1]
        self.assertEqual(last_log.get("triage_floor_reason"), "CONDITION_DISAGREEMENT")
        self.assertFalse(last_log["reasoning"]["conditions_agree"])


if __name__ == "__main__":
    unittest.main()
