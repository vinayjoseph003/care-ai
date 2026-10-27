# ==============================================================================
#  tests/test_low_confidence.py — Unit tests for Low-Confidence Mode
#
#  Offline, deterministic test suite using mocks only.
#  Covers:
#    a. Low-confidence detection (confidence < 50 or conditions_agree False):
#       - result["low_confidence"] is True
#       - _answer_streamed receives low_confidence=True
#       - _get_treatment_guidance and _get_red_flags use symptom topic, not last_top_condition
#       - xai_logs contains low_confidence_reason (code only, no symptom text)
#    b. High-confidence mode (confidence >= 50 and conditions_agree True):
#       - low_confidence is False
#       - guidance and red flags use last_top_condition
#    c. Confidence string coercion:
#       - "85%" is treated as numeric 85 (low_confidence is False when conditions_agree True)
# ==============================================================================

import os
import sys
import unittest
from unittest.mock import MagicMock

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Ensure offline mock-only tests can run without optional heavy dependencies
for _mod in ("scipy", "scipy.sparse"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

from bot.careai import CareAI
from core.triage_floor import _coerce_confidence


class TestLowConfidenceMode(unittest.TestCase):

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
        self.bot._classify_query = MagicMock(return_value="SYMPTOM")
        self.bot._retrieve = MagicMock(return_value=[])
        self.bot._factcheck = MagicMock(return_value={
            "claim_checks": [],
            "hallucination_risk": "LOW",
            "verdict": "VERIFIED",
        })
        self.bot._xai_explain_streamed = MagicMock(return_value="XAI explanation")

    def test_low_confidence_mode_triggered(self):
        """
        STEP 1 (a):
        Reasoning: confidence 30, conditions_agree False, differential rank 1 "Typhoid".
        collected_info symptom_or_topic: "sore throat with difficulty swallowing".
        Assert:
          - result["low_confidence"] is True
          - _answer_streamed receives low_confidence=True
          - _get_treatment_guidance called with symptom topic (not "Typhoid")
          - _get_red_flags called with symptom topic (not "Typhoid")
          - xai_logs has "low_confidence_reason" (code only, no symptom text)
        """
        user_message = "i have sore throat and i can't swallow comfortably"
        self.bot.collected_info = {"symptom_or_topic": "sore throat with difficulty swallowing"}

        self.bot._check_followup = MagicMock(return_value={
            "has_enough_info": True,
            "diagnostic_confidence": 30,
            "collected_info": self.bot.collected_info,
            "followup_question": None,
        })

        reasoning = {
            "triage_level": "SEE_DOCTOR",
            "confidence_score": 30,
            "conditions_agree": False,
            "differential_diagnosis": [
                {
                    "rank": 1,
                    "condition": "Typhoid",
                    "probability": "LOW",
                    "supporting_symptoms": ["discomfort upon waking"],
                }
            ],
            "specialist": "General Practitioner",
        }
        self.bot._reason_consistent = MagicMock(return_value=reasoning)

        self.bot._get_treatment_guidance = MagicMock(return_value={"home_care": ["Rest"]})
        self.bot._get_red_flags = MagicMock(return_value={"red_flags": []})
        self.bot._answer_streamed = MagicMock(return_value="Safe next steps for sore throat")

        result = self.bot.chat(user_message)

        # 1. result["low_confidence"] must be True
        self.assertTrue(result.get("low_confidence"), "low_confidence must be True in result")

        # 2. _answer_streamed must receive low_confidence=True
        self.bot._answer_streamed.assert_called_once()
        answer_kwargs = self.bot._answer_streamed.call_args.kwargs
        answer_args = self.bot._answer_streamed.call_args.args
        low_conf_arg = answer_kwargs.get("low_confidence") or (
            answer_args[4] if len(answer_args) > 4 else False
        )
        self.assertTrue(low_conf_arg, "_answer_streamed must receive low_confidence=True")

        # 3. _get_treatment_guidance and _get_red_flags must use symptom topic, not Typhoid
        self.bot._get_treatment_guidance.assert_called_once()
        tg_topic = self.bot._get_treatment_guidance.call_args[0][0]
        self.assertEqual(tg_topic, "sore throat with difficulty swallowing")
        self.assertNotEqual(tg_topic, "Typhoid")

        self.bot._get_red_flags.assert_called_once()
        rf_topic = self.bot._get_red_flags.call_args[0][0]
        self.assertEqual(rf_topic, "sore throat with difficulty swallowing")
        self.assertNotEqual(rf_topic, "Typhoid")

        # _xai_explain_streamed must receive low_confidence=True
        self.bot._xai_explain_streamed.assert_called_once()
        xai_kwargs = self.bot._xai_explain_streamed.call_args.kwargs
        xai_args = self.bot._xai_explain_streamed.call_args.args
        xai_low_conf = xai_kwargs.get("low_confidence") or (
            xai_args[2] if len(xai_args) > 2 else False
        )
        self.assertTrue(xai_low_conf, "_xai_explain_streamed must receive low_confidence=True")

        # 4. xai_logs entry has "low_confidence_reason" (code only, no symptom text)
        last_log = self.bot.xai_logs[-1]
        reason_code = last_log.get("low_confidence_reason")
        self.assertIn(reason_code, ("LOW_CONFIDENCE", "CONDITIONS_DISAGREE", "MISSING_CONFIDENCE"))
        self.assertNotIn("sore throat", str(reason_code).lower())
        self.assertNotIn("swallow", str(reason_code).lower())

    def test_high_confidence_behavior_unchanged(self):
        """
        STEP 1 (b):
        Reasoning: confidence 75, conditions_agree True.
        Assert:
          - low_confidence is False
          - guidance and red flags use last_top_condition ("Common Cold")
        """
        user_message = "i have a cold"
        self.bot.collected_info = {"symptom_or_topic": "runny nose"}

        self.bot._check_followup = MagicMock(return_value={
            "has_enough_info": True,
            "diagnostic_confidence": 75,
            "collected_info": self.bot.collected_info,
            "followup_question": None,
        })

        reasoning = {
            "triage_level": "SEE_DOCTOR",
            "confidence_score": 75,
            "conditions_agree": True,
            "differential_diagnosis": [
                {
                    "rank": 1,
                    "condition": "Common Cold",
                    "probability": "HIGH",
                    "supporting_symptoms": ["cold"],
                }
            ],
            "specialist": "General Practitioner",
        }
        self.bot._reason_consistent = MagicMock(return_value=reasoning)

        self.bot._get_treatment_guidance = MagicMock(return_value={"home_care": ["Rest"]})
        self.bot._get_red_flags = MagicMock(return_value={"red_flags": []})
        self.bot._answer_streamed = MagicMock(return_value="Guidance for cold")

        result = self.bot.chat(user_message)

        self.assertFalse(result.get("low_confidence"), "low_confidence must be False")
        self.bot._get_treatment_guidance.assert_called_once()
        self.assertEqual(self.bot._get_treatment_guidance.call_args[0][0], "Common Cold")
        self.bot._get_red_flags.assert_called_once()
        self.assertEqual(self.bot._get_red_flags.call_args[0][0], "Common Cold")

        self.bot._xai_explain_streamed.assert_called_once()
        xai_kwargs = self.bot._xai_explain_streamed.call_args.kwargs
        xai_args = self.bot._xai_explain_streamed.call_args.args
        xai_low_conf = xai_kwargs.get("low_confidence", False) or (
            xai_args[2] if len(xai_args) > 2 else False
        )
        self.assertFalse(xai_low_conf, "_xai_explain_streamed must receive low_confidence=False")

    def test_confidence_coercion_string_percent(self):
        """
        STEP 1 (c):
        Confidence '85%' (string) is treated as numeric 85.
        """
        self.assertEqual(_coerce_confidence("85%"), 85.0)
        self.assertEqual(_coerce_confidence("  85 % "), 85.0)

        user_message = "i have a cold"
        self.bot.collected_info = {"symptom_or_topic": "runny nose"}
        self.bot._check_followup = MagicMock(return_value={
            "has_enough_info": True,
            "diagnostic_confidence": 85,
            "collected_info": self.bot.collected_info,
            "followup_question": None,
        })

        reasoning = {
            "triage_level": "SEE_DOCTOR",
            "confidence_score": "85%",
            "conditions_agree": True,
            "differential_diagnosis": [
                {
                    "rank": 1,
                    "condition": "Common Cold",
                    "probability": "HIGH",
                    "supporting_symptoms": ["cold"],
                }
            ],
            "specialist": "General Practitioner",
        }
        self.bot._reason_consistent = MagicMock(return_value=reasoning)
        self.bot._get_treatment_guidance = MagicMock(return_value={"home_care": ["Rest"]})
        self.bot._get_red_flags = MagicMock(return_value={"red_flags": []})
        self.bot._answer_streamed = MagicMock(return_value="Guidance for cold")

        result = self.bot.chat(user_message)
        self.assertFalse(result.get("low_confidence"), "String '85%' must coerce to 85 and not trigger low_confidence")

    def test_answer_streamed_prompt_low_confidence(self):
        """
        STEP 1 (a):
        low_confidence=True: the system prompt built in _answer_streamed does not
        contain the top condition name on the TOP DIFFERENTIAL line
        (it says "UNCERTAIN - do not name a diagnosis").
        """
        self.bot._stream_answer = MagicMock(return_value="safe next steps")
        reasoning = {
            "triage_level": "SEE_DOCTOR",
            "specialist": "General Practitioner",
            "differential_diagnosis": [
                {
                    "rank": 1,
                    "condition": "Typhoid",
                    "probability": "LOW",
                    "supporting_symptoms": ["discomfort upon waking"],
                }
            ],
        }

        self.bot._answer_streamed(
            context_str="Context text",
            reasoning=reasoning,
            severity_result=None,
            top_condition="Typhoid",
            low_confidence=True,
        )

        self.bot._stream_answer.assert_called_once()
        messages = self.bot._stream_answer.call_args[0][0]
        system_prompt = messages[0]["content"]

        top_diff_lines = [
            line.strip() for line in system_prompt.splitlines()
            if line.startswith("TOP DIFFERENTIAL:")
        ]
        self.assertEqual(len(top_diff_lines), 1, "Must contain exactly one TOP DIFFERENTIAL line")
        self.assertEqual(
            top_diff_lines[0],
            "TOP DIFFERENTIAL: UNCERTAIN - do not name a diagnosis",
        )
        self.assertNotIn("Typhoid", top_diff_lines[0])

    def test_xai_explain_streamed_prompt_low_confidence(self):
        """
        STEP 1 (b):
        low_confidence=True: the prompt built in _xai_explain_streamed tells the model
        to explain in terms of the user's symptoms and the uncertainty, and does not
        present any condition as the likely cause.
        """
        self.bot.history = [{"role": "user", "content": "i have a sore throat"}]
        self.bot.last_top_condition = "Typhoid"
        self.bot._stream_xai = MagicMock(return_value="explanation")
        self.bot._xai_explain_streamed = CareAI._xai_explain_streamed.__get__(self.bot, CareAI)

        reasoning = {
            "differential_diagnosis": [
                {
                    "rank": 1,
                    "condition": "Typhoid",
                    "probability": "LOW",
                    "supporting_symptoms": ["discomfort upon waking"],
                    "key_evidence": "discomfort upon waking",
                }
            ],
            "uncertainty_zones": ["symptom overlap with viral illness"],
            "triage_level": "SEE_DOCTOR",
            "triage_reason": "Persistent symptoms",
            "knowledge_sources": ["MedQuAD"],
        }
        factcheck = {"verdict": "VERIFIED"}

        self.bot._xai_explain_streamed(reasoning, factcheck, low_confidence=True)

        self.bot._stream_xai.assert_called_once()
        messages = self.bot._stream_xai.call_args[0][0]
        user_prompt = messages[1]["content"]

        self.assertNotIn("Top condition identified", user_prompt)
        self.assertNotIn("Top condition identified: Typhoid", user_prompt)
        self.assertIn("explain in terms of the user's symptoms and the uncertainty", user_prompt.lower())
        self.assertIn("do not present any condition as the likely cause", user_prompt.lower())

    def test_prompts_unchanged_when_high_confidence(self):
        """
        STEP 1 (c):
        low_confidence=False: both prompts are unchanged from current behavior.
        """
        self.bot.history = [{"role": "user", "content": "i have a cold"}]
        self.bot.last_top_condition = "Common Cold"
        self.bot._stream_answer = MagicMock(return_value="answer")
        self.bot._stream_xai = MagicMock(return_value="explanation")
        self.bot._xai_explain_streamed = CareAI._xai_explain_streamed.__get__(self.bot, CareAI)

        reasoning = {
            "triage_level": "SEE_DOCTOR",
            "specialist": "General Practitioner",
            "differential_diagnosis": [
                {
                    "rank": 1,
                    "condition": "Common Cold",
                    "probability": "HIGH",
                    "supporting_symptoms": ["runny nose"],
                    "key_evidence": "runny nose",
                }
            ],
            "uncertainty_zones": ["none"],
            "triage_reason": "Viral cold symptoms",
            "knowledge_sources": ["MedQuAD"],
        }
        factcheck = {"verdict": "VERIFIED"}

        # Check _answer_streamed prompt with low_confidence=False
        self.bot._answer_streamed(
            context_str="Context text",
            reasoning=reasoning,
            severity_result=None,
            top_condition="Common Cold",
            low_confidence=False,
        )
        ans_messages = self.bot._stream_answer.call_args[0][0]
        ans_system = ans_messages[0]["content"]
        self.assertIn("TOP DIFFERENTIAL: Common Cold", ans_system)
        self.assertNotIn("TOP DIFFERENTIAL: UNCERTAIN", ans_system)

        # Check _xai_explain_streamed prompt with low_confidence=False
        self.bot._xai_explain_streamed(reasoning, factcheck, low_confidence=False)
        xai_messages = self.bot._stream_xai.call_args[0][0]
        xai_user = xai_messages[1]["content"]
        self.assertIn("Top condition identified: Common Cold", xai_user)
        self.assertIn("Key symptom pointing to this: runny nose", xai_user)
        self.assertNotIn("Assessment status: UNCERTAIN", xai_user)


if __name__ == "__main__":
    unittest.main()
