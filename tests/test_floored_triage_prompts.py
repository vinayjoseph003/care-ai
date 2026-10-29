"""
tests/test_floored_triage_prompts.py — Verify that post-floor triage is passed to all downstream prompts.

Tests:
1. When apply_triage_floor escalates SELF_CARE -> SEE_DOCTOR:
   - Answer prompt contains "TRIAGE LEVEL: SEE_DOCTOR" and NOT "SELF_CARE"
   - XAI prompt contains "Triage decision: SEE_DOCTOR" and NOT "Triage decision: SELF_CARE"
   - Treatment prompt contains "Triage level: SEE_DOCTOR" and NOT "SELF-CARE"
2. When SELF_CARE is unfloored:
   - Downstream prompts receive SELF_CARE
3. Emergency paths:
   - Triage levels (EMERGENCY / URGENT_CARE) remain unchanged and are passed through
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

from bot.careai import CareAI


class TestFlooredTriagePrompts(unittest.TestCase):

    def _setup_bot(self, reasoning_triage="SELF_CARE", confidence=40, severity="LOW", user_sev=None):
        """Build CareAI with stubbed components and canned reasoning."""
        mock_router = MagicMock()
        mock_vs = MagicMock()
        mock_vs.search.return_value = []
        mock_vs.search_by_symptoms.return_value = []
        mock_scorer = MagicMock()
        mock_scorer.extract_symptoms_from_text.return_value = []
        mock_scorer.score.return_value = {"severity_level": severity}

        bot = CareAI(
            vector_store=mock_vs,
            symptom_scorer=mock_scorer,
            router=mock_router,
        )
        bot._detect_language = MagicMock(return_value={"language": "English", "code": "en"})
        bot._check_followup = MagicMock(return_value={
            "has_enough_info": True,
            "diagnostic_confidence": 75,
            "collected_info": {"severity": user_sev} if user_sev else {},
            "followup_question": None,
        })
        bot._classify_query = MagicMock(return_value="SYMPTOM")
        bot._check_red_flags = MagicMock(return_value=(False, []))
        bot._is_emergency = MagicMock(return_value=False)
        bot._retrieve = MagicMock(return_value=[])

        canned_reasoning = {
            "triage_level": reasoning_triage,
            "differential_diagnosis": [
                {
                    "rank": 1,
                    "condition": "Tension Headache",
                    "probability": "HIGH",
                    "supporting_symptoms": ["headache"],
                    "key_evidence": "mild pain",
                }
            ],
            "specialist": "General Practitioner",
            "confidence_score": confidence,
            "conditions_agree": True,
            "triage_reason": "Based on symptoms",
            "uncertainty_zones": [],
            "knowledge_sources": ["medical guide"],
        }
        bot._reason_consistent = MagicMock(return_value=canned_reasoning)
        bot._factcheck = MagicMock(return_value={
            "claim_checks": [],
            "hallucination_risk": "LOW",
            "verdict": "VERIFIED",
        })

        # Track LLM prompt payloads
        recorded_prompts = []

        def fake_llm(system, user_content, **kwargs):
            recorded_prompts.append({"system": system, "user": user_content, "kwargs": kwargs})
            return "{}"

        bot._llm = fake_llm

        def fake_stream_answer(messages):
            # Capture answer prompt system message
            recorded_prompts.append({"type": "answer", "messages": messages})
            return "This is a canned answer [1]."

        def fake_stream_xai(messages, label=""):
            recorded_prompts.append({"type": "xai", "messages": messages})
            return "• Because symptom → condition\n• Because test → guide\n• Therefore see doctor because of floor."

        bot._stream_answer = fake_stream_answer
        bot._stream_xai = fake_stream_xai

        return bot, recorded_prompts

    def test_floored_self_care_passes_see_doctor_to_answer_and_xai_prompts(self):
        """When low confidence (<50) floors SELF_CARE -> SEE_DOCTOR, prompts must get SEE_DOCTOR."""
        bot, prompts = self._setup_bot(reasoning_triage="SELF_CARE", confidence=40)
        bot.chat("I have a slight headache")

        self.assertEqual(bot.last_triage, "SEE_DOCTOR")

        # 1. Answer prompt check
        answer_events = [p for p in prompts if p.get("type") == "answer"]
        self.assertEqual(len(answer_events), 1)
        answer_system = answer_events[0]["messages"][0]["content"]
        self.assertIn("TRIAGE LEVEL: SEE_DOCTOR", answer_system)
        self.assertNotIn("TRIAGE LEVEL: SELF_CARE", answer_system)
        # Ensure reasoning dumped in answer prompt does not claim SELF_CARE
        self.assertNotIn('"triage_level": "SELF_CARE"', answer_system)

        # 2. XAI prompt check
        xai_events = [p for p in prompts if p.get("type") == "xai"]
        self.assertEqual(len(xai_events), 1)
        xai_user = xai_events[0]["messages"][1]["content"]
        self.assertIn("Triage decision: SEE_DOCTOR", xai_user)
        self.assertNotIn("Triage decision: SELF_CARE", xai_user)

        # 3. Treatment guidance prompt check
        # Since last_triage is SEE_DOCTOR, _get_treatment_guidance was called, not _get_selfcare_guidance
        treatment_calls = [
            p for p in prompts
            if "system" in p and "TREATMENT_PROMPT" in str(p["system"]) or "Triage level:" in str(p.get("user", ""))
        ]
        self.assertTrue(len(treatment_calls) >= 1)
        treatment_user = treatment_calls[0]["user"]
        self.assertIn("Triage level: SEE_DOCTOR", treatment_user)
        self.assertNotIn("SELF-CARE level", treatment_user)

    def test_unfloored_self_care_stays_self_care(self):
        """When inputs are clean and confidence is high, SELF_CARE is preserved in all prompts."""
        bot, prompts = self._setup_bot(reasoning_triage="SELF_CARE", confidence=85)
        bot.chat("I have a slight headache")

        self.assertEqual(bot.last_triage, "SELF_CARE")

        # Answer prompt
        answer_events = [p for p in prompts if p.get("type") == "answer"]
        self.assertEqual(len(answer_events), 1)
        answer_system = answer_events[0]["messages"][0]["content"]
        self.assertIn("TRIAGE LEVEL: SELF_CARE", answer_system)

        # XAI prompt
        xai_events = [p for p in prompts if p.get("type") == "xai"]
        self.assertEqual(len(xai_events), 1)
        xai_user = xai_events[0]["messages"][1]["content"]
        self.assertIn("Triage decision: SELF_CARE", xai_user)

        # Treatment prompt (calls _get_selfcare_guidance)
        treatment_calls = [
            p for p in prompts
            if "system" in p and "TREATMENT_PROMPT" in str(p["system"]) or "SELF-CARE level" in str(p.get("user", ""))
        ]
        self.assertTrue(len(treatment_calls) >= 1)
        self.assertIn("SELF-CARE level", treatment_calls[0]["user"])

    def test_emergency_triage_passed_through_unchanged(self):
        """Emergency triage level passes through untouched to all prompts."""
        bot, prompts = self._setup_bot(reasoning_triage="EMERGENCY", confidence=90, severity="EMERGENCY")
        bot.chat("I have crushing chest pain")

        self.assertEqual(bot.last_triage, "EMERGENCY")

        # Answer prompt
        answer_events = [p for p in prompts if p.get("type") == "answer"]
        self.assertEqual(len(answer_events), 1)
        answer_system = answer_events[0]["messages"][0]["content"]
        self.assertIn("TRIAGE LEVEL: EMERGENCY", answer_system)

        # XAI prompt
        xai_events = [p for p in prompts if p.get("type") == "xai"]
        self.assertEqual(len(xai_events), 1)
        xai_user = xai_events[0]["messages"][1]["content"]
        self.assertIn("Triage decision: EMERGENCY", xai_user)


if __name__ == "__main__":
    unittest.main()
