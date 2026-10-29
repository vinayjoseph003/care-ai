"""
tests/test_output_consistency.py — Output consistency and prompt alignment tests.

Tests:
1. TREATMENT_PROMPT: escalation_timeline rule is conditioned on triage:
   - SEE_DOCTOR or URGENT_CARE -> must NOT advise waiting days ("consult a doctor within 24-48 hours, or emergency care if [symptom]")
   - SELF_CARE -> monitor 3-5 days
2. XAI prompt (core/prompts.py):
   - Closing sentence conditioned on triage
   - No "rest easy" / "feeling better soon" style reassurance unless SELF_CARE
   - For SEE_DOCTOR / URGENT_CARE: calm reminder to get medical evaluation
3. FACTCHECK_PROMPT:
   - Exemption: do NOT flag claims that restate user's own reported symptoms, timeline or history as unsupported
4. bot/careai.py:
   - Removing "Self-consistency: skipped" from uncertainty_zones when single pass is sufficient
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

from core.prompts import TREATMENT_PROMPT, XAI_EXPLAINER_PROMPT, FACTCHECK_PROMPT
from bot.careai import CareAI


class TestOutputConsistencyPrompts(unittest.TestCase):

    def test_treatment_prompt_escalation_timeline_conditioned_on_triage(self):
        """TREATMENT_PROMPT escalation_timeline rule must be conditioned on triage."""
        prompt_lower = TREATMENT_PROMPT.lower()
        self.assertIn("escalation_timeline", prompt_lower)
        self.assertIn("see_doctor", prompt_lower)
        self.assertIn("urgent_care", prompt_lower)
        self.assertIn("24-48 hours", prompt_lower)
        self.assertIn("3-5 days", prompt_lower)
        self.assertIn("self_care", prompt_lower)

    def test_xai_prompt_closing_sentence_conditioned_on_triage(self):
        """XAI closing sentence must be conditioned on triage: no rest easy unless SELF_CARE."""
        prompt_lower = XAI_EXPLAINER_PROMPT.lower()
        self.assertIn("closing sentence", prompt_lower)
        self.assertIn("rest easy", prompt_lower)
        self.assertIn("self_care", prompt_lower)
        self.assertTrue("medical evaluation" in prompt_lower or "doctor" in prompt_lower)

    def test_factcheck_prompt_user_history_exemption(self):
        """FACTCHECK_PROMPT must exempt claims restating user's symptoms, timeline, or history."""
        prompt_lower = FACTCHECK_PROMPT.lower()
        self.assertIn("symptoms", prompt_lower)
        self.assertIn("timeline", prompt_lower)
        self.assertIn("history", prompt_lower)
        self.assertTrue("restate" in prompt_lower or "user's own" in prompt_lower)


class TestSelfConsistencySkippedUncertaintyZones(unittest.TestCase):

    def test_single_pass_does_not_append_skipped_to_uncertainty_zones(self):
        """When single pass is sufficient, 'Self-consistency: skipped' must not be added to uncertainty_zones."""
        bot = CareAI.__new__(CareAI)
        bot.CONSISTENCY_THRESHOLD = 70
        bot.history = []
        bot.collected_info = {}

        result_a = {
            "confidence_score": 85,
            "differential_diagnosis": [
                {"condition": "Common Cold"},
                {"condition": "Viral Pharyngitis"},
            ],
            "uncertainty_zones": ["Uncertainty about onset"],
        }
        bot._reason = MagicMock(return_value=result_a)

        out = bot._reason_consistent([])
        self.assertTrue(out.get("conditions_agree"))
        uncertainty = out.get("uncertainty_zones", [])
        for u in uncertainty:
            self.assertNotIn("Self-consistency: skipped", u)


if __name__ == "__main__":
    unittest.main()

