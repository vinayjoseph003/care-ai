"""
Tests for clinical safety prompt rules in core/prompts.py.
Verifies:
- Activity-triggered pain rule contains an EXCEPTIONS clause for chest, abdomen, pelvis, groin
- The clause explicitly mentions red flags (fever, vomiting, shortness of breath, radiating pain)
- reasoning_trace schema documentation carries the same exception
- The activity rule preserves exercise, workout, gym, lifting
- REASONING_PROMPT formats without KeyError using the same args the code passes
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from core.prompts import REASONING_PROMPT
from bot.careai import CONVERSATIONAL_WRAPPER


class TestPromptRules(unittest.TestCase):

    def test_activity_rule_contains_exceptions_clause_for_vital_sites(self):
        """The activity rule must contain an EXCEPTIONS clause naming chest, abdomen/abdominal, pelvis/pelvic, groin."""
        self.assertIn("ACTIVITY-TRIGGERED PAIN RULE", REASONING_PROMPT)
        idx = REASONING_PROMPT.index("ACTIVITY-TRIGGERED PAIN RULE")
        rule_section = REASONING_PROMPT[idx:idx + 1000]

        self.assertIn("EXCEPTIONS", rule_section)
        rule_lower = rule_section.lower()
        self.assertIn("chest", rule_lower)
        self.assertTrue("abdominal" in rule_lower or "abdomen" in rule_lower)
        self.assertTrue("pelvic" in rule_lower or "pelvis" in rule_lower)
        self.assertIn("groin", rule_lower)

    def test_activity_rule_exceptions_clause_mentions_red_flags(self):
        """The exception clause must mention red flags: fever, vomiting, shortness of breath, radiating pain."""
        idx = REASONING_PROMPT.index("ACTIVITY-TRIGGERED PAIN RULE")
        rule_section = REASONING_PROMPT[idx:idx + 1000].lower()

        self.assertIn("red flag", rule_section)
        self.assertIn("fever", rule_section)
        self.assertIn("vomiting", rule_section)
        self.assertIn("shortness of breath", rule_section)
        self.assertIn("radiating pain", rule_section)

    def test_reasoning_trace_carries_same_exceptions(self):
        """The reasoning_trace line must carry the same exception for chest, abdominal/pelvic/groin or red flags."""
        idx = REASONING_PROMPT.index('"reasoning_trace"')
        line = REASONING_PROMPT[idx:REASONING_PROMPT.index('\n', idx)].lower()

        self.assertIn("chest", line)
        self.assertTrue("abdominal" in line or "abdomen" in line)
        self.assertTrue("pelvic" in line or "pelvis" in line)
        self.assertIn("groin", line)
        self.assertIn("red flag", line)

    def test_activity_rule_still_lists_exercise_workout_gym_lifting(self):
        """The rule must still preserve exercise, workout, gym, lifting."""
        idx = REASONING_PROMPT.index("ACTIVITY-TRIGGERED PAIN RULE")
        rule_section = REASONING_PROMPT[idx:idx + 1000].lower()

        self.assertIn("exercise", rule_section)
        self.assertIn("workout", rule_section)
        self.assertIn("gym", rule_section)
        self.assertIn("lifting", rule_section)

    def test_reasoning_prompt_formats_without_key_error(self):
        """CONVERSATIONAL_WRAPPER formats base_prompt=REASONING_PROMPT without KeyError or formatting bugs."""
        wrapped = CONVERSATIONAL_WRAPPER.format(base_prompt=REASONING_PROMPT)
        self.assertIn("ACTIVITY-TRIGGERED PAIN RULE", wrapped)


if __name__ == "__main__":
    unittest.main()

