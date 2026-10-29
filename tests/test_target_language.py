"""
tests/test_target_language.py — Unit and behavioural tests for target response language line.

Verifies:
- target_language_line pure helper output for English, Hindi, Telugu, None, {}, and missing "language" key
- Prepending of target_language_line in user prompts at the four LLM call sites:
  1. REASONING_PROMPT in _reason()
  2. REASONING_PROMPT in _run_differential_sc() (pass 2)
  3. TREATMENT_PROMPT in _get_selfcare_guidance()
  4. TREATMENT_PROMPT in _get_treatment_guidance()
"""

import sys
import unittest
from unittest.mock import MagicMock

# Mock scipy if not present in environment
for _mod in ("scipy", "scipy.sparse"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from bot.careai import CareAI, target_language_line


class TestTargetLanguage(unittest.TestCase):
    # ── 1. Pure Helper Unit Tests ───────────────────────────────

    def test_helper_english(self):
        line = target_language_line({"language": "English", "code": "en"})
        expected = (
            "Target response language: English. "
            "Write free-text values (condition descriptions, key findings, guidance, explanations) in this language. "
            "Keep JSON keys and all enumerated or categorical values (triage levels, probability levels such as HIGH/MEDIUM/LOW, and the specialist label) exactly as the prompt specifies, in English."
        )
        self.assertEqual(line, expected)
        self.assertIn("specialist", line)
        self.assertIn("HIGH/MEDIUM/LOW", line)
        self.assertIn("English", line)

    def test_helper_hindi(self):
        line = target_language_line({"language": "Hindi", "code": "hi"})
        expected = (
            "Target response language: Hindi. "
            "Write free-text values (condition descriptions, key findings, guidance, explanations) in this language. "
            "Keep JSON keys and all enumerated or categorical values (triage levels, probability levels such as HIGH/MEDIUM/LOW, and the specialist label) exactly as the prompt specifies, in English."
        )
        self.assertEqual(line, expected)
        self.assertIn("specialist", line)
        self.assertIn("HIGH/MEDIUM/LOW", line)
        self.assertIn("English", line)

    def test_helper_telugu(self):
        line = target_language_line({"language": "Telugu", "code": "te"})
        expected = (
            "Target response language: Telugu. "
            "Write free-text values (condition descriptions, key findings, guidance, explanations) in this language. "
            "Keep JSON keys and all enumerated or categorical values (triage levels, probability levels such as HIGH/MEDIUM/LOW, and the specialist label) exactly as the prompt specifies, in English."
        )
        self.assertEqual(line, expected)
        self.assertIn("specialist", line)
        self.assertIn("HIGH/MEDIUM/LOW", line)
        self.assertIn("English", line)

    def test_helper_none(self):
        line = target_language_line(None)
        expected = (
            "Target response language: English. "
            "Write free-text values (condition descriptions, key findings, guidance, explanations) in this language. "
            "Keep JSON keys and all enumerated or categorical values (triage levels, probability levels such as HIGH/MEDIUM/LOW, and the specialist label) exactly as the prompt specifies, in English."
        )
        self.assertEqual(line, expected)
        self.assertIn("specialist", line)
        self.assertIn("HIGH/MEDIUM/LOW", line)
        self.assertIn("English", line)

    def test_helper_empty_dict(self):
        line = target_language_line({})
        expected = (
            "Target response language: English. "
            "Write free-text values (condition descriptions, key findings, guidance, explanations) in this language. "
            "Keep JSON keys and all enumerated or categorical values (triage levels, probability levels such as HIGH/MEDIUM/LOW, and the specialist label) exactly as the prompt specifies, in English."
        )
        self.assertEqual(line, expected)
        self.assertIn("specialist", line)
        self.assertIn("HIGH/MEDIUM/LOW", line)
        self.assertIn("English", line)

    def test_helper_missing_language_key(self):
        line = target_language_line({"code": "hi"})
        expected = (
            "Target response language: English. "
            "Write free-text values (condition descriptions, key findings, guidance, explanations) in this language. "
            "Keep JSON keys and all enumerated or categorical values (triage levels, probability levels such as HIGH/MEDIUM/LOW, and the specialist label) exactly as the prompt specifies, in English."
        )
        self.assertEqual(line, expected)
        self.assertIn("specialist", line)
        self.assertIn("HIGH/MEDIUM/LOW", line)
        self.assertIn("English", line)

    # ── 2. Behavioural Call Site Tests ──────────────────────────

    def test_reason_prepends_target_language(self):
        """_reason() prepends target_language_line to user prompt for REASONING_PROMPT."""
        bot = CareAI.__new__(CareAI)
        bot.history = [{"role": "user", "content": "bukhaar hai"}]
        bot.collected_info = {"onset": "yesterday"}
        bot.user_language = {"language": "Hindi", "code": "hi"}

        recorded_calls = []

        def fake_llm(system, user_content, **kwargs):
            recorded_calls.append((system, user_content, kwargs))
            return "{}"

        bot._llm = fake_llm
        bot._parse_json = lambda raw, default: default

        bot._reason("MedlinePlus: Fever context")

        self.assertEqual(len(recorded_calls), 1)
        system, user_content, _ = recorded_calls[0]
        expected_line = target_language_line({"language": "Hindi", "code": "hi"})
        self.assertTrue(
            user_content.startswith(expected_line),
            f"User prompt does not start with target language line:\n{user_content[:200]}"
        )
        self.assertIn("Conversation:", user_content)

    def test_run_differential_sc_prepends_target_language(self):
        """_run_differential_sc() pass 2 prepends target_language_line to user prompt for REASONING_PROMPT."""
        bot = CareAI.__new__(CareAI)
        bot.history = [{"role": "user", "content": "talanoppi ga undi"}]
        bot.collected_info = {}
        bot.CONSISTENCY_THRESHOLD = 90
        bot.user_language = {"language": "Telugu", "code": "te"}

        recorded_calls = []

        def fake_llm(system, user_content, **kwargs):
            recorded_calls.append((system, user_content, kwargs))
            return "{}"

        bot._llm = fake_llm
        bot._parse_json = lambda raw, default: {"differential_diagnosis": [{"condition": "Tension Headache"}]}

        bot._reason = lambda ctx: {
            "confidence_score": 50,
            "differential_diagnosis": [{"condition": "Migraine"}],
            "uncertainty_zones": [],
        }

        bot._reason_consistent("MedlinePlus: Headache context")

        self.assertEqual(len(recorded_calls), 1)
        system, user_content, _ = recorded_calls[0]
        expected_line = target_language_line({"language": "Telugu", "code": "te"})
        self.assertTrue(
            user_content.startswith(expected_line),
            f"User prompt does not start with target language line:\n{user_content[:200]}"
        )
        self.assertIn("Conversation:", user_content)

    def test_get_selfcare_guidance_prepends_target_language(self):
        """_get_selfcare_guidance() prepends target_language_line to user prompt for TREATMENT_PROMPT."""
        bot = CareAI.__new__(CareAI)
        bot.precaution_map = {}
        bot.user_language = {"language": "Hindi", "code": "hi"}

        recorded_calls = []

        def fake_llm(system, user_content, **kwargs):
            recorded_calls.append((system, user_content, kwargs))
            return "{}"

        bot._llm = fake_llm
        bot._parse_json = lambda raw, default: default

        bot._get_selfcare_guidance("Common Cold")

        self.assertEqual(len(recorded_calls), 1)
        system, user_content, _ = recorded_calls[0]
        expected_line = target_language_line({"language": "Hindi", "code": "hi"})
        self.assertTrue(
            user_content.startswith(expected_line),
            f"User prompt does not start with target language line:\n{user_content[:200]}"
        )
        self.assertIn("Condition: Common Cold", user_content)

    def test_get_treatment_guidance_prepends_target_language(self):
        """_get_treatment_guidance() prepends target_language_line to user prompt for TREATMENT_PROMPT."""
        bot = CareAI.__new__(CareAI)
        bot.precaution_map = {}
        bot.user_language = {"language": "Telugu", "code": "te"}

        recorded_calls = []

        def fake_llm(system, user_content, **kwargs):
            recorded_calls.append((system, user_content, kwargs))
            return "{}"

        bot._llm = fake_llm
        bot._parse_json = lambda raw, default: default

        bot._get_treatment_guidance("Hypertension", "SEE_DOCTOR")

        self.assertEqual(len(recorded_calls), 1)
        system, user_content, _ = recorded_calls[0]
        expected_line = target_language_line({"language": "Telugu", "code": "te"})
        self.assertTrue(
            user_content.startswith(expected_line),
            f"User prompt does not start with target language line:\n{user_content[:200]}"
        )
        self.assertIn("Condition: Hypertension", user_content)


if __name__ == "__main__":
    unittest.main()
