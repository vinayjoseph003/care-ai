"""
Tests for multilingual safety messages:
- EMERGENCY_MESSAGES and UNAVAILABLE_MESSAGES dictionaries
- safety_lang helper (script detection with session language fallback)
- Hindi emergency without LLM/router invocation
- Telugu outage returning translated service_unavailable message
- Preservation of existing English behavior
"""

import sys
import unittest
from unittest.mock import MagicMock

# Ensure offline mock-only tests can run without optional heavy dependencies (e.g. scipy)
for _mod in ("scipy", "scipy.sparse"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from bot.careai import (
    CareAI,
    SAFE_UNAVAILABLE_MESSAGE,
)
from core.llm_router import AllProvidersFailedError, LLMRouter

try:
    from bot.careai import EMERGENCY_MESSAGES, UNAVAILABLE_MESSAGES, safety_lang
except ImportError:
    EMERGENCY_MESSAGES = {}
    UNAVAILABLE_MESSAGES = {}
    def safety_lang(user_message, user_language_code):
        return "en"


class TestTranslatedSafety(unittest.TestCase):
    def test_messages_contain_112_and_no_911_or_999(self):
        """Every message in both dicts contains '112' and not '911'/'999'."""
        self.assertTrue(EMERGENCY_MESSAGES, "EMERGENCY_MESSAGES must not be empty")
        self.assertTrue(UNAVAILABLE_MESSAGES, "UNAVAILABLE_MESSAGES must not be empty")

        for lang in ("en", "hi", "te", "ta"):
            self.assertIn(lang, EMERGENCY_MESSAGES, f"Missing {lang} in EMERGENCY_MESSAGES")
            msg = EMERGENCY_MESSAGES[lang]
            self.assertIn("112", msg, f"EMERGENCY_MESSAGES[{lang}] missing 112")
            self.assertNotIn("911", msg, f"EMERGENCY_MESSAGES[{lang}] contains 911")
            self.assertNotIn("999", msg, f"EMERGENCY_MESSAGES[{lang}] contains 999")

            self.assertIn(lang, UNAVAILABLE_MESSAGES, f"Missing {lang} in UNAVAILABLE_MESSAGES")
            umsg = UNAVAILABLE_MESSAGES[lang]
            self.assertIn("112", umsg, f"UNAVAILABLE_MESSAGES[{lang}] missing 112")
            self.assertNotIn("911", umsg, f"UNAVAILABLE_MESSAGES[{lang}] contains 911")
            self.assertNotIn("999", umsg, f"UNAVAILABLE_MESSAGES[{lang}] contains 999")

    def test_safety_lang_matching_script(self):
        """Helper returns hi/te/ta for matching script in user_message."""
        self.assertEqual(safety_lang("सीने में दर्द", "en"), "hi")
        self.assertEqual(safety_lang("ఛాతీ నొప్పి", "en"), "te")
        self.assertEqual(safety_lang("நெஞ்சு வலி", "en"), "ta")

    def test_safety_lang_prior_indic_code_with_english_text(self):
        """English text with a prior hi/te/ta language code -> matching code."""
        self.assertEqual(safety_lang("I have chest pain", "hi"), "hi")
        self.assertEqual(safety_lang("I have severe pain", "te"), "te")
        self.assertEqual(safety_lang("I cannot breathe", "ta"), "ta")

    def test_safety_lang_unknown_script_or_en(self):
        """Unknown script and 'en' -> en."""
        self.assertEqual(safety_lang("I have chest pain", "en"), "en")
        self.assertEqual(safety_lang("J'ai mal à la poitrine", "fr"), "en")
        self.assertEqual(safety_lang("Tengo dolor de pecho", "es"), "en")

    def test_hindi_chest_pain_emergency_no_router_call(self):
        """Hindi chest-pain text returns the Hindi emergency message and router is never called."""
        mock_router = MagicMock(spec=LLMRouter)
        mock_vs = MagicMock()
        mock_vs.search.return_value = []
        mock_scorer = MagicMock()

        bot = CareAI(
            vector_store=mock_vs,
            symptom_scorer=mock_scorer,
            router=mock_router,
        )

        res = bot.chat("मुझे सीने में दर्द है")
        self.assertEqual(res.get("type"), "emergency")
        self.assertEqual(res.get("message"), EMERGENCY_MESSAGES.get("hi"))
        self.assertEqual(mock_router.complete.call_count, 0)
        self.assertEqual(mock_router.stream.call_count, 0)

    def test_outage_with_telugu_message_returns_telugu_text(self):
        """Outage with a Telugu message returns the Telugu unavailable text."""
        mock_router = MagicMock(spec=LLMRouter)
        mock_router.complete.side_effect = AllProvidersFailedError("All LLM providers failed.")
        mock_router.stream.side_effect = AllProvidersFailedError("All LLM providers failed.")
        mock_vs = MagicMock()
        mock_vs.search.return_value = []
        mock_scorer = MagicMock()

        bot = CareAI(
            vector_store=mock_vs,
            symptom_scorer=mock_scorer,
            router=mock_router,
        )

        res = bot.chat("నాకు తలనొప్పిగా ఉంది")
        self.assertEqual(res.get("type"), "service_unavailable")
        self.assertEqual(res.get("message"), UNAVAILABLE_MESSAGES.get("te"))

    def test_existing_english_tests_stay_green(self):
        """English emergency and outage messages retain English text."""
        mock_router = MagicMock(spec=LLMRouter)
        mock_router.complete.side_effect = AllProvidersFailedError("All LLM providers failed.")
        mock_vs = MagicMock()
        mock_scorer = MagicMock()

        bot = CareAI(
            vector_store=mock_vs,
            symptom_scorer=mock_scorer,
            router=mock_router,
        )

        # English emergency
        em_res = bot.chat("I have severe chest pain")
        self.assertEqual(em_res.get("type"), "emergency")
        self.assertIn("MEDICAL EMERGENCY DETECTED", em_res.get("message", ""))
        self.assertIn("112", em_res.get("message", ""))

        # English outage
        bot.reset()
        out_res = bot.chat("I have a mild cough")
        self.assertEqual(out_res.get("type"), "service_unavailable")
        self.assertEqual(out_res.get("message"), SAFE_UNAVAILABLE_MESSAGE)


if __name__ == "__main__":
    unittest.main()

