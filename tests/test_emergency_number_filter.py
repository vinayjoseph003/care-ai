"""
tests/test_emergency_number_filter.py — Tests for emergency number normalization (911/999 -> 112).

Verifies:
- "call 911 now" -> "call 112 now"
- "dial 999" -> "dial 112"
- "1911", "9110mg", "2999" unchanged (digit boundaries)
- "take 999 mg" and "999mg" unchanged (unit boundaries with optional whitespace)
- All specified units (mg, mcg, g, ml, kg, cm, mm, iu, %, units) prevent replacement
- "112" and "108" are untouched
- "" and None safe (None or non-str returned unchanged)
- Two occurrences in the same string replaced
- Nested treatment dict/list recursively normalized without mutating originals
- Streaming buffer chunk-split case ("9" then "11") never displays 911 in rendered output
"""

import sys
import unittest
from unittest.mock import MagicMock

# Mock scipy if not present in environment
for _mod in ("scipy", "scipy.sparse"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from core.text_safety import normalize_emergency_numbers


class TestEmergencyNumberFilter(unittest.TestCase):
    def test_basic_replacements(self):
        """Standalone 911 and 999 are normalized to 112."""
        self.assertEqual(normalize_emergency_numbers("call 911 now"), "call 112 now")
        self.assertEqual(normalize_emergency_numbers("dial 999"), "dial 112")
        self.assertEqual(normalize_emergency_numbers("In emergency, call 911 immediately."), "In emergency, call 112 immediately.")
        self.assertEqual(normalize_emergency_numbers("dial 999 or go to the nearest ER"), "dial 112 or go to the nearest ER")

    def test_digit_boundary_guards(self):
        """Numbers with adjacent digits (1911, 9110mg, 2999) must remain unchanged."""
        self.assertEqual(normalize_emergency_numbers("1911"), "1911")
        self.assertEqual(normalize_emergency_numbers("9110mg"), "9110mg")
        self.assertEqual(normalize_emergency_numbers("2999"), "2999")
        self.assertEqual(normalize_emergency_numbers("The year was 1911."), "The year was 1911.")
        self.assertEqual(normalize_emergency_numbers("Reference code: 9110."), "Reference code: 9110.")

    def test_unit_boundary_guards(self):
        """Numbers followed by optional whitespace and a medical/measurement unit must remain unchanged."""
        # mg
        self.assertEqual(normalize_emergency_numbers("take 999 mg"), "take 999 mg")
        self.assertEqual(normalize_emergency_numbers("999mg"), "999mg")
        self.assertEqual(normalize_emergency_numbers("take 911 mg"), "take 911 mg")
        self.assertEqual(normalize_emergency_numbers("911mg"), "911mg")

        # Other units: mcg, g, ml, kg, cm, mm, iu, %, units
        self.assertEqual(normalize_emergency_numbers("dose is 911 mcg"), "dose is 911 mcg")
        self.assertEqual(normalize_emergency_numbers("dose is 911mcg"), "dose is 911mcg")
        self.assertEqual(normalize_emergency_numbers("weight is 999 g"), "weight is 999 g")
        self.assertEqual(normalize_emergency_numbers("weight is 999g"), "weight is 999g")
        self.assertEqual(normalize_emergency_numbers("volume 911 ml"), "volume 911 ml")
        self.assertEqual(normalize_emergency_numbers("volume 911ml"), "volume 911ml")
        self.assertEqual(normalize_emergency_numbers("load 999 kg"), "load 999 kg")
        self.assertEqual(normalize_emergency_numbers("load 999kg"), "load 999kg")
        self.assertEqual(normalize_emergency_numbers("height 911 cm"), "height 911 cm")
        self.assertEqual(normalize_emergency_numbers("height 911cm"), "height 911cm")
        self.assertEqual(normalize_emergency_numbers("size 999 mm"), "size 999 mm")
        self.assertEqual(normalize_emergency_numbers("size 999mm"), "size 999mm")
        self.assertEqual(normalize_emergency_numbers("dose 911 iu"), "dose 911 iu")
        self.assertEqual(normalize_emergency_numbers("dose 911IU"), "dose 911IU")
        self.assertEqual(normalize_emergency_numbers("999%"), "999%")
        self.assertEqual(normalize_emergency_numbers("999 %"), "999 %")
        self.assertEqual(normalize_emergency_numbers("give 911 units"), "give 911 units")
        self.assertEqual(normalize_emergency_numbers("give 911units"), "give 911units")

    def test_indian_emergency_numbers_untouched(self):
        """112 and 108 are untouched."""
        self.assertEqual(normalize_emergency_numbers("112"), "112")
        self.assertEqual(normalize_emergency_numbers("call 112"), "call 112")
        self.assertEqual(normalize_emergency_numbers("108"), "108")
        self.assertEqual(normalize_emergency_numbers("ambulance 108"), "ambulance 108")

    def test_empty_none_and_non_str_safe(self):
        """Empty string, None, and non-string types return safely unchanged."""
        self.assertEqual(normalize_emergency_numbers(""), "")
        self.assertIsNone(normalize_emergency_numbers(None))
        self.assertEqual(normalize_emergency_numbers(123), 123)
        self.assertFalse(normalize_emergency_numbers(False))

    def test_multiple_occurrences(self):
        """Multiple occurrences in a single string are all normalized."""
        self.assertEqual(
            normalize_emergency_numbers("call 911 or 999 immediately"),
            "call 112 or 112 immediately"
        )
        self.assertEqual(
            normalize_emergency_numbers("If 911 fails dial 999"),
            "If 112 fails dial 112"
        )

    def test_nested_treatment_dict_and_list(self):
        """Nested structures (dict, list) have all string fields normalized without mutating original."""
        original = {
            "home_care": ["Rest well", "If chest pain occurs, call 911 immediately."],
            "lifestyle_tips": ["Avoid stress", "Stay hydrated"],
            "what_to_avoid": ["Heavy lifting; dial 999 if pain worsens"],
            "escalation_timeline": "If symptoms persist for 2 days, consult GP or call 911.",
            "metadata": {
                "flag_code": 911,  # int non-str unchanged
                "notes": ["Emergency: 999"]
            }
        }
        import copy
        orig_copy = copy.deepcopy(original)

        cleaned = normalize_emergency_numbers(original)

        # Content assertions
        self.assertEqual(cleaned["home_care"][1], "If chest pain occurs, call 112 immediately.")
        self.assertEqual(cleaned["what_to_avoid"][0], "Heavy lifting; dial 112 if pain worsens")
        self.assertEqual(cleaned["escalation_timeline"], "If symptoms persist for 2 days, consult GP or call 112.")
        self.assertEqual(cleaned["metadata"]["flag_code"], 911)
        self.assertEqual(cleaned["metadata"]["notes"][0], "Emergency: 112")

        # Original unmutated
        self.assertEqual(original, orig_copy)

    def test_stream_buffer_chunk_split(self):
        """Simulated streaming where '9' and '11' arrive in separate chunks never shows 911 in rendered buffer."""
        chunks = [
            "Please ",
            "stay calm ",
            "and call ",
            "9",
            "11",
            " right away ",
            "for help."
        ]
        streamed_text = ""
        rendered_history = []

        for chunk in chunks:
            streamed_text += chunk
            # UI renders normalize_emergency_numbers(streamed_text)
            rendered = normalize_emergency_numbers(streamed_text)
            rendered_history.append(rendered)
            # The rendered buffer must never show standalone 911
            self.assertNotIn("911", rendered, f"Standalone 911 leaked in rendered output: '{rendered}'")

        self.assertEqual(rendered_history[-1], "Please stay calm and call 112 right away for help.")

    def test_careai_chat_sanitizes_return_fields_without_mutating_state(self):
        """CareAI.chat() applies normalization to message, treatment, red_flags, xai_explanation on return without mutating internal state."""
        import copy
        from bot.careai import CareAI
        bot = CareAI.__new__(CareAI)
        bot._is_emergency = lambda msg: False

        raw_pipeline_result = {
            "type": "answer",
            "message": "Please call 911 if pain escalates.",
            "treatment": {
                "home_care": ["Rest and call 999 if dizzy."],
                "escalation_timeline": "Call 911 in 24 hours.",
            },
            "red_flags": {
                "red_flags": ["Severe chest pain — call 911."],
                "emergency_threshold": "Dial 999 immediately.",
            },
            "xai_explanation": "Because symptom is acute, patient advised: call 911.",
        }

        # Mock internal state like history
        bot.history = [{"role": "user", "content": "help"}, {"role": "assistant", "content": "Please call 911 if pain escalates."}]
        bot._chat_pipeline = MagicMock(return_value=raw_pipeline_result)

        history_before = copy.deepcopy(bot.history)

        result = bot.chat("help")

        # Returned dictionary fields are normalized
        self.assertEqual(result["message"], "Please call 112 if pain escalates.")
        self.assertEqual(result["treatment"]["home_care"][0], "Rest and call 112 if dizzy.")
        self.assertEqual(result["treatment"]["escalation_timeline"], "Call 112 in 24 hours.")
        self.assertEqual(result["red_flags"]["red_flags"][0], "Severe chest pain — call 112.")
        self.assertEqual(result["red_flags"]["emergency_threshold"], "Dial 112 immediately.")
        self.assertEqual(result["xai_explanation"], "Because symptom is acute, patient advised: call 112.")

        # bot.history must NOT be mutated
        self.assertEqual(bot.history, history_before)
        self.assertEqual(bot.history[1]["content"], "Please call 911 if pain escalates.")


if __name__ == "__main__":
    unittest.main()
