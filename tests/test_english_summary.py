# ==============================================================================
#  tests/test_english_summary.py — Unit tests for English Summary Bridge
#
#  Offline, deterministic test suite using stubs/mocks only.
#  Does NOT build or write the real vector cache, and makes no network calls.
#
#  Covers:
#    a. _check_followup stores "english_summary" from LLM JSON into
#       self.english_summary; missing or empty preserves previous value.
#    b. Bridge active (user_language != "en" or non-ASCII letters in user text):
#       _retrieve builds query from english_summary (+ English collected_info)
#       and passes it to vs.search and scorer.extract_symptoms_from_text.
#       With code "en" (and ASCII text) the query is unchanged.
#    c. _filter_zero_overlap_differentials: Hindi raw text + english_summary
#       "sore throat and difficulty swallowing since morning" + differential
#       supporting ["difficulty swallowing"] is NOT dropped.
#    d. _score_severity uses english_summary for symptom extraction when
#       bridge is active.
#    e. Bridge active but english_summary empty -> current behavior (no crash,
#       raw-text query).
#    f. english_summary is cleared by reset() and is covered by snapshot/rollback.
# ==============================================================================

import copy
import json
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

from bot.careai import CareAI, ConvState
from core.llm_router import AllProvidersFailedError


class TestEnglishSummaryBridge(unittest.TestCase):

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

    def test_a_check_followup_stores_and_preserves_english_summary(self):
        """
        _check_followup stores english_summary in self.english_summary.
        If the field is missing or empty, previous value is kept.
        """
        # 1. Initial LLM response with english_summary
        llm_payload_1 = {
            "has_enough_info": False,
            "diagnostic_confidence": 35,
            "english_summary": "User reports acute sore throat and painful swallowing since morning.",
            "collected_info": {"symptom_or_topic": "sore throat"},
            "followup_question": "Do you also have a fever?",
            "followup_reason": "associations",
            "no_new_info_count": 0,
        }
        self.bot._llm = MagicMock(return_value=json.dumps(llm_payload_1))
        self.bot.history = [{"role": "user", "content": "गले में दर्द है"}]

        res1 = self.bot._check_followup()
        self.assertEqual(
            getattr(self.bot, "english_summary", None),
            "User reports acute sore throat and painful swallowing since morning.",
        )

        # 2. Subsequent LLM response with empty/missing english_summary
        llm_payload_2 = {
            "has_enough_info": True,
            "diagnostic_confidence": 75,
            "collected_info": {"associations": "no fever"},
            "followup_question": None,
            "followup_reason": "",
            "no_new_info_count": 0,
        }
        self.bot._llm = MagicMock(return_value=json.dumps(llm_payload_2))
        res2 = self.bot._check_followup()
        self.assertEqual(
            getattr(self.bot, "english_summary", None),
            "User reports acute sore throat and painful swallowing since morning.",
            "Missing english_summary must preserve the previous value",
        )

        # 3. LLM response with empty string
        llm_payload_3 = copy.deepcopy(llm_payload_2)
        llm_payload_3["english_summary"] = "   "
        self.bot._llm = MagicMock(return_value=json.dumps(llm_payload_3))
        res3 = self.bot._check_followup()
        self.assertEqual(
            getattr(self.bot, "english_summary", None),
            "User reports acute sore throat and painful swallowing since morning.",
            "Empty whitespace english_summary must preserve the previous value",
        )

    def test_b_retrieve_builds_query_from_english_summary_when_bridge_active(self):
        """
        When bridge is active (code != 'en' or non-ASCII letters in user text),
        _retrieve builds query from english_summary (+ English collected_info).
        With code 'en' and ASCII text, the query is unchanged.
        """
        # Case 1: user_language code != 'en' (Hindi)
        self.bot.user_language = {"language": "Hindi", "code": "hi"}
        self.bot.english_summary = "persistent dry cough and chest tightness"
        self.bot.collected_info = {"site": "chest", "onset": "3 days"}
        self.bot.history = [
            {"role": "user", "content": "मुझे तीन दिन से खांसी और सीने में जकड़न है"}
        ]

        self.bot._retrieve(k=6)
        call_args = self.mock_vs.search.call_args
        self.assertIsNotNone(call_args, "vs.search should have been called")
        query_arg = call_args[0][0]
        self.assertIn("persistent dry cough and chest tightness", query_arg)
        self.assertIn("chest", query_arg)
        self.assertNotIn("मुझे", query_arg, "Hindi raw text should not be in query when bridge is active")

        # Scorer symptom extraction should also use this query
        scorer_calls = self.mock_scorer.extract_symptoms_from_text.call_args_list
        self.assertTrue(any("persistent dry cough" in str(c) for c in scorer_calls))

        # Case 2: user_language code == 'en', but user text contains non-ASCII letters
        self.mock_vs.search.reset_mock()
        self.mock_scorer.extract_symptoms_from_text.reset_mock()
        self.bot.user_language = {"language": "English", "code": "en"}
        self.bot.english_summary = "severe headache since yesterday"
        self.bot.collected_info = {}
        self.bot.history = [
            {"role": "user", "content": "मुझे कल से बहुत तेज सिरदर्द है"}
        ]

        self.bot._retrieve(k=6)
        query_arg = self.mock_vs.search.call_args[0][0]
        self.assertIn("severe headache since yesterday", query_arg)
        self.assertNotIn("सिरदर्द", query_arg)

        # Case 3: code == 'en' and ASCII text -> unchanged behavior (uses user history)
        self.mock_vs.search.reset_mock()
        self.bot.user_language = {"language": "English", "code": "en"}
        self.bot.english_summary = "mild headache"
        self.bot.collected_info = {}
        self.bot.history = [
            {"role": "user", "content": "I have had a mild headache all day"}
        ]

        self.bot._retrieve(k=6)
        query_arg = self.mock_vs.search.call_args[0][0]
        self.assertEqual(query_arg, "I have had a mild headache all day")

    def test_c_filter_zero_overlap_keeps_differentials_matching_english_summary(self):
        """
        Hindi raw text + english_summary + supporting_symptoms is NOT dropped
        by _filter_zero_overlap_differentials.
        """
        self.bot.user_language = {"language": "Hindi", "code": "hi"}
        self.bot.history = [
            {"role": "user", "content": "मुझे सुबह से गले में दर्द है और निगलने में परेशानी हो रही है"}
        ]
        self.bot.english_summary = "sore throat and difficulty swallowing since morning"
        self.bot.collected_info = {}

        differentials = [
            {"rank": 1, "condition": "Pharyngitis", "supporting_symptoms": ["sore throat"]},
            {"rank": 2, "condition": "Esophagitis", "supporting_symptoms": ["difficulty swallowing"]},
            {"rank": 3, "condition": "Typhoid", "supporting_symptoms": ["rose spots", "step-ladder fever"]},
        ]

        filtered = self.bot._filter_zero_overlap_differentials(differentials)
        conditions_remaining = [d["condition"] for d in filtered]

        self.assertIn("Pharyngitis", conditions_remaining, "Rank 1 is always kept")
        self.assertIn("Esophagitis", conditions_remaining, "Esophagitis matches 'difficulty swallowing' in english_summary")
        self.assertNotIn("Typhoid", conditions_remaining, "Typhoid has zero overlap with symptoms")
        self.assertEqual(self.bot.last_diff_dropped_count, 1)

    def test_d_score_severity_uses_english_summary_when_bridge_active(self):
        """
        _score_severity uses english_summary for symptom extraction when the bridge is active.
        """
        self.bot.user_language = {"language": "Hindi", "code": "hi"}
        self.bot.english_summary = "severe chest pain radiating to jaw"
        self.bot.history = [
            {"role": "user", "content": "मुझे सीने में बहुत तेज दर्द है जो जबड़े तक जा रहा है"}
        ]
        self.bot.collected_info = {}

        self.bot._score_severity()

        call_args = self.mock_scorer.extract_symptoms_from_text.call_args
        self.assertIsNotNone(call_args, "extract_symptoms_from_text should have been called")
        text_passed = call_args[0][0]
        self.assertIn(self.bot.english_summary, text_passed)

    def test_e_bridge_active_empty_english_summary_falls_back_safely(self):
        """
        Bridge active but english_summary empty -> current behavior (no crash, raw-text query).
        """
        self.bot.user_language = {"language": "Hindi", "code": "hi"}
        self.bot.english_summary = ""
        self.bot.history = [
            {"role": "user", "content": "मुझे बुखार और ठंड लग रही है"}
        ]
        self.bot.collected_info = {}

        # 1. _retrieve should not crash and should use raw text
        self.bot._retrieve(k=6)
        query_arg = self.mock_vs.search.call_args[0][0]
        self.assertIn("मुझे बुखार और ठंड लग रही है", query_arg)

        # 2. _score_severity should not crash
        self.mock_scorer.extract_symptoms_from_text.reset_mock()
        self.bot._score_severity()
        text_arg = self.mock_scorer.extract_symptoms_from_text.call_args[0][0]
        self.assertIn("मुझे बुखार और ठंड लग रही है", text_arg)

        # 3. _filter_zero_overlap_differentials should not crash
        differentials = [
            {"rank": 1, "condition": "Viral Fever", "supporting_symptoms": ["fever"]},
            {"rank": 2, "condition": "Typhoid", "supporting_symptoms": ["chills"]},
        ]
        res = self.bot._filter_zero_overlap_differentials(differentials)
        self.assertTrue(len(res) >= 1)

    def test_f_reset_clears_english_summary_and_snapshot_rollback_covers_it(self):
        """
        english_summary is cleared by reset() and is covered by the failure snapshot/rollback.
        """
        # 1. Cleared by reset()
        self.bot.english_summary = "Patient reports migraine with aura"
        self.bot.reset()
        self.assertEqual(
            getattr(self.bot, "english_summary", None),
            "",
            "reset() must clear english_summary",
        )

        # 2. Covered by snapshot rollback in chat()
        self.bot.english_summary = "Preserved summary from previous turn"
        self.bot._is_emergency = MagicMock(return_value=False)

        def mock_failing_pipeline(user_msg):
            self.bot.english_summary = "Corrupted summary during failed turn"
            raise AllProvidersFailedError("Simulated LLM network failure")

        self.bot._chat_pipeline = mock_failing_pipeline

        res = self.bot.chat("Hello")
        self.assertEqual(res.get("type"), "service_unavailable")
        self.assertEqual(
            self.bot.english_summary,
            "Preserved summary from previous turn",
            "Snapshot rollback must restore english_summary on failure",
        )

    def test_g_score_severity_includes_raw_user_text_when_bridge_active(self):
        """
        (g) bridge active (Hindi msg in history), later English user msg 'I also have chest pain',
        english_summary lacks it -> text passed to scorer.extract_symptoms_from_text contains 'chest pain'.
        """
        self.bot.user_language = {"language": "Hindi", "code": "hi"}
        self.bot.english_summary = "Patient reports persistent fever and cough"
        self.bot.history = [
            {"role": "user", "content": "मुझे बुखार है"},
            {"role": "assistant", "content": "Do you have any other symptoms?"},
            {"role": "user", "content": "I also have chest pain"},
        ]
        self.bot.collected_info = {}

        self.bot._score_severity()

        call_args = self.mock_scorer.extract_symptoms_from_text.call_args
        self.assertIsNotNone(call_args, "extract_symptoms_from_text should have been called")
        text_passed = call_args[0][0]
        self.assertIn("chest pain", text_passed)

    def test_h_english_summary_capped_and_ignores_schema_placeholder(self):
        """
        (h) english_summary stored from LLM JSON is capped at 300 chars
        and ignored if it contains 'Short 1-2 sentence'.
        """
        # 1. Cap at 300 chars
        long_summary = "Patient reports " + "a" * 350
        llm_payload_long = {
            "has_enough_info": True,
            "diagnostic_confidence": 70,
            "english_summary": long_summary,
            "collected_info": {},
            "followup_question": None,
            "followup_reason": "",
            "no_new_info_count": 0,
        }
        self.bot._llm = MagicMock(return_value=json.dumps(llm_payload_long))
        self.bot.history = [{"role": "user", "content": "headache"}]
        self.bot._check_followup()
        self.assertEqual(len(self.bot.english_summary), 300)
        self.assertEqual(self.bot.english_summary, long_summary[:300])

        # 2. Ignored if it contains "Short 1-2 sentence"
        self.bot.english_summary = "Previous valid summary"
        llm_payload_placeholder = {
            "has_enough_info": True,
            "diagnostic_confidence": 70,
            "english_summary": "Short 1-2 sentence English clinical summary of symptoms",
            "collected_info": {},
            "followup_question": None,
            "followup_reason": "",
            "no_new_info_count": 0,
        }
        self.bot._llm = MagicMock(return_value=json.dumps(llm_payload_placeholder))
        self.bot._check_followup()
        self.assertEqual(
            self.bot.english_summary,
            "Previous valid summary",
            "Prompt placeholder string must be ignored and previous summary preserved",
        )


if __name__ == "__main__":
    unittest.main()
