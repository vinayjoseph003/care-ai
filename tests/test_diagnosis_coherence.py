# ==============================================================================
#  tests/test_diagnosis_coherence.py — Unit tests for Diagnosis Coherence
#
#  Offline, deterministic test suite using mocks only.
#  Covers:
#    - Preserving rank 1 differential even when supporting_symptoms lack literal overlap
#    - Dropping rank >= 2 differentials when zero-overlap
#    - Ensuring last_top_condition is the rank 1 condition, never an escalated/demoted rank 3
#    - Ensuring the condition string passed to _answer_streamed, _get_treatment_guidance,
#      and _get_red_flags is synchronized to the same top condition
#    - Recording diff_filter_dropped in xai_logs
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


class TestDiagnosisCoherence(unittest.TestCase):

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

    def test_pipeline_preserves_rank1_and_synchronizes_top_condition(self):
        """
        Mock reasoning with:
          - rank 1: 'mild viral illness' (MODERATE, supporting_symptoms: ['difficulty swallowing'])
          - rank 2: 'strep throat' (LOW, supporting_symptoms: ['odynophagia'])
          - rank 3: 'Typhoid' (LOW, supporting_symptoms: ['discomfort upon waking'])
        User text: 'i have sore throat and i can't swallow anything comfortably'
        collected_info contains: 'discomfort upon waking'

        Assert:
          - last_top_condition == 'mild viral illness'
          - 'Typhoid' is not the top condition
          - the condition string passed to _answer_streamed, _get_treatment_guidance,
            and _get_red_flags is the same one ('mild viral illness')
        """
        user_message = "i have sore throat and i can't swallow anything comfortably"
        self.bot.collected_info = {"onset": "discomfort upon waking"}

        self.bot._check_followup = MagicMock(return_value={
            "has_enough_info": True,
            "diagnostic_confidence": 75,
            "collected_info": {"onset": "discomfort upon waking"},
            "followup_question": None,
        })
        self.bot._classify_query = MagicMock(return_value="SYMPTOM")
        self.bot._retrieve = MagicMock(return_value=[])

        reasoning = {
            "triage_level": "SEE_DOCTOR",
            "differential_diagnosis": [
                {
                    "rank": 1,
                    "condition": "mild viral illness",
                    "probability": "MODERATE",
                    "supporting_symptoms": ["difficulty swallowing"],
                },
                {
                    "rank": 2,
                    "condition": "strep throat",
                    "probability": "LOW",
                    "supporting_symptoms": ["odynophagia"],
                },
                {
                    "rank": 3,
                    "condition": "Typhoid",
                    "probability": "LOW",
                    "supporting_symptoms": ["discomfort upon waking"],
                },
            ],
            "specialist": "General Practitioner",
            "confidence_score": 80,
            "conditions_agree": True,
        }
        self.bot._reason_consistent = MagicMock(return_value=reasoning)

        # Spies for guidance, red flags, and answer streaming
        self.bot._get_selfcare_guidance = MagicMock()
        self.bot._get_treatment_guidance = MagicMock(return_value={
            "home_care": ["Rest"],
            "escalation_timeline": "See doctor if worse",
        })
        self.bot._get_red_flags = MagicMock(return_value={"red_flags": []})
        self.bot._answer_streamed = MagicMock(return_value="Clinical advice for viral illness")
        self.bot._factcheck = MagicMock(return_value={
            "claim_checks": [],
            "hallucination_risk": "LOW",
            "verdict": "VERIFIED",
        })
        self.bot._xai_explain_streamed = MagicMock(return_value="XAI explanation")

        result = self.bot.chat(user_message)

        # 1. last_top_condition must remain the rank 1 diagnosis ("mild viral illness")
        self.assertEqual(self.bot.last_top_condition, "mild viral illness")
        self.assertNotEqual(self.bot.last_top_condition, "Typhoid")

        # 2. _get_treatment_guidance must receive "mild viral illness"
        self.bot._get_treatment_guidance.assert_called_once()
        tg_condition = self.bot._get_treatment_guidance.call_args[0][0]
        self.assertEqual(tg_condition, "mild viral illness")

        # 3. _get_red_flags must receive "mild viral illness"
        self.bot._get_red_flags.assert_called_once()
        rf_condition = self.bot._get_red_flags.call_args[0][0]
        self.assertEqual(rf_condition, "mild viral illness")

        # 4. _answer_streamed must receive "mild viral illness" as top_condition
        self.bot._answer_streamed.assert_called_once()
        answer_call = self.bot._answer_streamed.call_args
        answer_condition = (
            answer_call.kwargs.get("top_condition")
            if "top_condition" in answer_call.kwargs
            else (answer_call.args[3] if len(answer_call.args) > 3 else None)
        )
        self.assertEqual(answer_condition, "mild viral illness")
        self.assertEqual(tg_condition, answer_condition)
        self.assertEqual(rf_condition, answer_condition)

        # 5. xai_logs must record diff_filter_dropped count (strep throat dropped -> 1)
        last_log = self.bot.xai_logs[-1]
        self.assertEqual(last_log.get("diff_filter_dropped"), 1)

    def test_filter_zero_overlap_never_drops_rank1_and_drops_zero_overlap_rank2(self):
        """
        Direct test for _filter_zero_overlap_differentials:
        - Rank 1 has zero overlap with user text
        - Rank 2 has zero overlap with user text
        - Rank 3 has overlap with user text
        Result:
          - Rank 1 MUST be kept (index 0 is never dropped)
          - Rank 2 MUST be dropped
          - Rank 3 MUST be kept
          - Filtered list has length 2, starting with rank 1
        """
        self.bot.history = [{"role": "user", "content": "i have sore throat and i can't swallow anything comfortably"}]
        self.bot.collected_info = {"onset": "discomfort upon waking"}

        diffs = [
            {
                "rank": 1,
                "condition": "mild viral illness",
                "probability": "MODERATE",
                "supporting_symptoms": ["difficulty swallowing"],
            },
            {
                "rank": 2,
                "condition": "strep throat",
                "probability": "LOW",
                "supporting_symptoms": ["odynophagia"],
            },
            {
                "rank": 3,
                "condition": "Typhoid",
                "probability": "LOW",
                "supporting_symptoms": ["discomfort upon waking"],
            },
        ]

        filtered = self.bot._filter_zero_overlap_differentials(diffs)

        # In fixed code: rank 1 is kept, rank 2 is dropped, rank 3 is kept -> 2 items
        # In current code: rank 1 is dropped, rank 2 is dropped, only rank 3 is returned -> 1 item
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["condition"], "mild viral illness")
        self.assertEqual(filtered[1]["condition"], "Typhoid")


if __name__ == "__main__":
    unittest.main()
