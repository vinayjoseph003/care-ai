"""
Tests for claim-level fact-checking default behavior in bot/careai.py.
Verifies fail-closed counting of supported claims:
- Only c.get("supported") is True (boolean) counts as supported.
- Missing key, None, strings ("true"/"false"), and non-dict entries are UNSUPPORTED.
- No crash on non-dict / None entries.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from bot.careai import CareAI


class TestFactcheckDefault(unittest.TestCase):

    def setUp(self):
        # Instantiate CareAI with mocks for stateless resources to avoid network/model calls
        self.bot = CareAI(
            vector_store=MagicMock(),
            symptom_scorer=MagicMock(),
            router=MagicMock(),
        )
        self.sample_answer = (
            "Migraine causes severe throbbing pain on one side of the head. "
            "It is often accompanied by nausea and light sensitivity. "
            "Resting in a dark quiet room can help relieve symptoms."
        )
        self.sample_context = "Migraine medical reference context."

    def test_all_supported_true_yields_verified_low(self):
        canned_fc = {
            "verdict": "UNVERIFIED",
            "claim_checks": [
                {"claim": "Claim 1", "supported": True},
                {"claim": "Claim 2", "supported": True},
                {"claim": "Claim 3", "supported": True},
            ],
        }
        self.bot._llm = MagicMock(return_value=json.dumps(canned_fc))

        res = self.bot._factcheck(self.sample_answer, self.sample_context)
        self.assertEqual(res["verdict"], "VERIFIED")
        self.assertEqual(res["hallucination_risk"], "LOW")

    def test_missing_supported_key_counts_as_unsupported_not_verified(self):
        # 1 of 3 missing "supported" key -> must count as unsupported (ratio = 1/3 <= 0.35 -> PARTIALLY_VERIFIED)
        canned_fc = {
            "verdict": "VERIFIED",
            "claim_checks": [
                {"claim": "Claim 1", "supported": True},
                {"claim": "Claim 2", "supported": True},
                {"claim": "Claim 3"},  # missing "supported"
            ],
        }
        self.bot._llm = MagicMock(return_value=json.dumps(canned_fc))

        res = self.bot._factcheck(self.sample_answer, self.sample_context)
        self.assertNotEqual(res["verdict"], "VERIFIED")
        self.assertEqual(res["verdict"], "PARTIALLY_VERIFIED")
        self.assertEqual(res["hallucination_risk"], "MEDIUM")

    def test_string_true_counts_as_unsupported(self):
        # "supported": "true" (string, not bool True) -> must count as unsupported
        canned_fc = {
            "verdict": "VERIFIED",
            "claim_checks": [
                {"claim": "Claim 1", "supported": True},
                {"claim": "Claim 2", "supported": True},
                {"claim": "Claim 3", "supported": "true"},  # string "true"
            ],
        }
        self.bot._llm = MagicMock(return_value=json.dumps(canned_fc))

        res = self.bot._factcheck(self.sample_answer, self.sample_context)
        self.assertNotEqual(res["verdict"], "VERIFIED")
        self.assertEqual(res["verdict"], "PARTIALLY_VERIFIED")
        self.assertEqual(res["hallucination_risk"], "MEDIUM")

    def test_none_or_nondict_entry_unsupported_no_crash(self):
        # Non-dict and None entries must be counted as unsupported without throwing exceptions
        canned_fc = {
            "verdict": "VERIFIED",
            "claim_checks": [
                {"claim": "Claim 1", "supported": True},
                None,
                "unexpected string entry",
            ],
        }
        self.bot._llm = MagicMock(return_value=json.dumps(canned_fc))

        # 1 of 3 supported -> 2 unsupported -> ratio = 2/3 > 0.35 -> UNVERIFIED / HIGH
        res = self.bot._factcheck(self.sample_answer, self.sample_context)
        self.assertEqual(res["verdict"], "UNVERIFIED")
        self.assertEqual(res["hallucination_risk"], "HIGH")

    def test_one_of_three_unsupported_yields_partially_verified(self):
        # Explicit boolean False
        canned_fc = {
            "verdict": "VERIFIED",
            "claim_checks": [
                {"claim": "Claim 1", "supported": True},
                {"claim": "Claim 2", "supported": True},
                {"claim": "Claim 3", "supported": False},
            ],
        }
        self.bot._llm = MagicMock(return_value=json.dumps(canned_fc))

        res = self.bot._factcheck(self.sample_answer, self.sample_context)
        self.assertEqual(res["verdict"], "PARTIALLY_VERIFIED")
        self.assertEqual(res["hallucination_risk"], "MEDIUM")

    def test_zero_claims_path_unchanged(self):
        # Short answer with < 4 words yields 0 claims
        short_answer = "Too short."
        self.bot._llm = MagicMock()

        res = self.bot._factcheck(short_answer, self.sample_context)
        self.assertEqual(res["verdict"], "UNVERIFIED")
        self.assertEqual(res["hallucination_risk"], "MEDIUM")
        self.assertEqual(res["claim_checks"], [])
        self.assertEqual(res["unsupported_claims"], ["No checkable claims found"])
        self.bot._llm.assert_not_called()

    def test_unparseable_reply_yields_unverified_high(self):
        # Invalid JSON returned by LLM
        self.bot._llm = MagicMock(return_value="Malformed { not json")

        res = self.bot._factcheck(self.sample_answer, self.sample_context)
        self.assertEqual(res["verdict"], "UNVERIFIED")
        self.assertEqual(res["hallucination_risk"], "HIGH")
        self.assertEqual(res["unsupported_claims"], ["Fact-check parse failed"])


if __name__ == "__main__":
    unittest.main()

