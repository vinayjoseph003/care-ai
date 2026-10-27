# ==============================================================================
#  tests/test_retrieval_negation.py — Unit tests for Retrieval & Negation
#
#  Offline, deterministic test suite using stubs/mocks only.
#  Does NOT build or write the real vector cache.
#
#  Covers:
#    a. Negation boundaries (extract_symptoms_from_text):
#       - "no fever, chills, or swollen glands" -> none extracted
#       - "no vomiting and severe chest pain" -> "chest pain" IS extracted
#       - "no fever and chills" -> "chills" IS extracted
#       - "no fever but severe headache" -> headache extracted, fever not
#       - "fever, no chills" -> fever extracted, chills not
#       - "no fever and no chills" -> neither
#    b. Interleaved merge in _retrieve:
#       - stub vs.search_by_symptoms returns k distinct docs and vs.search returns
#         k other docs; _retrieve must return at most k docs that include BOTH
#         sources (interleaved), deduped by id.
#    c. ENT category override with full_query
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

from core.symptom_scorer import SymptomScorer
from core.vector_store import VectorStore
from bot.careai import CareAI


class TestRetrievalNegation(unittest.TestCase):

    def setUp(self):
        self.severity_map = {
            "fever": 3,
            "chills": 3,
            "headache": 2,
            "sore throat": 2,
            "difficulty swallowing": 3,
            "cough": 2,
            "chest pain": 5,
            "vomiting": 3,
            "swollen glands": 2,
        }
        self.scorer = SymptomScorer(self.severity_map)

    def test_extract_symptoms_negation(self):
        """
        Original failure case & basic negation.
        """
        full_conversation = (
            "i have sore throat and i can't swallow comfortably "
            "no fever, chills, or swollen glands"
        )
        extracted = self.scorer.extract_symptoms_from_text(full_conversation)
        self.assertNotIn("fever", extracted, "Negated 'fever' must not be extracted")
        self.assertNotIn("chills", extracted, "Negated 'chills' must not be extracted")
        self.assertIn("sore throat", extracted, "'sore throat' is not negated and must be extracted")

        both = self.scorer.extract_symptoms_from_text("fever and chills")
        self.assertIn("fever", both)
        self.assertIn("chills", both)

        without_chills = self.scorer.extract_symptoms_from_text("without chills, I have fever")
        self.assertIn("fever", without_chills)
        self.assertNotIn("chills", without_chills)

        headache_only = self.scorer.extract_symptoms_from_text("I have a severe headache today")
        self.assertIn("headache", headache_only)

    def test_negation_boundaries(self):
        """
        STEP 1 (a) Negation boundaries:
        - 'no fever, chills, or swollen glands' -> none of the three extracted (comma/'or' lists stay negated)
        - 'no vomiting and severe chest pain' -> 'chest pain' IS extracted
        - 'no fever and chills' -> 'chills' IS extracted (conservative: ambiguity counts as present)
        - 'no fever but severe headache' -> headache extracted, fever not
        - 'fever, no chills' -> fever extracted, chills not
        - 'no fever and no chills' -> neither
        """
        # 1. "no fever, chills, or swollen glands" -> none extracted
        res1 = self.scorer.extract_symptoms_from_text("no fever, chills, or swollen glands")
        self.assertNotIn("fever", res1)
        self.assertNotIn("chills", res1)
        self.assertNotIn("swollen glands", res1)

        # 2. "no vomiting and severe chest pain" -> "chest pain" IS extracted
        res2 = self.scorer.extract_symptoms_from_text("no vomiting and severe chest pain")
        self.assertIn("chest pain", res2, "chest pain after 'and' must be extracted")
        self.assertNotIn("vomiting", res2, "vomiting after 'no' must not be extracted")

        # 3. "no fever and chills" -> "chills" IS extracted (conservative: ambiguity counts as present)
        res3 = self.scorer.extract_symptoms_from_text("no fever and chills")
        self.assertIn("chills", res3, "chills after 'and' must be extracted conservatively")
        self.assertNotIn("fever", res3, "fever after 'no' must not be extracted")

        # 4. "no fever but severe headache" -> headache extracted, fever not
        res4 = self.scorer.extract_symptoms_from_text("no fever but severe headache")
        self.assertIn("headache", res4)
        self.assertNotIn("fever", res4)

        # 5. "fever, no chills" -> fever extracted, chills not
        res5 = self.scorer.extract_symptoms_from_text("fever, no chills")
        self.assertIn("fever", res5)
        self.assertNotIn("chills", res5)

        # 6. "no fever and no chills" -> neither
        res6 = self.scorer.extract_symptoms_from_text("no fever and no chills")
        self.assertNotIn("fever", res6)
        self.assertNotIn("chills", res6)

    def test_retrieve_runs_full_query_and_merges_with_symptoms(self):
        """
        _retrieve must:
        (1) run the full-query text search
        (2) when extracted symptoms are non-empty, still run it and merge/dedupe
            with the symptom search by chunk id, keeping top-k cap
        """
        stub_vs = MagicMock()
        mock_router = MagicMock()
        mock_scorer = MagicMock()

        mock_scorer.extract_symptoms_from_text.return_value = ["headache"]

        text_chunk_1 = {"id": "chunk_shared", "relevance_score": 0.8, "text": "Shared info"}
        text_chunk_2 = {"id": "chunk_text_only", "relevance_score": 0.7, "text": "Text only info"}
        sym_chunk_1 = {"id": "chunk_shared", "relevance_score": 0.85, "text": "Shared info"}
        sym_chunk_2 = {"id": "chunk_sym_only", "relevance_score": 0.75, "text": "Sym only info"}

        stub_vs.search.return_value = [text_chunk_1, text_chunk_2]
        stub_vs.search_by_symptoms.return_value = [sym_chunk_1, sym_chunk_2]

        bot = CareAI(vector_store=stub_vs, symptom_scorer=mock_scorer, router=mock_router)
        bot.history = [
            {"role": "user", "content": "i have sore throat and i can't swallow comfortably"}
        ]

        results = bot._retrieve(k=3, query_category="SYMPTOM")

        stub_vs.search.assert_called_once()
        search_query_arg = stub_vs.search.call_args[0][0]
        self.assertIn("sore throat", search_query_arg)

        stub_vs.search_by_symptoms.assert_called_once()
        self.assertEqual(stub_vs.search_by_symptoms.call_args[0][0], ["headache"])

        ids = [doc.get("id") for doc in results]
        self.assertEqual(len(ids), len(set(ids)), "Results must be deduped by chunk id")
        self.assertIn("chunk_shared", ids)
        self.assertIn("chunk_text_only", ids)
        self.assertIn("chunk_sym_only", ids)
        self.assertLessEqual(len(results), 3, "Results must obey the top-k cap")

    def test_retrieve_interleaves_sources(self):
        """
        STEP 1 (b) Merge interleaving:
        stub vs.search_by_symptoms returns k distinct docs and vs.search returns
        k other docs; _retrieve must return at most k docs that include BOTH
        sources (interleaved), deduped by id.
        """
        k = 4
        sym_docs = [{"id": f"sym_{i}", "text": f"Sym doc {i}"} for i in range(k)]
        text_docs = [{"id": f"text_{i}", "text": f"Text doc {i}"} for i in range(k)]

        stub_vs = MagicMock()
        stub_vs.search_by_symptoms.return_value = sym_docs
        stub_vs.search.return_value = text_docs

        mock_scorer = MagicMock()
        mock_scorer.extract_symptoms_from_text.return_value = ["fever"]
        mock_router = MagicMock()

        bot = CareAI(vector_store=stub_vs, symptom_scorer=mock_scorer, router=mock_router)
        bot.history = [{"role": "user", "content": "fever symptom assessment"}]

        results = bot._retrieve(k=k, query_category="SYMPTOM")

        self.assertLessEqual(len(results), k)
        res_ids = [d["id"] for d in results]
        sym_count = sum(1 for rid in res_ids if rid.startswith("sym_"))
        text_count = sum(1 for rid in res_ids if rid.startswith("text_"))
        self.assertGreater(sym_count, 0, "Must contain docs from search_by_symptoms")
        self.assertGreater(text_count, 0, "Must contain docs from search (text)")
        self.assertEqual(res_ids, ["sym_0", "text_0", "sym_1", "text_1"])

    def test_search_by_symptoms_ent_override_with_full_query(self):
        """
        search_by_symptoms evaluates ENT category override against full_query.
        """
        vs = VectorStore.__new__(VectorStore)
        vs.documents = [
            {
                "id": "ent_doc_1",
                "category": "ENT",
                "symptoms": ["headache"],
                "text": "Pharyngitis and throat infection clinical guidance",
            }
        ]
        vs.vocab = {"headache": 0}
        vs.idf = MagicMock()
        vs.search = MagicMock(return_value=[])

        blocked_results = vs.search_by_symptoms(["headache"], k=6, query_category="SYMPTOM")
        self.assertEqual(len(blocked_results), 0, "Without override, ENT category must be blocked")

        full_query = "i have sore throat and i can't swallow comfortably"
        unblocked_results = vs.search_by_symptoms(
            ["headache"], k=6, query_category="SYMPTOM", full_query=full_query
        )
        self.assertEqual(len(unblocked_results), 1, "With 'throat' in full_query, ENT must be unblocked")
        self.assertEqual(unblocked_results[0]["id"], "ent_doc_1")


if __name__ == "__main__":
    unittest.main()
