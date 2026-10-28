"""
tests/test_english_only_retrieval.py — Tests for English-only chunk filtering in retrieval.

Verifies:
- is_english_chunk excludes Spanish by URL (/spanish/) or language field ("Spanish", "es")
- is_english_chunk keeps English MedlinePlus, MedQuAD, NHS, ICD-10, and CSV chunks
- Relative order of non-Spanish chunks is preserved
- Over-fetching fills top-k when enough English chunks exist
- Empty-after-filter returns empty list and does not crash
"""

import sys
import unittest
from unittest.mock import MagicMock

# Mock scipy if not present in environment
for _mod in ("scipy", "scipy.sparse"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from bot.careai import CareAI, is_english_chunk


class TestEnglishOnlyRetrieval(unittest.TestCase):
    def setUp(self):
        self.spanish_url_chunk = {
            "id": "mplus_560",
            "title": "Dolor de garganta",
            "source": "MedlinePlus (NIH)",
            "url": "https://medlineplus.gov/spanish/sorethroat.html",
            "text": "Health Topic: Dolor de garganta.",
        }
        self.spanish_lang_chunk = {
            "id": "mplus_999",
            "title": "Faringitis",
            "source": "MedlinePlus (NIH)",
            "language": "Spanish",
            "text": "Health Topic: Faringitis.",
        }
        self.spanish_es_lang_chunk = {
            "id": "mplus_998",
            "title": "Gripe",
            "source": "MedlinePlus (NIH)",
            "language": "es",
            "text": "Health Topic: Gripe.",
        }
        self.english_mplus_chunk = {
            "id": "mplus_1714",
            "title": "Sore Throat",
            "source": "MedlinePlus (NIH)",
            "url": "https://medlineplus.gov/sorethroat.html",
            "language": "English",
            "text": "Health Topic: Sore Throat.",
        }
        self.medquad_chunk = {
            "id": "medquad_100",
            "category": "General Medicine",
            "source": "MedQuAD — MedicalQuestionAnswering.csv",
            "question": "What causes sore throat?",
            "text": "Question: What causes sore throat?",
        }
        self.nhs_chunk = {
            "id": "nhs_200",
            "title": "Sore throat",
            "source": "NHS",
            "url": "https://www.nhs.uk/conditions/sore-throat/",
            "text": "Sore throats are very common and usually nothing to worry about.",
        }
        self.icd10_chunk = {
            "id": "icd10_J02",
            "title": "Acute pharyngitis",
            "source": "ICD-10-CM 2026 (CMS)",
            "text": "ICD-10 codes starting with J02.",
        }
        self.csv_chunk = {
            "id": "common_cold",
            "disease": "Common Cold",
            "source": "Mendeley Medical Dataset",
            "text": "Common Cold symptoms: throat irritation, cough.",
        }

    # ── 1. Pure function tests ─────────────────────────────────

    def test_exclude_spanish_by_url(self):
        """Chunks with /spanish/ in URL must be excluded."""
        self.assertFalse(is_english_chunk(self.spanish_url_chunk))
        chunk_upper = dict(self.spanish_url_chunk, url="https://medlineplus.gov/SPANISH/sorethroat.html")
        self.assertFalse(is_english_chunk(chunk_upper))

    def test_exclude_spanish_by_language_field(self):
        """Chunks with language='Spanish' or 'es' must be excluded."""
        self.assertFalse(is_english_chunk(self.spanish_lang_chunk))
        self.assertFalse(is_english_chunk(self.spanish_es_lang_chunk))
        chunk_case = dict(self.spanish_lang_chunk, language="spanish")
        self.assertFalse(is_english_chunk(chunk_case))

    def test_include_english_sources(self):
        """English MedlinePlus, MedQuAD, NHS, ICD-10, and CSV chunks must be included."""
        self.assertTrue(is_english_chunk(self.english_mplus_chunk))
        self.assertTrue(is_english_chunk(self.medquad_chunk))
        self.assertTrue(is_english_chunk(self.nhs_chunk))
        self.assertTrue(is_english_chunk(self.icd10_chunk))
        self.assertTrue(is_english_chunk(self.csv_chunk))

    def test_filter_removes_spanish_and_preserves_order(self):
        """Filtering a mixed list drops Spanish chunks and preserves exact relative order."""
        chunks = [
            self.medquad_chunk,
            self.spanish_url_chunk,
            self.nhs_chunk,
            self.spanish_lang_chunk,
            self.english_mplus_chunk,
            self.icd10_chunk,
            self.spanish_es_lang_chunk,
            self.csv_chunk,
        ]
        filtered = [c for c in chunks if is_english_chunk(c)]
        expected = [
            self.medquad_chunk,
            self.nhs_chunk,
            self.english_mplus_chunk,
            self.icd10_chunk,
            self.csv_chunk,
        ]
        self.assertEqual(filtered, expected)

    # ── 2. Behavioural / _retrieve pipeline tests ─────────────

    def test_retrieve_overfetch_fills_top_k(self):
        """_retrieve over-fetches so top-k (k=4) is filled with English chunks even when Spanish chunks rank first."""
        bot = CareAI.__new__(CareAI)
        bot.collected_info = {}
        bot.history = [{"role": "user", "content": "sore throat for two days"}]
        bot._is_bridge_active = lambda text="": False
        bot.english_summary = None
        bot.scorer = MagicMock()
        bot.scorer.extract_symptoms_from_text.return_value = []

        # VectorStore returns 3 Spanish chunks followed by 5 English chunks
        search_results = [
            dict(self.spanish_url_chunk, id="es_1"),
            dict(self.spanish_lang_chunk, id="es_2"),
            dict(self.spanish_es_lang_chunk, id="es_3"),
            dict(self.english_mplus_chunk, id="en_1"),
            dict(self.nhs_chunk, id="en_2"),
            dict(self.medquad_chunk, id="en_3"),
            dict(self.icd10_chunk, id="en_4"),
            dict(self.csv_chunk, id="en_5"),
        ]

        bot.vs = MagicMock()
        bot.vs.search.side_effect = lambda query, k, query_category: search_results[:k]

        # Request k=4; should skip first 3 Spanish and return top 4 English chunks
        results = bot._retrieve(k=4, query_category="SYMPTOM")
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertTrue(is_english_chunk(r), f"Spanish chunk leaked into results: {r}")
        self.assertEqual([r["id"] for r in results], ["en_1", "en_2", "en_3", "en_4"])

    def test_retrieve_empty_after_filter_does_not_crash(self):
        """When all retrieved chunks are Spanish, _retrieve returns [] and _fmt_context handles it."""
        bot = CareAI.__new__(CareAI)
        bot.collected_info = {}
        bot.history = [{"role": "user", "content": "sore throat"}]
        bot._is_bridge_active = lambda text="": False
        bot.english_summary = None
        bot.scorer = MagicMock()
        bot.scorer.extract_symptoms_from_text.return_value = []

        all_spanish = [
            dict(self.spanish_url_chunk, id="es_1"),
            dict(self.spanish_lang_chunk, id="es_2"),
            dict(self.spanish_es_lang_chunk, id="es_3"),
        ]
        bot.vs = MagicMock()
        bot.vs.search.side_effect = lambda query, k, query_category: all_spanish[:k]

        results = bot._retrieve(k=3, query_category="SYMPTOM")
        self.assertEqual(results, [])
        ctx = bot._fmt_context(results)
        self.assertEqual(ctx, "No relevant context found in the knowledge base.")


if __name__ == "__main__":
    unittest.main()

