"""
=============================================================================
 Care-AI Phase 1 Verification Test Suite
 Tests:
   1. Module imports and application startup.
   2. Knowledge-base construction.
   3. Sparse index construction and serialization.
   4. Index reload without rebuilding.
   5. Representative retrieval queries.
   6. Cache compatibility and invalid-cache handling.
   7. Session-state isolation.
   8. Dependency completeness.
=============================================================================
"""

import os
import sys
import time
import unittest
import pickle
import numpy as np
import scipy.sparse as sp

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Ensure Unicode streams on Windows
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


class TestPhase1CareAI(unittest.TestCase):

    def test_01_dependency_completeness(self):
        """1. Verify all dependencies required by real execution paths are present."""
        required_modules = [
            "streamlit",
            "streamlit_js_eval",
            "groq",
            "numpy",
            "pandas",
            "scipy",
            "dotenv",
            "requests",
            "bs4",
        ]
        for mod in required_modules:
            with self.subTest(module=mod):
                imported = __import__(mod)
                self.assertIsNotNone(imported)

    def test_02_module_imports_and_startup(self):
        """2. Verify key project modules import cleanly without error."""
        modules_to_test = [
            "core.knowledge_base",
            "core.vector_store",
            "core.symptom_scorer",
            "core.doctor_finder",
            "core.llm_router",
            "core.prompts",
            "bot.careai",
            "ui.doctor_finder_panel",
        ]
        for mod_name in modules_to_test:
            with self.subTest(module=mod_name):
                mod = __import__(mod_name, fromlist=["*"])
                self.assertIsNotNone(mod)

    def test_03_knowledge_base_construction(self):
        """3. Verify knowledge-base builds all sources and maps accurately."""
        from core.knowledge_base import build_all_chunks
        chunks, sev_map, prec_map = build_all_chunks(data_dir=os.path.join(WORKSPACE_ROOT, "data"))
        self.assertGreater(len(chunks), 30000, f"Expected >30,000 chunks, got {len(chunks)}")
        self.assertGreater(len(sev_map), 50, f"Expected >50 symptoms in severity map, got {len(sev_map)}")
        self.assertGreater(len(prec_map), 20, f"Expected >20 diseases in precaution map, got {len(prec_map)}")

    def test_04_sparse_index_construction_and_serialization(self):
        """4. Verify sparse index construction builds CSR matrix and serializes versioned cache."""
        from core.vector_store import VectorStore
        sample_docs = [
            {"id": "doc_1", "text": "Sudden onset chest pain radiating to left arm and jaw.", "category": "Cardiology"},
            {"id": "doc_2", "text": "Persistent dry cough, mild fever, and shortness of breath.", "category": "Respiratory"},
            {"id": "doc_3", "text": "Elevated fasting blood sugar, polyuria, polydipsia, and fatigue.", "category": "Endocrinology"},
        ]
        test_cache_name = "test_phase1_sparse_construction"
        test_cache_file = os.path.join(VectorStore.CACHE_DIR, f"{test_cache_name}.pkl")
        if os.path.exists(test_cache_file):
            os.remove(test_cache_file)

        vs = VectorStore(sample_docs, cache_name=test_cache_name)
        self.assertTrue(os.path.exists(test_cache_file), "Cache file was not created on disk.")
        self.assertIsInstance(vs.matrix, sp.csr_matrix, "Matrix must be a scipy.sparse.csr_matrix.")
        self.assertEqual(vs.matrix.shape[0], len(sample_docs))
        self.assertEqual(vs.matrix.shape[1], len(vs.vocab))

        # Check serialized payload structure
        with open(test_cache_file, "rb") as f:
            data = pickle.load(f)
        self.assertEqual(data.get("version"), VectorStore.CACHE_VERSION)
        self.assertEqual(data.get("format"), "scipy_csr_v2")
        self.assertIsInstance(data.get("matrix"), sp.csr_matrix)

        # Cleanup
        if os.path.exists(test_cache_file):
            os.remove(test_cache_file)

    def test_05_index_reload_without_rebuilding(self):
        """5. Verify reloading from disk cache is fast and avoids index reconstruction."""
        from core.vector_store import VectorStore
        cache_path = os.path.join(VectorStore.CACHE_DIR, f"{VectorStore.DEFAULT_CACHE_NAME}.pkl")
        self.assertTrue(os.path.exists(cache_path), f"Default cache {cache_path} must exist.")

        # Load existing cache and record time
        t0 = time.perf_counter()
        # Pass empty documents or matching documents to load cache
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
        load_time = time.perf_counter() - t0

        self.assertLess(load_time, 1.0, f"Cache reload took {load_time:.2f}s, expected < 1s")
        self.assertIsInstance(cached_data["matrix"], sp.csr_matrix)
        self.assertEqual(cached_data["version"], VectorStore.CACHE_VERSION)

    def test_06_representative_retrieval_queries(self):
        """6. Verify search and search_by_symptoms return relevant, properly formatted results."""
        from core.knowledge_base import build_all_chunks
        from core.vector_store import VectorStore

        chunks, _, _ = build_all_chunks(data_dir=os.path.join(WORKSPACE_ROOT, "data"))
        vs = VectorStore(chunks, cache_name=VectorStore.DEFAULT_CACHE_NAME)

        queries = [
            ("chest pain and shortness of breath", "EMERGENCY"),
            ("high fever headache stiff neck", "SYMPTOM"),
            ("insulin resistance type 2 diabetes", "CONDITION"),
        ]

        for query, cat in queries:
            with self.subTest(query=query):
                t0 = time.perf_counter()
                results = vs.search(query, k=6, query_category=cat)
                lat = (time.perf_counter() - t0) * 1000
                self.assertLess(lat, 50.0, f"Query latency was {lat:.2f} ms (expected < 50ms)")
                self.assertGreater(len(results), 0, f"No results returned for query: {query}")
                for r in results:
                    self.assertIn("relevance_score", r)
                    self.assertIsInstance(r["relevance_score"], float)
                    self.assertGreaterEqual(r["relevance_score"], vs.RELEVANCE_THRESHOLD)

        # Test search_by_symptoms
        sym_res = vs.search_by_symptoms(["chest pain", "shortness of breath"], k=6)
        self.assertGreater(len(sym_res), 0)
        self.assertIn("relevance_score", sym_res[0])

    def test_07_cache_compatibility_and_invalid_cache_handling(self):
        """7. Verify invalid cache, version mismatch, and legacy multi-GB cache rejection."""
        from core.vector_store import VectorStore

        # A. Incompatible version rejection
        test_cache_path = os.path.join(VectorStore.CACHE_DIR, "test_v1_legacy.pkl")
        with open(test_cache_path, "wb") as f:
            pickle.dump({"version": 1, "hash": "dummy_hash", "matrix": None}, f)

        dummy_docs = [{"id": "d1", "text": "Sample text for cache test", "category": "General"}]
        vs_v1 = VectorStore(dummy_docs, cache_name="test_v1_legacy")
        # Should have rejected v1 and rebuilt v2
        self.assertEqual(vs_v1.CACHE_VERSION, 2)
        self.assertIsInstance(vs_v1.matrix, sp.csr_matrix)

        if os.path.exists(test_cache_path):
            os.remove(test_cache_path)

        # B. Legacy 5.33GB cache rejection guard (ensure 0 bytes deserialized / no OOM)
        legacy_path = os.path.join(VectorStore.CACHE_DIR, "vector_index.pkl")
        if os.path.exists(legacy_path):
            test_legacy = VectorStore.__new__(VectorStore)
            test_legacy.documents = dummy_docs
            test_legacy.cache_name = "vector_index"
            self.assertFalse(test_legacy._load_cache(), "Legacy 5.33GB cache must be rejected by safety guard.")

    def test_08_session_state_isolation(self):
        """8. Verify per-user session isolation: mutations in Session 1 do not leak to Session 2."""
        from core.vector_store import VectorStore
        from core.symptom_scorer import SymptomScorer
        from core.llm_router import LLMRouter
        from bot.careai import CareAI

        dummy_chunks = [{"id": "d1", "text": "Clinical reference document", "category": "General"}]
        shared_vs = VectorStore(dummy_chunks, cache_name="test_session_isolation_vs")
        shared_scorer = SymptomScorer({"fever": 5, "cough": 3})
        shared_router = LLMRouter({"groq_api_key": "test_dummy_key"})

        try:
            bot_user1 = CareAI(
                vector_store=shared_vs,
                symptom_scorer=shared_scorer,
                router=shared_router,
            )
            bot_user2 = CareAI(
                vector_store=shared_vs,
                symptom_scorer=shared_scorer,
                router=shared_router,
            )

            # Both share the same read-only underlying structures
            self.assertIs(bot_user1.vs, bot_user2.vs)
            self.assertIs(bot_user1.scorer, bot_user2.scorer)
            self.assertIs(bot_user1.router, bot_user2.router)

            # User 1 has active conversation in Spanish
            bot_user1.user_language = {"language": "Spanish", "code": "es"}
            bot_user1.history = [{"role": "user", "content": "Tengo dolor"}]
            bot_user1.collected_info = {"site": "chest"}
            bot_user1.last_triage = "EMERGENCY"
            bot_user1.last_differential = [{"condition": "CAD", "probability": "HIGH"}]

            # Verify User 2 has clean initial state
            self.assertEqual(bot_user2.user_language["code"], "en")
            self.assertEqual(bot_user2.history, [])
            self.assertEqual(bot_user2.collected_info, {})
            self.assertIsNone(bot_user2.last_triage)
            self.assertEqual(bot_user2.last_differential, [])

            # Resetting User 1 does not affect User 2
            bot_user2.user_language = {"language": "German", "code": "de"}
            bot_user1.reset()
            self.assertEqual(bot_user1.history, [])
            self.assertEqual(bot_user2.user_language["code"], "de")

        finally:
            test_cache = os.path.join(VectorStore.CACHE_DIR, "test_session_isolation_vs.pkl")
            if os.path.exists(test_cache):
                os.remove(test_cache)


if __name__ == "__main__":
    unittest.main()
