"""
=============================================================================
 Care-AI Phase 3 Verification Test Suite: Hosted Staging Deployment
 Tests:
   1. Clean application startup & knowledge base initialization.
   2. Provider selection: LLM_PRIMARY_PROVIDER=gemini sets Gemini as primary.
   3. Primary provider live generation & latency check (Gemini).
   4. Automatic fallback when primary fails (Gemini -> OpenRouter).
   5. Knowledge-base retrieval & displayed sources format.
   6. English conversation turn.
   7. Telugu conversation turn (multilingual safety).
   8. Concurrent session isolation with shared resources.
   9. HTML injection defense (<script> and <img> escaping).
  10. Credential & stack trace leak prevention on technical failure.
=============================================================================
"""

import os
import sys
import html
import time
import unittest
from unittest.mock import patch, MagicMock

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

from dotenv import load_dotenv
load_dotenv(os.path.join(WORKSPACE_ROOT, ".env"))

from core.llm_router import LLMRouter
from bot.careai import CareAI, ConvState
from core.vector_store import VectorStore
from core.symptom_scorer import SymptomScorer
from core.knowledge_base import build_all_chunks
from ui.app import sanitize_html


class TestPhase3HostedDeployment(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.router_config = {
            "primary_provider":   os.environ.get("LLM_PRIMARY_PROVIDER", "gemini"),
            "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "openrouter_model":   os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
            "gemini_api_key":     os.environ.get("GEMINI_API_KEY", ""),
            "gemini_model":       os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            "groq_api_key":       os.environ.get("GROQ_API_KEY", ""),
            "groq_model":         os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "ollama_model":       os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
            "ollama_host":        os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        }
        # Shared knowledge resources (using production cache)
        all_chunks, severity_map, precaution_map = build_all_chunks(data_dir=os.path.join(WORKSPACE_ROOT, "data"))
        cls.chunks = all_chunks
        cls.severity_map = severity_map
        cls.precaution_map = precaution_map
        cls.vector_store = VectorStore(all_chunks)
        cls.symptom_scorer = SymptomScorer(severity_map)

    def test_01_clean_startup_knowledge_base(self):
        """1. Verify knowledge chunks, severity map, precaution map load accurately."""
        self.assertGreater(len(self.chunks), 30000)
        self.assertGreater(len(self.severity_map), 100)
        self.assertGreater(len(self.precaution_map), 30)
        self.assertIsNotNone(self.vector_store.matrix)

    def test_02_primary_provider_selection(self):
        """2. Verify LLM_PRIMARY_PROVIDER=gemini puts Gemini first in router priority chain."""
        router = LLMRouter(self.router_config)
        self.assertEqual(router.priority[0], "gemini")
        self.assertEqual(router.priority[1], "openrouter")
        status = router.status()
        self.assertEqual(status["primary"], "gemini")

    def test_03_primary_provider_generation(self):
        """3. Verify live response from selected primary provider (Gemini)."""
        if not self.router_config["gemini_api_key"]:
            self.skipTest("No GEMINI_API_KEY configured")

        router = LLMRouter(self.router_config)
        t0 = time.perf_counter()
        resp = router.complete(
            [{"role": "user", "content": "Reply with 'GEMINI_PRIMARY_OK' and nothing else."}],
            max_tokens=20,
            temp=0.0
        )
        latency = time.perf_counter() - t0
        self.assertIn("GEMINI_PRIMARY_OK", resp.strip())
        self.assertEqual(router.active_provider, "gemini")
        # Sub-second or low latency
        self.assertLess(latency, 5.0)

    def test_04_fallback_to_openrouter(self):
        """4. Verify automatic fallback to OpenRouter when Gemini fails."""
        if not (self.router_config["gemini_api_key"] and self.router_config["openrouter_api_key"]):
            self.skipTest("Requires both GEMINI_API_KEY and OPENROUTER_API_KEY")

        # Simulate Gemini failing with invalid credential
        cfg_fallback = dict(self.router_config)
        cfg_fallback["gemini_api_key"] = "AIzaSy-invalid-dummy-key"

        router = LLMRouter(cfg_fallback)
        self.assertEqual(router.priority[0], "gemini")

        resp = router.complete(
            [{"role": "user", "content": "Reply with 'OPENROUTER_FALLBACK_OK' and nothing else."}],
            max_tokens=25,
            temp=0.0
        )
        self.assertEqual(router.active_provider, "openrouter")
        self.assertIn("OPENROUTER_FALLBACK_OK", resp.strip())

    def test_05_knowledge_base_retrieval_and_sources(self):
        """5. Verify vector search returns relevant documents and formatted sources."""
        results = self.vector_store.search("chest pain radiating to arm shortness of breath", k=4)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("text", r)
            self.assertIn("source", r)
            self.assertTrue(len(r["text"]) > 0)

    def test_06_english_conversation_turn(self):
        """6. Verify English conversation produces valid structured clinical response."""
        router = LLMRouter(self.router_config)
        bot = CareAI(
            knowledge_chunks=self.chunks,
            severity_map=self.severity_map,
            precaution_map=self.precaution_map,
            vector_store=self.vector_store,
            symptom_scorer=self.symptom_scorer,
            router=router,
        )
        resp = bot.chat("I have a persistent sore throat and slight cough for 3 days, no fever.")
        self.assertIn("type", resp)
        self.assertTrue(len(resp["message"]) > 0)
        self.assertIn(resp["type"], ("followup", "answer"))

    def test_07_telugu_conversation_safety(self):
        """7. Verify Telugu input is handled smoothly without encoding failures."""
        router = LLMRouter(self.router_config)
        bot = CareAI(
            knowledge_chunks=self.chunks,
            severity_map=self.severity_map,
            precaution_map=self.precaution_map,
            vector_store=self.vector_store,
            symptom_scorer=self.symptom_scorer,
            router=router,
        )
        resp = bot.chat("నాకు కడుపు నొప్పి మరియు వాంతులు ఉన్నాయి")
        self.assertIn("type", resp)
        self.assertTrue(len(resp["message"]) > 0)

    def test_08_session_isolation(self):
        """8. Verify two concurrent sessions with shared resources do not leak state."""
        router = LLMRouter(self.router_config)
        bot1 = CareAI(
            knowledge_chunks=self.chunks,
            severity_map=self.severity_map,
            precaution_map=self.precaution_map,
            vector_store=self.vector_store,
            symptom_scorer=self.symptom_scorer,
            router=router,
        )
        bot2 = CareAI(
            knowledge_chunks=self.chunks,
            severity_map=self.severity_map,
            precaution_map=self.precaution_map,
            vector_store=self.vector_store,
            symptom_scorer=self.symptom_scorer,
            router=router,
        )

        bot1._add_turn("user", "Session 1 patient with severe migraine")
        bot1.collected_info["site"] = "head"

        self.assertEqual(len(bot2.history), 0)
        self.assertEqual(len(bot2.collected_info), 0)

    def test_09_html_injection_defense(self):
        """9. Verify HTML injection payloads are escaped and cannot execute script tags."""
        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert(1)>",
            "<a href='javascript:alert(1)'>click me</a>",
            "hello <b>world</b> <style>body{display:none}</style>",
        ]
        for payload in malicious_inputs:
            sanitized = sanitize_html(payload)
            self.assertNotIn("<script>", sanitized)
            self.assertNotIn("<img", sanitized)
            self.assertNotIn("<a ", sanitized)
            self.assertNotIn("<style>", sanitized)
            self.assertIn("&lt;", sanitized)
            self.assertIn("&gt;", sanitized)

    def test_10_error_handling_sanitization(self):
        """10. Verify complete provider exhaustion does not leak credentials or stack traces."""
        bad_config = {
            "openrouter_api_key": "sk-or-v1-PRIVATE_DEMO_KEY_123",
            "gemini_api_key": "AIzaSy-PRIVATE_DEMO_KEY_456",
            "groq_api_key": "gsk_PRIVATE_DEMO_KEY_789",
            "ollama_host": "http://127.0.0.1:9999",
        }
        router = LLMRouter(bad_config)
        try:
            router.complete([{"role": "user", "content": "hi"}], max_tokens=10)
            self.fail("Should have raised RuntimeError")
        except Exception as e:
            msg = str(e)
            self.assertNotIn("PRIVATE_DEMO_KEY_123", msg)
            self.assertNotIn("PRIVATE_DEMO_KEY_456", msg)
            self.assertNotIn("PRIVATE_DEMO_KEY_789", msg)
            self.assertIn("All LLM providers failed", msg)


if __name__ == "__main__":
    unittest.main()
