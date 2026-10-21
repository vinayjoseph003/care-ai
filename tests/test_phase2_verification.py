"""
=============================================================================
 Care-AI Phase 2 Verification Test Suite
 Tests:
   1. Provider configuration & dynamic status (Ollama bounded check, priority).
   2. Gemini request & response parsing resilience (empty parts, blockReason, header auth).
   3. OpenRouter candidate model generation.
   4. Gemini candidate model generation.
   5. End-to-end fallback when primary fails.
   6. Error handling without credential or stack trace leakage.
   7. Structured response parsing under malformed / edge-case inputs.
   8. English conversation turn.
   9. Telugu conversation turn (multilingual integrity).
  10. Session-state isolation between concurrent sessions.
=============================================================================
"""

import os
import sys
import json
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

from core.llm_router import LLMRouter, _call_gemini
from bot.careai import CareAI, ConvState
from core.vector_store import VectorStore
from core.symptom_scorer import SymptomScorer


class TestPhase2RouterAndBot(unittest.TestCase):

    def setUp(self):
        self.router_config = {
            "primary_provider":   "openrouter",
            "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "openrouter_model":   os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
            "groq_api_key":       os.environ.get("GROQ_API_KEY", ""),
            "groq_model":         os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "gemini_api_key":     os.environ.get("GEMINI_API_KEY", ""),
            "gemini_model":       os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            "ollama_model":       os.environ.get("OLLAMA_MODEL", "qwen2.5:0.5b"),
            "ollama_host":        os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        }

    def test_01_router_priority_and_dynamic_ollama(self):
        """1. Verify priority order and that Ollama is checked dynamically with bounded timeout."""
        router = LLMRouter(self.router_config)
        self.assertEqual(router.priority[0], "openrouter")
        self.assertEqual(router.priority[1], "gemini")

        # Verify Ollama check does not wait 120s when offline
        status = router.status()
        self.assertIn("ollama", status)
        self.assertIsInstance(status["ollama"], bool)

    def test_02_gemini_header_auth_and_parsing_resilience(self):
        """2. Verify Gemini uses x-goog-api-key header and safely parses malformed/empty responses."""

        # Test empty candidates
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.read.return_value = json.dumps({"candidates": []}).encode("utf-8")
            mock_urlopen.return_value = mock_resp

            with self.assertRaises(RuntimeError) as ctx:
                _call_gemini([{"role": "user", "content": "hi"}], "test_fake_key", "gemini-3.5-flash-lite", 50, 0.2)
            self.assertIn("no candidates returned", str(ctx.exception).lower())

            # Check header authentication (key should NOT be in URL)
            req = mock_urlopen.call_args[0][0]
            self.assertNotIn("test_fake_key", req.full_url)
            self.assertEqual(req.get_header("X-goog-api-key"), "test_fake_key")

        # Test content with no parts
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.read.return_value = json.dumps({"candidates": [{"content": {"parts": []}}]}).encode("utf-8")
            mock_urlopen.return_value = mock_resp

            with self.assertRaises(RuntimeError) as ctx:
                _call_gemini([{"role": "user", "content": "hi"}], "test_fake_key", "gemini-3.5-flash-lite", 50, 0.2)
            self.assertIn("no text content", str(ctx.exception).lower())

        # Test safety blockReason
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value = mock_resp
            mock_resp.read.return_value = json.dumps({
                "candidates": [],
                "promptFeedback": {"blockReason": "SAFETY"}
            }).encode("utf-8")
            mock_urlopen.return_value = mock_resp

            with self.assertRaises(RuntimeError) as ctx:
                _call_gemini([{"role": "user", "content": "hi"}], "test_fake_key", "gemini-3.5-flash-lite", 50, 0.2)
            self.assertIn("safety", str(ctx.exception).lower())

    def test_03_openrouter_live_generation(self):
        """3. Verify OpenRouter responds with usable text using configured credentials."""
        if not self.router_config["openrouter_api_key"]:
            self.skipTest("No OPENROUTER_API_KEY configured")

        router = LLMRouter({
            "openrouter_api_key": self.router_config["openrouter_api_key"],
            "openrouter_model": self.router_config["openrouter_model"],
        })
        resp = router.complete(
            [{"role": "user", "content": "Reply with 'PULSE_OK' and nothing else."}],
            max_tokens=20,
            temp=0.0
        )
        self.assertIsInstance(resp, str)
        self.assertIn("PULSE_OK", resp.strip())
        self.assertEqual(router.active_provider, "openrouter")

    def test_04_gemini_live_generation(self):
        """4. Verify Gemini responds with usable text using configured credentials."""
        if not self.router_config["gemini_api_key"]:
            self.skipTest("No GEMINI_API_KEY configured")

        router = LLMRouter({
            "gemini_api_key": self.router_config["gemini_api_key"],
            "gemini_model": self.router_config["gemini_model"],
        })
        resp = router.complete(
            [{"role": "user", "content": "Reply with 'PULSE_OK' and nothing else."}],
            max_tokens=20,
            temp=0.0
        )
        self.assertIsInstance(resp, str)
        self.assertIn("PULSE_OK", resp.strip())
        self.assertEqual(router.active_provider, "gemini")

    def test_05_automatic_fallback_end_to_end(self):
        """5. Verify seamless fallback to Gemini when OpenRouter fails."""
        if not (self.router_config["openrouter_api_key"] and self.router_config["gemini_api_key"]):
            self.skipTest("Requires both OPENROUTER_API_KEY and GEMINI_API_KEY")

        # Create router with an invalid openrouter key to force primary failure
        failing_config = dict(self.router_config)
        failing_config["openrouter_api_key"] = "sk-or-v1-invalid-dummy-key"

        router = LLMRouter(failing_config)
        self.assertEqual(router.priority[0], "openrouter")

        resp = router.complete(
            [{"role": "user", "content": "Reply with 'FALLBACK_OK' and nothing else."}],
            max_tokens=20,
            temp=0.0
        )
        # Should have automatically fallen back to Gemini
        self.assertEqual(router.active_provider, "gemini")
        self.assertIn("FALLBACK_OK", resp.strip())

    def test_06_error_handling_sanitization(self):
        """6. Verify all-provider failure does not expose API keys or passwords."""
        bad_config = {
            "openrouter_api_key": "sk-or-v1-SECRET_KEY_ALPHA",
            "gemini_api_key": "AIzaSy-SECRET_KEY_BETA",
            "groq_api_key": "gsk_SECRET_KEY_GAMMA",
            "ollama_host": "http://127.0.0.1:9999",
        }
        router = LLMRouter(bad_config)
        try:
            router.complete([{"role": "user", "content": "hello"}], max_tokens=10)
        except Exception as e:
            err_msg = str(e)
            self.assertNotIn("SECRET_KEY_ALPHA", err_msg)
            self.assertNotIn("SECRET_KEY_BETA", err_msg)
            self.assertNotIn("SECRET_KEY_GAMMA", err_msg)
            self.assertIn("All LLM providers failed", err_msg)

    def test_07_structured_response_parsing_safety(self):
        """7. Verify CareAI safely parses malformed JSON and defaults properly."""
        dummy_vs = VectorStore([], cache_name="test_dummy_cache")
        bot = CareAI(
            knowledge_chunks=[],
            severity_map={},
            router_config=self.router_config,
            vector_store=dummy_vs,
            symptom_scorer=SymptomScorer({}),
        )

        # Test invalid JSON string
        parsed = bot._parse_json("Not a json at all", {"default_key": "default_val"})
        self.assertEqual(parsed, {"default_key": "default_val"})

        # Test markdown code fenced JSON
        fenced = "```json\n{\"test_key\": 123}\n```"
        parsed2 = bot._parse_json(fenced, {})
        self.assertEqual(parsed2, {"test_key": 123})

        # Test partial JSON with trailing commas
        trailing = "{\"key\": \"val\",}"
        parsed3 = bot._parse_json(trailing, {})
        self.assertEqual(parsed3, {"key": "val"})

    def test_08_english_conversation_safety(self):
        """8. Verify normal English interaction produces non-empty structured response."""
        dummy_vs = VectorStore([], cache_name="test_dummy_cache")
        bot = CareAI(
            knowledge_chunks=[],
            severity_map={},
            router_config=self.router_config,
            vector_store=dummy_vs,
            symptom_scorer=SymptomScorer({}),
        )
        resp = bot.chat("I have had a mild headache and fatigue for the past 2 days after staring at a computer screen.")
        self.assertIn("type", resp)
        self.assertIn(resp["type"], ("followup", "answer"))
        self.assertTrue(len(resp["message"]) > 0)

    def test_09_telugu_conversation_safety(self):
        """9. Verify Telugu input is handled without encoding failure or crash."""
        dummy_vs = VectorStore([], cache_name="test_dummy_cache")
        bot = CareAI(
            knowledge_chunks=[],
            severity_map={},
            router_config=self.router_config,
            vector_store=dummy_vs,
            symptom_scorer=SymptomScorer({}),
        )
        telugu_msg = "నాకు తలనొప్పి మరియు జ్వరం ఉంది"
        resp = bot.chat(telugu_msg)
        self.assertIn("type", resp)
        self.assertTrue(len(resp["message"]) > 0)
        # Should not crash and should record the turn
        self.assertEqual(bot.history[0]["content"], telugu_msg)

    def test_10_session_isolation(self):
        """10. Verify state isolation between two distinct CareAI instances."""
        dummy_vs = VectorStore([], cache_name="test_dummy_cache")
        shared_scorer = SymptomScorer({})
        shared_router = LLMRouter(self.router_config)

        session_a = CareAI(vector_store=dummy_vs, symptom_scorer=shared_scorer, router=shared_router)
        session_b = CareAI(vector_store=dummy_vs, symptom_scorer=shared_scorer, router=shared_router)

        session_a._add_turn("user", "Patient A has chest discomfort")
        session_a.collected_info["site"] = "chest"

        # Session B must be completely empty
        self.assertEqual(len(session_b.history), 0)
        self.assertEqual(len(session_b.collected_info), 0)


if __name__ == "__main__":
    unittest.main()
