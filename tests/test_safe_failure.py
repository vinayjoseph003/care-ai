# ==============================================================================
#  tests/test_safe_failure.py — Unit tests for LLM Safe Failure Handling
#
#  Offline, deterministic test suite using mocks only.
#  Covers:
#    - Router raising AllProvidersFailedError -> chat() returns service_unavailable
#    - Router raising StreamInterruptedError -> chat() returns service_unavailable
#    - No "[ERROR]" string in any user-facing output
#    - History unchanged when provider failure occurs
#    - Red-flag emergency pre-check still works with a completely failing router
#    - Mid-stream failure raises StreamInterruptedError and does NOT fall through
#      to subsequent providers (no duplicate or mixed tokens)
# ==============================================================================

import contextlib
import copy
import io
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure offline mock-only tests can run without optional heavy dependencies (e.g. scipy)
for _mod in ("scipy", "scipy.sparse"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

from core.llm_router import (
    LLMRouter,
    AllProvidersFailedError,
    StreamInterruptedError,
)
from bot.careai import CareAI, SAFE_UNAVAILABLE_MESSAGE, _SHARED_RESOURCES


class TestSafeFailure(unittest.TestCase):
    def setUp(self):
        # Create a CareAI instance with mock dependencies to stay completely offline
        self.mock_router = MagicMock(spec=LLMRouter)
        self.mock_vs = MagicMock()
        self.mock_vs.search.return_value = []
        self.mock_vs.search_by_symptoms.return_value = []
        self.mock_scorer = MagicMock()
        self.mock_scorer.extract_symptoms_from_text.return_value = []
        self.mock_scorer.score.return_value = {}

        self.bot = CareAI(
            vector_store=self.mock_vs,
            symptom_scorer=self.mock_scorer,
            router=self.mock_router,
        )

    def test_complete_raising_returns_service_unavailable(self):
        """When router.complete raises AllProvidersFailedError, chat() returns service_unavailable."""
        self.mock_router.complete.side_effect = AllProvidersFailedError("All LLM providers failed.")

        result = self.bot.chat("I have a mild headache since yesterday")
        self.assertEqual(result.get("type"), "service_unavailable")
        self.assertEqual(result.get("message"), SAFE_UNAVAILABLE_MESSAGE)

    def test_stream_raising_returns_service_unavailable(self):
        """When router.stream raises AllProvidersFailedError, chat() returns service_unavailable."""
        # Max out follow-ups so chat() proceeds to streaming answer
        self.bot.followup_count = 5
        self.mock_router.complete.return_value = '{"language": "English", "code": "en", "triage_level": "SELF_CARE", "confidence_score": 70}'
        self.mock_router.stream.side_effect = AllProvidersFailedError("All LLM providers failed.")

        result = self.bot.chat("I have a mild headache")
        self.assertEqual(result.get("type"), "service_unavailable")
        self.assertEqual(result.get("message"), SAFE_UNAVAILABLE_MESSAGE)

    def test_stream_interrupted_returns_service_unavailable(self):
        """When router.stream raises StreamInterruptedError, chat() returns service_unavailable."""
        self.bot.followup_count = 5
        self.mock_router.complete.return_value = '{"language": "English", "code": "en", "triage_level": "SELF_CARE", "confidence_score": 70}'
        self.mock_router.stream.side_effect = StreamInterruptedError("Stream from openrouter interrupted")

        result = self.bot.chat("I have a mild headache")
        self.assertEqual(result.get("type"), "service_unavailable")
        self.assertEqual(result.get("message"), SAFE_UNAVAILABLE_MESSAGE)

    def test_stream_helper_does_not_swallow_errors(self):
        """_stream() must not swallow AllProvidersFailedError or StreamInterruptedError."""
        self.mock_router.stream.side_effect = AllProvidersFailedError("All failed")
        with self.assertRaises(AllProvidersFailedError):
            self.bot._stream([{"role": "user", "content": "hi"}])

        self.mock_router.stream.side_effect = StreamInterruptedError("Interrupted")
        with self.assertRaises(StreamInterruptedError):
            self.bot._stream([{"role": "user", "content": "hi"}])

    def test_no_error_prefix_in_output(self):
        """Ensure no '[ERROR]' text appears in any output or message."""
        self.mock_router.complete.side_effect = AllProvidersFailedError("All LLM providers failed.")

        result = self.bot.chat("I have a fever")
        self.assertNotIn("[ERROR]", result.get("message", ""))
        self.assertNotIn("[ERROR]", SAFE_UNAVAILABLE_MESSAGE)

    def test_history_unchanged_on_failure(self):
        """History must remain exactly unchanged when a provider failure occurs."""
        initial_history_len = len(self.bot.history)
        self.assertEqual(initial_history_len, 0)

        self.mock_router.complete.side_effect = AllProvidersFailedError("All LLM providers failed.")

        result = self.bot.chat("I have joint pain")
        self.assertEqual(result.get("type"), "service_unavailable")
        self.assertEqual(len(self.bot.history), initial_history_len)

        # Test history preservation when prior conversation turns exist
        self.bot.history = [
            {"role": "user", "content": "Hello doctor"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
        ]
        history_snapshot = list(self.bot.history)

        result2 = self.bot.chat("My knee is aching")
        self.assertEqual(result2.get("type"), "service_unavailable")
        self.assertEqual(self.bot.history, history_snapshot)

    def test_red_flag_emergency_still_works_with_failing_router(self):
        """Deterministic emergency check runs before any LLM call and succeeds even if router is broken."""
        self.mock_router.complete.side_effect = AllProvidersFailedError("Router is completely dead.")
        self.mock_router.stream.side_effect = AllProvidersFailedError("Router is completely dead.")

        emergency_messages = [
            "I have severe chest pain",
            "Help, I cannot breathe",
            "He is having a heart attack",
            "Thoughts of self-harm",
        ]
        for msg in emergency_messages:
            result = self.bot.chat(msg)
            self.assertEqual(result.get("type"), "emergency", f"Failed for emergency: {msg}")
            self.assertIn("MEDICAL EMERGENCY DETECTED", result.get("message", ""))
            # Ensure router was never called for this emergency pre-check
            self.mock_router.complete.assert_not_called()
            self.mock_router.stream.assert_not_called()

    def test_session_state_is_deepcopyable(self):
        """copy.deepcopy every non-shared attribute of a fresh CareAI and assert no exception."""
        bot = CareAI(
            vector_store=self.mock_vs,
            symptom_scorer=self.mock_scorer,
            router=self.mock_router,
        )
        non_shared = {
            k: v for k, v in vars(bot).items()
            if k not in _SHARED_RESOURCES
        }
        try:
            cloned = copy.deepcopy(non_shared)
        except Exception as e:
            self.fail(f"Failed to deepcopy session state attributes: {e}")
        self.assertEqual(len(cloned), len(non_shared))

    def test_failed_turn_restores_all_session_state(self):
        """Failure mid-turn restores all per-session attributes without state leaks."""
        class Script:
            def __init__(self):
                self.left = 10**9
                self.mode = "outage"

            def complete(self, messages, temp=0.2, max_tokens=900, **kw):
                if self.left <= 0:
                    raise AllProvidersFailedError("sim outage")
                self.left -= 1
                return "ok"

            def stream(self, messages, max_tokens=600, temp=0.3, **kw):
                if self.left <= 0:
                    if self.mode == "interrupt":
                        yield "partial "
                        raise StreamInterruptedError("sim interrupt")
                    raise AllProvidersFailedError("sim outage")
                self.left -= 1
                yield "ok "
                yield "answer."

        def make_bot():
            mock_router = MagicMock(spec=LLMRouter)
            script = Script()
            mock_router.complete.side_effect = script.complete
            mock_router.stream.side_effect = script.stream
            bot = CareAI(
                vector_store=self.mock_vs,
                symptom_scorer=self.mock_scorer,
                router=mock_router,
            )
            return bot, script

        def snapshot_session(o):
            return {
                k: copy.deepcopy(v)
                for k, v in vars(o).items()
                if k not in _SHARED_RESOURCES
            }

        for scenario, prior in (("fresh session", False), ("after one healthy turn", True)):
            for mode in ("outage", "interrupt"):
                for k in range(0, 26):
                    bot, script = make_bot()
                    if prior:
                        script.left = 10**9
                        with contextlib.redirect_stdout(io.StringIO()):
                            bot.chat("I have had a mild headache since this morning")

                    before = snapshot_session(bot)
                    script.left, script.mode = k, mode

                    with contextlib.redirect_stdout(io.StringIO()):
                        r = bot.chat("it is on the left side and feels throbbing")
                    r_type = r.get("type") if isinstance(r, dict) else None

                    if r_type != "service_unavailable":
                        # Turn completed normally; sweep done for this scenario & mode
                        break

                    with self.subTest(k=k, mode=mode, scenario=scenario):
                        self.assertEqual(r_type, "service_unavailable")
                        after = snapshot_session(bot)
                        self.assertEqual(before, after)


class TestRouterStreamBehavior(unittest.TestCase):
    """Test LLMRouter stream isolation and mid-stream failure behavior."""

    def test_stream_all_providers_failed_raises_exception(self):
        """When all providers fail, stream() raises AllProvidersFailedError and yields no '[ERROR]'."""
        router = LLMRouter({
            "openrouter_api_key": "fake_key",
            "gemini_api_key": "fake_key",
        })

        with patch("core.llm_router._call_openrouter_stream", side_effect=Exception("OpenRouter timeout")), \
             patch("core.llm_router._call_gemini_stream", side_effect=Exception("Gemini quota")):

            gen = router.stream([{"role": "user", "content": "Hello"}])
            with self.assertRaises(AllProvidersFailedError) as ctx:
                list(gen)

            self.assertIn("All LLM providers failed", str(ctx.exception))

    def test_mid_stream_failure_does_not_mix_providers(self):
        """If a provider fails mid-stream after tokens are yielded, do not fall through to another provider."""
        router = LLMRouter({
            "openrouter_api_key": "fake_key_or",
            "gemini_api_key": "fake_key_gemini",
        })

        # Generator that yields 2 tokens then raises an exception
        def failing_stream(*args, **kwargs):
            yield "Token1 "
            yield "Token2 "
            raise RuntimeError("Connection dropped mid-stream")

        def second_stream(*args, **kwargs):
            yield "Unexpected Token from fallback"

        with patch("core.llm_router._call_openrouter_stream", side_effect=failing_stream), \
             patch("core.llm_router._call_gemini_stream", side_effect=second_stream) as mock_gemini:

            tokens_received = []
            gen = router.stream([{"role": "user", "content": "Hello"}])

            with self.assertRaises(StreamInterruptedError):
                for token in gen:
                    tokens_received.append(token)

            # Verified: received initial tokens from the first provider
            self.assertEqual(tokens_received, ["Token1 ", "Token2 "])
            # Verified: did NOT fall through to the second provider
            mock_gemini.assert_not_called()


if __name__ == "__main__":
    unittest.main()
