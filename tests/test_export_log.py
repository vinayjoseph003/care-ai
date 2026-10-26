# ==============================================================================
#  tests/test_export_log.py — Unit tests for In-Memory Session Log Export
#
#  Offline, deterministic test suite using mocks only.
#  Covers:
#    - export_log_json returns valid JSON equal to round-tripped xai_logs
#    - export_log_json does NOT call builtins.open (no disk write)
#    - export_log_json serializes sets and custom non-serializable objects via default=str
# ==============================================================================

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Ensure offline mock-only tests can run without optional heavy dependencies (e.g. scipy)
for _mod in ("scipy", "scipy.sparse"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

from bot.careai import CareAI


class TestExportLog(unittest.TestCase):
    def setUp(self):
        # Create a CareAI instance with mock dependencies to stay completely offline
        self.mock_router = MagicMock()
        self.mock_vs = MagicMock()
        self.mock_scorer = MagicMock()

        self.bot = CareAI(
            vector_store=self.mock_vs,
            symptom_scorer=self.mock_scorer,
            router=self.mock_router,
        )

        # Populate sample log entries with unicode
        self.sample_logs = [
            {
                "turn": 1,
                "user_message": "తీవ్రమైన తలనొప్పి మరియు జ్వరం",
                "language": {"language": "Telugu", "code": "te"},
                "socrates_data": {"site": "head", "onset": "2 days ago"},
                "severity": {"severity_level": "MEDIUM", "total_score": 12},
                "differential": [
                    {"rank": 1, "condition": "Migraine", "probability": "HIGH"}
                ],
                "triage": "SEE_DOCTOR",
                "specialist": "Neurologist",
                "final_answer": "దయచేసి న్యూరాలజిస్ట్‌ను సంప్రదించండి.",
            }
        ]
        self.bot.xai_logs = self.sample_logs

    def test_export_log_json_returns_valid_roundtripped_json(self):
        """(a) export_log_json returns valid JSON equal to the round-tripped xai_logs."""
        raw_json = self.bot.export_log_json()
        self.assertIsInstance(raw_json, str)
        parsed = json.loads(raw_json)
        self.assertEqual(parsed, self.bot.xai_logs)

    def test_export_log_json_does_not_call_open(self):
        """(b) export_log_json does not call builtins.open (in-memory only)."""
        with patch("builtins.open") as mock_open:
            result = self.bot.export_log_json()
            mock_open.assert_not_called()
            self.assertIsInstance(result, str)

    def test_export_log_json_handles_set_and_custom_object(self):
        """xai_logs containing a set and a custom object does NOT raise in export_log_json and returns valid JSON."""
        class CustomObj:
            def __str__(self):
                return "custom_object_instance"

        self.bot.xai_logs = [
            {
                "turn": 1,
                "symptoms_set": {"headache", "fever"},
                "custom_metadata": CustomObj(),
            }
        ]

        # Must not raise TypeError, must return valid JSON string
        raw_json = self.bot.export_log_json()
        self.assertIsInstance(raw_json, str)

        parsed = json.loads(raw_json)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["turn"], 1)
        self.assertEqual(parsed[0]["custom_metadata"], "custom_object_instance")
        self.assertIsInstance(parsed[0]["symptoms_set"], str)


if __name__ == "__main__":
    unittest.main()
