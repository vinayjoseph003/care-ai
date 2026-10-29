"""
Unit tests for knowledge-base load guard (safe_load).
Tests verify:
- Success returns (result, None)
- Generic Exception returns (None, LOAD_FAILED_MESSAGE)
- MemoryError returns (None, LOAD_FAILED_MESSAGE)
- LOAD_FAILED_MESSAGE contains '112'
- BaseExceptions (KeyboardInterrupt, SystemExit) propagate
- Log message contains exception type name but NOT the exception message text
"""

import logging
import os
import sys
import unittest

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ui.helpers import safe_load, LOAD_FAILED_MESSAGE


class TestLoadGuard(unittest.TestCase):

    def test_success_returns_result_and_none(self):
        def stub_loader():
            return {"chunks": [1, 2, 3], "status": "loaded"}

        res, err = safe_load(stub_loader)
        self.assertEqual(res, {"chunks": [1, 2, 3], "status": "loaded"})
        self.assertIsNone(err)

    def test_generic_exception_returns_none_and_message(self):
        def failing_loader():
            raise RuntimeError("Database load error")

        res, err = safe_load(failing_loader)
        self.assertIsNone(res)
        self.assertEqual(err, LOAD_FAILED_MESSAGE)

    def test_memory_error_returns_none_and_message(self):
        def oom_loader():
            raise MemoryError("Out of memory building sparse index")

        res, err = safe_load(oom_loader)
        self.assertIsNone(res)
        self.assertEqual(err, LOAD_FAILED_MESSAGE)

    def test_message_contains_112(self):
        self.assertIn("112", LOAD_FAILED_MESSAGE)
        expected = (
            "The knowledge base could not be loaded. Please refresh the page in a minute. "
            "If this is a medical emergency, call 112 now."
        )
        self.assertEqual(LOAD_FAILED_MESSAGE, expected)

    def test_base_exceptions_propagate(self):
        def ki_loader():
            raise KeyboardInterrupt()

        def se_loader():
            raise SystemExit(1)

        with self.assertRaises(KeyboardInterrupt):
            safe_load(ki_loader)

        with self.assertRaises(SystemExit):
            safe_load(se_loader)

    def test_log_line_contains_exception_type_not_message_text(self):
        secret_text = "SECRET_PATIENT_OR_KEY_DATA_98765"

        def leaking_loader():
            raise ValueError(secret_text)

        logger = logging.getLogger("ui.helpers")
        with self.assertLogs(logger, level="ERROR") as cm:
            res, err = safe_load(leaking_loader)

        self.assertIsNone(res)
        self.assertEqual(err, LOAD_FAILED_MESSAGE)

        combined_logs = " ".join(cm.output)
        self.assertIn("ValueError", combined_logs)
        self.assertNotIn(secret_text, combined_logs)


if __name__ == "__main__":
    unittest.main()

