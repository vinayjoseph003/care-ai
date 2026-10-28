"""
Tests for UI low-confidence indicators:
- Sidebar top diagnosis label when low_confidence is True vs False / empty
- Reliability badge style and label when low_confidence is True vs False
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Snapshot environment before ui.app calls load_dotenv() so test isolation is preserved
_env_backup = dict(os.environ)

# Ensure offline mock-only tests can run without optional heavy dependencies (e.g. scipy)
for _mod in ("scipy", "scipy.sparse"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from ui.app import top_diagnosis_label, reliability_badge

# Restore environment
os.environ.clear()
os.environ.update(_env_backup)


class TestLowConfidenceUI(unittest.TestCase):
    def test_top_diagnosis_label_low_confidence_true(self):
        xai_logs = [{"low_confidence": True, "confidence_score": 75}]
        self.assertEqual(top_diagnosis_label(xai_logs, "Migraine"), "Not determined")

    def test_top_diagnosis_label_low_confidence_false(self):
        xai_logs = [{"low_confidence": False, "confidence_score": 85}]
        self.assertEqual(top_diagnosis_label(xai_logs, "Migraine"), "Migraine")

    def test_top_diagnosis_label_empty_xai_logs(self):
        self.assertEqual(top_diagnosis_label([], "Migraine"), "Migraine")
        self.assertEqual(top_diagnosis_label(None, "Migraine"), "Migraine")

    def test_reliability_badge_low_confidence_verified_low(self):
        icon, label, cls = reliability_badge("LOW", "VERIFIED", low_confidence=True)
        self.assertEqual(label, "Confirm with doctor")
        self.assertEqual(icon, "🟡")
        self.assertEqual(cls, "badge-medium")

    def test_reliability_badge_not_low_confidence_verified_low(self):
        icon, label, cls = reliability_badge("LOW", "VERIFIED", low_confidence=False)
        self.assertEqual(label, "Well-supported")
        self.assertEqual(icon, "✅")
        self.assertEqual(cls, "badge-low")

    def test_reliability_badge_high_risk_unchanged(self):
        icon, label, cls = reliability_badge("HIGH", "VERIFIED", low_confidence=True)
        self.assertEqual(label, "Verify with doctor")
        self.assertEqual(icon, "⚠️")
        self.assertEqual(cls, "badge-high")


if __name__ == "__main__":
    unittest.main()
