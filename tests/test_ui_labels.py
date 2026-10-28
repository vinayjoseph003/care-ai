"""
tests/test_ui_labels.py — Tests for UI label formatting and honesty.

Verifies:
- format_specialist_heading("General Practitioner (GP)") does not end with a stray "s" and keeps "GP"
- format_specialist_heading("General Practitioner / Gp") keeps "GP" uppercase
- format_specialist_heading("ENT Specialist") keeps "ENT" uppercase
- format_specialist_heading handles empty string and None safely
- Welcome card in ui/app.py no longer contains "Tamil" or "தமிழ்"
- Doctor finder heading in ui/doctor_finder_panel.py does not append a stray "s"
- Doctor finder section title style uses sans-serif, #06695C, and no all-caps / wide letter spacing
"""

import os
import re
import unittest
from ui.helpers import format_specialist_heading


class TestUiLabels(unittest.TestCase):
    # ── 1. Specialist Heading Formatting ─────────────────────────

    def test_specialist_heading_no_stray_s_and_keeps_gp(self):
        """format_specialist_heading must not end with a stray 's' and must keep 'GP'."""
        res = format_specialist_heading("General Practitioner (GP)")
        self.assertEqual(res, "General Practitioner (GP)")
        self.assertFalse(res.endswith("s"))
        self.assertIn("GP", res)

        # Also strip if stray 's' was already attached
        res_attached = format_specialist_heading("General Practitioner (GP)s")
        self.assertEqual(res_attached, "General Practitioner (GP)")
        self.assertFalse(res_attached.endswith("s"))

    def test_specialist_heading_gp_uppercase(self):
        """'General Practitioner / Gp' must have 'GP' in uppercase."""
        res = format_specialist_heading("General Practitioner / Gp")
        self.assertEqual(res, "General Practitioner / GP")
        self.assertIn("GP", res)
        self.assertNotIn("Gp", res)

    def test_specialist_heading_ent_uppercase(self):
        """'ENT Specialist' must keep 'ENT' in uppercase."""
        res = format_specialist_heading("ENT Specialist")
        self.assertEqual(res, "ENT Specialist")
        self.assertIn("ENT", res)

        res_case = format_specialist_heading("Ent Specialist")
        self.assertEqual(res_case, "ENT Specialist")
        self.assertIn("ENT", res_case)

    def test_specialist_heading_empty_and_none_safe(self):
        """Empty string and None must return safely as empty string."""
        self.assertEqual(format_specialist_heading(""), "")
        self.assertEqual(format_specialist_heading(None), "")

    # ── 2. Welcome Card Honesty (No Tamil) ───────────────────────

    def test_welcome_card_no_tamil(self):
        """The welcome card markup in ui/app.py must not reference Tamil or தமிழ்."""
        app_path = os.path.join(os.path.dirname(__file__), "..", "ui", "app.py")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract empty-state welcome card block
        match = re.search(r'<div class="empty-state">.*?</div>\s*</div>', content, re.DOTALL)
        self.assertIsNotNone(match, "Could not find empty-state block in ui/app.py")
        welcome_card = match.group(0)

        self.assertNotIn("Tamil", welcome_card, "Welcome card text must not claim Tamil support")
        self.assertNotIn("தமிழ்", welcome_card, "Welcome card chips must not include Tamil chip")
        self.assertIn("Hindi, Telugu, or English", welcome_card)

    # ── 3. Doctor Finder Heading Template & CSS ──────────────────

    def test_doctor_finder_heading_no_stray_s(self):
        """ui/doctor_finder_panel.py must not hardcode 'Nearby {specialist}s'."""
        df_path = os.path.join(os.path.dirname(__file__), "..", "ui", "doctor_finder_panel.py")
        with open(df_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertNotIn("Nearby {specialist}s", content, "Stray 's' template found in doctor finder heading")

    def test_doctor_finder_heading_css_style(self):
        """df-section-title in doctor finder CSS must use sans-serif, #06695C, and no wide/all-caps spacing."""
        df_path = os.path.join(os.path.dirname(__file__), "..", "ui", "doctor_finder_panel.py")
        with open(df_path, "r", encoding="utf-8") as f:
            content = f.read()

        css_match = re.search(r'\.df-section-title\s*\{([^}]+)\}', content)
        self.assertIsNotNone(css_match, "Could not find .df-section-title rule in ui/doctor_finder_panel.py")
        rule = css_match.group(1)

        self.assertNotIn("'IBM Plex Mono'", rule)
        self.assertNotIn("0.14em", rule)
        self.assertNotIn("uppercase", rule)
        self.assertIn("#06695C", rule)


if __name__ == "__main__":
    unittest.main()

