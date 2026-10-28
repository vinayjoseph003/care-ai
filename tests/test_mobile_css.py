"""
Tests for Mobile & Sidebar CSS rules:
- stExpandSidebarButton visibility and removal of obsolete collapsedControl
- set_page_config initial_sidebar_state="auto"
- Responsive @media queries for .block-container, .treatment-grid, and .df-card-grid
- Sidebar privacy notice
"""

import os
import re
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_PY_PATH = os.path.join(PROJECT_ROOT, "ui", "app.py")
DOCTOR_FINDER_PATH = os.path.join(PROJECT_ROOT, "ui", "doctor_finder_panel.py")


class TestMobileCSS(unittest.TestCase):
    def setUp(self):
        with open(APP_PY_PATH, "r", encoding="utf-8") as f:
            self.app_content = f.read()
        with open(DOCTOR_FINDER_PATH, "r", encoding="utf-8") as f:
            self.df_content = f.read()

    def test_a_expand_sidebar_button_visibility(self):
        """(a) CSS contains 'stExpandSidebarButton' with 'visibility: visible'."""
        self.assertIn("stExpandSidebarButton", self.app_content)
        pattern = r'\[data-testid="stExpandSidebarButton"\][^{]*\{[^}]*visibility\s*:\s*visible'
        self.assertTrue(
            re.search(pattern, self.app_content, re.IGNORECASE),
            "Expected [data-testid=\"stExpandSidebarButton\"] to have 'visibility: visible'."
        )

    def test_b_no_collapsed_control(self):
        """(b) CSS no longer contains the obsolete 'collapsedControl' rule."""
        self.assertNotIn(
            "collapsedControl",
            self.app_content,
            "Found obsolete 'collapsedControl' selector in ui/app.py."
        )

    def test_c_initial_sidebar_state_auto(self):
        """(c) set_page_config uses initial_sidebar_state='auto'."""
        self.assertIn(
            'initial_sidebar_state="auto"',
            self.app_content,
            "Expected set_page_config to specify initial_sidebar_state=\"auto\"."
        )

    def test_d_media_768_block_container_padding(self):
        """(d) app.py contains @media (max-width: 768px) with .block-container padding reduced (<=1rem horizontal) and bottom >= 6rem."""
        self.assertIn("@media (max-width: 768px)", self.app_content)
        # Search within @media (max-width: 768px) block for .block-container padding
        media_match = re.search(r'@media\s*\(max-width:\s*768px\)\s*\{(.*?)\n\}', self.app_content, re.DOTALL)
        self.assertIsNotNone(media_match, "Could not find @media (max-width: 768px) block.")
        block_match = re.search(r'\.block-container\s*\{[^}]*padding\s*:\s*([^;]+);', media_match.group(1))
        self.assertIsNotNone(block_match, "Could not find .block-container padding in 768px media block.")
        padding_val = block_match.group(1)
        # Parse padding values (e.g. '0.8rem 1rem 6rem' or '0.8rem 1rem 6rem !important')
        parts = [p.replace('!important', '').strip() for p in padding_val.split()]
        self.assertGreaterEqual(len(parts), 3, f"Unexpected padding format: {padding_val}")
        # Horizontal padding is parts[1], bottom is parts[2]
        h_pad = float(re.search(r'([0-9.]+)', parts[1]).group(1))
        b_pad = float(re.search(r'([0-9.]+)', parts[2]).group(1))
        self.assertLessEqual(h_pad, 1.0, f"Horizontal padding {h_pad}rem is > 1rem")
        self.assertGreaterEqual(b_pad, 6.0, f"Bottom padding {b_pad}rem is < 6rem")

    def test_e_grid_collapse(self):
        """(e) .treatment-grid collapses to 1fr and .df-card-grid to 1fr at max-width 480px."""
        # .treatment-grid in ui/app.py collapses to 1fr (either at 768px or 480px)
        treatment_collapse = re.search(
            r'@media[^{]*max-width:\s*(?:768|480)px[^{]*\{.*?\.treatment-grid\s*\{[^}]*grid-template-columns\s*:\s*1fr',
            self.app_content,
            re.DOTALL
        )
        self.assertIsNotNone(treatment_collapse, ".treatment-grid must collapse to 1fr in responsive media block.")

        # .df-card-grid in ui/doctor_finder_panel.py collapses to 1fr at max-width 480px
        df_collapse = re.search(
            r'@media[^{]*max-width:\s*480px[^{]*\{.*?\.df-card-grid\s*\{[^}]*grid-template-columns\s*:\s*1fr',
            self.df_content,
            re.DOTALL
        )
        self.assertIsNotNone(df_collapse, ".df-card-grid must collapse to 1fr at max-width 480px in ui/doctor_finder_panel.py.")

    def test_f_sidebar_privacy_notice(self):
        """(f) a sidebar notice string exists: 'Refreshing the page clears your chat (privacy).'"""
        self.assertIn("Refreshing the page clears your chat (privacy).", self.app_content)

    def test_g_sidebar_toggle_pinned_and_styled(self):
        """(g) Both expand and collapse sidebar buttons are pinned fixed at top: 12px, left: 12px, 36px size with icon color #061A3A."""
        pattern_pinned = (
            r'\[data-testid="stExpandSidebarButton"\],\s*'
            r'\[data-testid="stSidebarCollapseButton"\]\s*\{'
            r'[^}]*position\s*:\s*fixed\s*!important;'
            r'[^}]*top\s*:\s*12px\s*!important;'
            r'[^}]*left\s*:\s*12px\s*!important;'
        )
        self.assertTrue(
            re.search(pattern_pinned, self.app_content, re.DOTALL),
            "Expected both stExpandSidebarButton and stSidebarCollapseButton to be pinned with position: fixed, top: 12px, left: 12px."
        )
        pattern_size = (
            r'\[data-testid="stExpandSidebarButton"\],\s*'
            r'\[data-testid="stSidebarCollapseButton"\]\s*\{'
            r'[^}]*width\s*:\s*36px\s*!important;'
            r'[^}]*height\s*:\s*36px\s*!important;'
        )
        self.assertTrue(
            re.search(pattern_size, self.app_content, re.DOTALL),
            "Expected 36px width and height for toggle buttons."
        )
        pattern_icon = (
            r'\[data-testid="stExpandSidebarButton"\]\s*svg,\s*'
            r'\[data-testid="stSidebarCollapseButton"\]\s*svg\s*\{'
            r'[^}]*#061A3A'
        )
        self.assertTrue(
            re.search(pattern_icon, self.app_content, re.DOTALL),
            "Expected toggle button svgs to have explicit #061A3A color."
        )

    def test_h_sidebar_brand_logo_referenced_and_exists(self):
        """(h) ui/app.py references assets/images/logo-light.png with alt 'Care-AI logo' and file exists."""
        self.assertIn("assets/images/logo-light.png", self.app_content)
        self.assertIn('alt="Care-AI logo"', self.app_content)
        logo_path = os.path.join(PROJECT_ROOT, "assets", "images", "logo-light.png")
        self.assertTrue(os.path.isfile(logo_path), f"Logo file does not exist at {logo_path}")


if __name__ == "__main__":
    unittest.main()

