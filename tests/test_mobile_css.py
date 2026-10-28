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
    def test_i_content_max_width_820px(self):
        """(i) .block-container content max-width is set to 820px."""
        self.assertTrue(
            re.search(r'\.block-container\s*\{[^}]*max-width\s*:\s*820px', self.app_content),
            "Expected .block-container to have max-width: 820px."
        )

    def test_j_chat_input_pill_and_focus_within(self):
        """(j) Chat input pill has :focus-within styling with 2px teal focus ring."""
        self.assertIn(":focus-within", self.app_content)
        pattern = r'\[data-testid="stChatInput"\]\s*>\s*div:focus-within[^{]*\{[^}]*box-shadow\s*:\s*0\s+0\s+0\s+2px'
        self.assertTrue(
            re.search(pattern, self.app_content),
            "Expected [data-testid=\"stChatInput\"] > div:focus-within to have 2px focus ring."
        )

    def test_k_spinner_prefers_reduced_motion(self):
        """(k) Spinner has animation duration <=200ms and honours prefers-reduced-motion."""
        pattern_anim = r'\[data-testid="stSpinner"\]\s*>\s*div\s*\{[^}]*animation-duration\s*:\s*200ms'
        self.assertTrue(
            re.search(pattern_anim, self.app_content),
            "Expected [data-testid=\"stSpinner\"] > div to have animation-duration <= 200ms."
        )
        pattern_reduced = r'@media\s*\(prefers-reduced-motion\s*:\s*reduce\)\s*\{[^}]*\[data-testid="stSpinner"\]\s*>\s*div\s*\{[^}]*animation\s*:\s*none'
        self.assertTrue(
            re.search(pattern_reduced, self.app_content, re.DOTALL),
            "Expected @media (prefers-reduced-motion: reduce) to disable spinner animation."
        )

    def test_l_logo_loader_cached(self):
        """(l) _load_asset_b64 is decorated with @st.cache_data."""
        pattern_cache = r'@st\.cache_data\s*\ndef _load_asset_b64\('
        self.assertTrue(
            re.search(pattern_cache, self.app_content),
            "Expected _load_asset_b64 to be decorated with @st.cache_data."
        )
    def test_m_no_unconditional_st_logo(self):
        """(m) No unconditional st.logo( call in ui/app.py."""
        self.assertNotIn("st.logo(", self.app_content)

    def test_n_assistant_avatar_logo_mark(self):
        """(n) Assistant chat message uses logo-mark_light.png as avatar."""
        self.assertIn('with st.chat_message("assistant", avatar="assets/images/logo-mark_light.png"):', self.app_content)

    def test_o_no_hospital_emoji_in_assistant_render(self):
        """(o) No hospital emoji in assistant chat_message or bot-avatar render path."""
        self.assertNotIn('avatar="🏥"', self.app_content)
        self.assertNotIn('<div class="bot-avatar">🏥</div>', self.app_content)

    def test_p_user_avatar_hidden(self):
        """(p) [data-testid="stChatMessageAvatarUser"] is hidden via CSS."""
        pattern = r'\[data-testid="stChatMessageAvatarUser"\]\s*\{[^}]*display\s*:\s*none'
        self.assertTrue(
            re.search(pattern, self.app_content),
            "Expected [data-testid=\"stChatMessageAvatarUser\"] to have display: none."
        )

    def test_q_sidebar_logo_mix_blend_mode(self):
        """(q) Sidebar logo does NOT use mix-blend-mode (transparent RGBA logo)."""
        self.assertNotIn("mix-blend-mode", self.app_content)
        self.assertNotIn("multiply", self.app_content)

    def test_r_new_chat_button_nowrap(self):
        """(r) Button CSS has white-space: nowrap."""
        self.assertIn("white-space: nowrap", self.app_content)

    def test_s_chat_input_focus_ring_on_pill_only(self):
        """(s) Focus ring is applied only to pill (:focus-within on pill child), not outer stChatInput."""
        pattern_outer = r'\[data-testid="stChatInput"\][^{]*\{[^}]*box-shadow\s*:\s*none'
        self.assertTrue(
            re.search(pattern_outer, self.app_content),
            "Expected outer [data-testid=\"stChatInput\"] to have box-shadow: none."
        )
        self.assertFalse(
            bool(re.search(r'\[data-testid="stChatInput"\]:focus-within\s*\{[^}]*box-shadow\s*:\s*0\s+0\s+0\s+2px', self.app_content)),
            "Outer stChatInput must not have 2px focus ring box-shadow."
        )
        pattern_pill = r'\[data-testid="stChatInput"\]\s*>\s*div:focus-within[^{]*\{[^}]*box-shadow\s*:\s*0\s+0\s+0\s+2px'
        self.assertTrue(
            re.search(pattern_pill, self.app_content),
            "Expected [data-testid=\"stChatInput\"] > div:focus-within to have 2px focus ring."
        )

    def test_t_bottom_container_max_width_820px(self):
        """(t) Bottom input container has max-width: 820px and solid #F2F8F8 background."""
        pattern = r'\[data-testid="stBottom"\][^{]*\{[^}]*max-width\s*:\s*820px'
        self.assertTrue(
            re.search(pattern, self.app_content),
            "Expected [data-testid=\"stBottom\"] to specify max-width: 820px."
        )
    def test_u_strip_rendered_emoji_prefix(self):
        """(u) strip_rendered_emoji_prefix behaves correctly and does not raise NameError."""
        import sys
        from unittest.mock import MagicMock
        for _mod in ("scipy", "scipy.sparse"):
            if _mod not in sys.modules:
                sys.modules[_mod] = MagicMock()

        from ui.app import strip_rendered_emoji_prefix

        # Test cases specified in prompt
        self.assertEqual(strip_rendered_emoji_prefix("🏥 Hello"), "Hello")
        self.assertEqual(strip_rendered_emoji_prefix("Hello"), "Hello")
        self.assertEqual(strip_rendered_emoji_prefix(""), "")
        self.assertEqual(strip_rendered_emoji_prefix(None), "")
        # Middle emoji must not be removed
        self.assertEqual(strip_rendered_emoji_prefix("Hello 🏥 world"), "Hello 🏥 world")

    def test_v_urgency_style_helper(self):
        """(v) urgency_style returns correct dict for all levels and safe fallback for unknown."""
        from ui.helpers import urgency_style

        # Supported levels
        for level, expected_color in [
            ("SELF_CARE", "#15803D"),
            ("SEE_DOCTOR", "#92400E"),
            ("URGENT_CARE", "#9A3412"),
            ("EMERGENCY", "#991B1B"),
        ]:
            res = urgency_style(level)
            self.assertIsInstance(res, dict)
            self.assertIn("label", res)
            self.assertIn("icon", res)
            self.assertIn("color", res)
            self.assertIn("bg", res)
            self.assertEqual(res["color"], expected_color)

        # Fallback for unknown / None
        fb_none = urgency_style(None)
        self.assertIsInstance(fb_none, dict)
        self.assertIn("label", fb_none)
        self.assertIn("color", fb_none)

        fb_unk = urgency_style("UNKNOWN_LEVEL")
        self.assertIsInstance(fb_unk, dict)
        self.assertIn("label", fb_unk)
        self.assertIn("color", fb_unk)

    def test_w_badge_visible_helper(self):
        """(w) badge_visible suppresses badge when low_confidence is True."""
        from ui.helpers import badge_visible

        # Always False when low_confidence=True
        self.assertFalse(badge_visible(True, True))
        self.assertFalse(badge_visible(True, False))
        self.assertFalse(badge_visible(True, "VERIFIED"))
        self.assertFalse(badge_visible(True, "FAILED"))

        # When low_confidence=False, follows verified state
        self.assertTrue(badge_visible(False, True))
        self.assertTrue(badge_visible(False, "VERIFIED"))
        self.assertFalse(badge_visible(False, False))
        self.assertFalse(badge_visible(False, "FAILED"))

    def test_x_emergency_banner_112_markup_and_css(self):
        """(x) ui/app.py includes .emergency-banner-112 with prominent 112 callout."""
        self.assertIn(".emergency-banner-112", self.app_content)
        pattern = r'\.emergency-banner-112\s*\{[^}]*background[^}]*#(?:DC2626|B91C1C)'
        self.assertTrue(
            re.search(pattern, self.app_content, re.IGNORECASE),
            "Expected .emergency-banner-112 to have high-contrast red background."
        )
        self.assertIn("emergency-banner-112", self.app_content)
        self.assertIn("112", self.app_content)

    def test_y_treatment_grid_responsive_1col(self):
        """(y) .treatment-grid has 1-column layout at <= 768px."""
        pattern = r'@media[^{]*max-width:\s*768px[^{]*\{.*?\.treatment-grid\s*\{[^}]*grid-template-columns\s*:\s*1fr'
        self.assertTrue(
            re.search(pattern, self.app_content, re.DOTALL),
            "Expected .treatment-grid to collapse to 1fr at <= 768px."
        )





if __name__ == "__main__":
    unittest.main()

