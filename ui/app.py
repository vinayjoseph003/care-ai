"""
Care-AI — Streamlit Browser UI
Place this file at: ui/app.py
Run with: streamlit run ui/app.py
"""

import os
import sys
import re
import queue
import html
import threading
import traceback
import base64
import streamlit as st
from dotenv import load_dotenv

# ── Path setup (so imports from project root work) ──────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.knowledge_base import build_all_chunks
from bot import CareAI
from bot.careai import SAFE_UNAVAILABLE_MESSAGE
from ui.doctor_finder_panel import render_doctor_finder

load_dotenv()

# Sync Streamlit Cloud secrets to os.environ if running in hosted environment
try:
    if hasattr(st, "secrets"):
        for _sec_k, _sec_v in st.secrets.items():
            if isinstance(_sec_v, (str, int, float, bool)) and _sec_k not in os.environ:
                os.environ[_sec_k] = str(_sec_v)
except Exception:
    pass


def _get_setting(key: str, default: str = "") -> str:
    """Retrieve setting from os.environ or st.secrets (Streamlit Cloud)."""
    val = os.environ.get(key)
    if val:
        return val
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


from ui.helpers import (
    strip_rendered_emoji_prefix,
    sanitize_html,
    top_diagnosis_label,
    reliability_badge,
    badge_visible,
    urgency_style,
    _prob_to_pct,
    _prob_css_class,
    _prob_fill_color,
    _prob_label_color,
    _risk_badge_class,
)


# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════

_ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "images"))
_LOGO_LIGHT_PATH = os.path.join(_ASSETS_DIR, "logo-light.png")
_LOGO_MARK_PATH = os.path.join(_ASSETS_DIR, "logo-mark_light.png")


@st.cache_data
def _load_asset_b64(path: str) -> str:
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        pass
    return ""


_LOGO_LIGHT_B64 = _load_asset_b64(_LOGO_LIGHT_PATH) or _load_asset_b64("assets/images/logo-light.png")
_LOGO_MARK_B64 = _load_asset_b64(_LOGO_MARK_PATH) or _load_asset_b64("assets/images/logo-mark_light.png")

st.set_page_config(
    page_title="Care-AI · Medical Assistant",
    page_icon="assets/images/logo-mark_light.png",
    layout="wide",
    initial_sidebar_state="auto",
)



# ════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Light professional medical theme
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Design tokens ───────────────────────────────────────── */
:root {
  --navy:        #061A3A;
  --text:        #061A3A;
  --text-muted:  #243B5A;
  --text-dim:    #4A5D6E;
  --bg:          #F2F8F8;
  --surface:     #FFFFFF;
  --surface2:    #E7EFEF;
  --surface3:    #DCEBEB;
  --border:      #CFDFDF;
  --border-hi:   #A8C5C5;
  --teal:        #10D4B8;
  --teal-dark:   #06695C;
  --teal-deep:   #04544A;
  --teal-dim:    rgba(16, 212, 184, 0.12);
  --cyan:        #1BC8D8;
  --blue:        #0EA5C9;
  --soft-green:  #7FE0B8;
  --green:       #16A34A;
  --green-dim:   #DCFCE7;
  --yellow:      #D97706;
  --yellow-dim:  #FEF3C7;
  --red:         #DC2626;
  --red-dim:     #FEE2E2;
  --orange:      #EA580C;
  --orange-dim:  #FFEDD5;
  --accent:      #06695C;
  --accent-dim:  rgba(6, 105, 92, 0.12);
  --font-sans:   -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Noto Sans Devanagari", "Noto Sans Telugu", "Noto Sans Tamil", sans-serif;
  --font-mono:   ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
  --mono:        var(--font-mono);
  --serif:       var(--font-sans);
  --radius-sm:   6px;
  --radius-md:   10px;
  --radius-lg:   14px;
}

/* ── Global reset ────────────────────────────────────────── */
html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-sans) !important;
}

*:focus-visible {
  outline: 2px solid var(--teal) !important;
  outline-offset: 2px !important;
}

.stApp, [data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
}

/* ── Hide Streamlit chrome ───────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  background: transparent !important;
  padding: 1.6rem 2.2rem 4rem !important;
  max-width: 820px !important;
  margin: 0 auto;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 3px;
}

/* ══════════════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background-color: var(--bg) !important;
  border-right: 1px solid var(--border) !important;
  min-width: 248px !important;
  max-width: 300px !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] > div:first-child {
  overflow-y: auto !important;
  padding: 56px 1rem 1.1rem !important;
  background-color: var(--bg) !important;
}
[data-testid="stSidebarHeader"],
[data-testid="stLogo"] {
  display: none !important;
}

/* Sidebar brand lockup */
.sb-brand {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  padding: 0.2rem 0 1rem;
  margin-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}
.sb-brand-logo {
  width: 180px;
  max-width: 100%;
  height: auto;
  display: block;
  mix-blend-mode: multiply !important;
}

/* Language badge — topmost prominent element */
.sb-lang-badge {
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--teal-dim);
  border: 1.5px solid var(--teal);
  border-radius: 10px;
  padding: 0.65rem 0.9rem;
  margin-bottom: 0.7rem;
}
.sb-lang-icon {
  font-size: 1.1rem;
  flex-shrink: 0;
}
.sb-lang-label {
  font-family: var(--font-sans);
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--teal-dark) !important;
  margin-bottom: 0.15rem;
}
.sb-lang-value {
  font-family: var(--font-sans);
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text) !important;
  line-height: 1.2;
}

/* Metric card */
.sb-metric {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 0.65rem 0.9rem;
  margin-bottom: 0.5rem;
}
.sb-metric-label {
  font-family: var(--font-sans);
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.02em;
  color: var(--text-muted) !important;
  margin-bottom: 0.2rem;
}
.sb-metric-value {
  font-family: var(--font-sans);
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text) !important;
}
.sv-teal   { color: var(--teal-dark)   !important; }
.sv-green  { color: var(--green)  !important; }
.sv-yellow { color: var(--yellow) !important; }
.sv-orange { color: var(--orange) !important; }
.sv-red    { color: var(--red)    !important; }
.sv-blue   { color: var(--blue)   !important; }

/* Sidebar section divider */
.sb-section {
  font-family: var(--font-sans);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.04em;
  color: var(--text-dim) !important;
  margin: 1.2rem 0 0.5rem;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid var(--border);
}

/* SOCRATES progress */
.sb-progress-wrap {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 0.7rem 0.9rem;
  margin-bottom: 0.5rem;
}
.sb-progress-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.45rem;
}
.sb-progress-title {
  font-family: var(--font-sans);
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.02em;
  color: var(--text-muted) !important;
}
.sb-progress-count {
  font-family: var(--font-sans);
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--teal-dark) !important;
}
.sb-progress-bar-bg {
  height: 4px;
  background: var(--border);
  border-radius: 2px;
  overflow: hidden;
}
.sb-progress-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--teal), var(--accent));
  border-radius: 2px;
  transition: width 0.4s ease;
}
/* SOCRATES field dots */
.sb-socrates-dots {
  display: flex;
  gap: 4px;
  margin-top: 0.5rem;
  flex-wrap: wrap;
}
.sb-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  border: 1px solid var(--border-hi);
  background: var(--surface3);
  transition: background 0.2s;
}
.sb-dot.filled {
  background: var(--teal);
  border-color: var(--teal);
}

/* Sidebar disclaimer */
.sb-disclaimer {
  font-family: var(--font-sans);
  font-size: 0.75rem;
  color: var(--text-muted) !important;
  line-height: 1.55;
  padding: 0.7rem 0.9rem;
  background: rgba(217,119,6,0.05);
  border: 1px solid rgba(217,119,6,0.15);
  border-radius: var(--radius-sm);
  margin-top: 0.5rem;
}

/* Streamlit button overrides inside sidebar */
.stButton button,
div[data-testid="stColumn"] .stButton button {
  white-space: nowrap !important;
}
[data-testid="stSidebar"] .stButton button {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.06em !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.45rem 0.8rem !important;
  transition: border-color 0.15s, background 0.15s !important;
  white-space: nowrap !important;
}
[data-testid="stSidebar"] .stButton button:hover {
  border-color: var(--border-hi) !important;
  background: var(--surface3) !important;
}

/* ══════════════════════════════════════════════════════════
   MAIN HEADER
══════════════════════════════════════════════════════════ */
.main-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  margin-bottom: 1.6rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}
.main-header-left {}
.main-title {
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: var(--font-sans);
  font-size: 1.35rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--navy) !important;
  line-height: 1.2;
  margin: 0;
}
.main-title-mark {
  width: 32px;
  height: 32px;
  object-fit: contain;
  display: inline-block;
  vertical-align: middle;
}
.main-subtitle {
  font-family: var(--font-sans);
  font-size: 0.85rem;
  color: var(--text-muted);
  margin-top: 0.25rem;
  font-style: normal;
  font-weight: 400;
}

/* ══════════════════════════════════════════════════════════
   CHAT MESSAGES
══════════════════════════════════════════════════════════ */

/* Strip Streamlit's default chat message styling */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
}
[data-testid="stChatMessageAvatarUser"] {
  display: none !important;
}

/* User bubble — right-aligned */
.user-msg-wrap {
  display: flex;
  justify-content: flex-end;
  margin: 8px 0;
}
.user-bubble {
  background: #E3F7F4;
  border: 1px solid #C4ECE5;
  border-radius: 16px 16px 4px 16px;
  padding: 10px 16px;
  max-width: 75%;
  font-family: var(--font-sans);
  font-size: 0.95rem;
  line-height: 1.55;
  color: #061A3A;
}

/* Bot message container */
.bot-msg-wrap {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin: 8px 0;
}
.bot-avatar {
  width: 32px; height: 32px;
  background: #FFFFFF;
  border: 1px solid #D5E4E4;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.9rem;
  flex-shrink: 0;
  margin-top: 2px;
  box-shadow: 0 1px 3px rgba(6, 26, 58, 0.04);
}
.bot-bubble {
  background: #FFFFFF;
  border: 1px solid #D5E4E4;
  border-radius: 4px 16px 16px 16px;
  padding: 12px 18px;
  max-width: 85%;
  font-family: var(--font-sans);
  font-size: 0.95rem;
  line-height: 1.6;
  color: #061A3A;
  flex: 1;
  box-shadow: 0 1px 3px rgba(6, 26, 58, 0.04), 0 1px 2px rgba(6, 26, 58, 0.02);
}

/* Triage banner */
.triage-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-sans);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  padding: 4px 12px;
  border-radius: 6px;
  margin-bottom: 0.8rem;
}
.triage-SELF_CARE {
  background: #DCFCE7;
  color: #15803D;
  border: 1px solid #16A34A;
}
.triage-SEE_DOCTOR {
  background: #FEF3C7;
  color: #92400E;
  border: 1px solid #D97706;
}
.triage-URGENT_CARE {
  background: #FFEDD5;
  color: #9A3412;
  border: 1px solid #EA580C;
}
.triage-EMERGENCY {
  background: #FEE2E2;
  color: #991B1B;
  border: 1px solid #DC2626;
  animation: triage-pulse 2s ease-in-out infinite;
}
.triage-DEFAULT {
  background: #E7EFEF;
  color: #061A3A;
  border: 1px solid #CBD5E0;
}
@keyframes triage-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
  50%       { box-shadow: 0 0 0 4px rgba(220,38,38,0.12); }
}

/* Emergency Full-Width Banner (.emergency-banner-112) */
.emergency-banner-112 {
  background: #DC2626 !important;
  color: #FFFFFF !important;
  border-radius: var(--radius-md);
  padding: 1.25rem 1.5rem;
  margin: 0.8rem 0 1.2rem;
  box-shadow: 0 4px 14px rgba(220, 38, 38, 0.3);
  border: 2px solid #B91C1C;
  font-family: var(--font-sans);
}
.emergency-banner-lead {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 0.6rem;
  flex-wrap: wrap;
}
.emergency-banner-badge {
  background: #FFFFFF;
  color: #DC2626;
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0.05em;
  padding: 4px 10px;
  border-radius: 4px;
  text-transform: uppercase;
}
.emergency-banner-call {
  font-size: 1.1rem;
  font-weight: 700;
  color: #FFFFFF;
}
.emergency-banner-number {
  font-size: 2.2rem;
  font-weight: 800;
  line-height: 1;
  color: #FFFFFF;
  letter-spacing: -0.02em;
}
.emergency-banner-text {
  font-size: 0.95rem;
  line-height: 1.6;
  color: #FFFFFF;
  opacity: 0.98;
}

/* Low-confidence notice (amber WCAG AA palette) */
.low-confidence-box {
  background: #FEF3C7;
  color: #78350F;
  border: 1px solid #F59E0B;
  border-left: 4px solid #D97706;
  border-radius: var(--radius-md);
  padding: 0.85rem 1rem;
  font-family: var(--font-sans);
  font-size: 0.88rem;
  line-height: 1.5;
  margin-bottom: 1rem;
}
  50%       { box-shadow: 0 0 0 4px rgba(220,38,38,0.12); }
}

/* ══════════════════════════════════════════════════════════
   RESULT PANELS — section label
══════════════════════════════════════════════════════════ */
.panel-section-label {
  font-family: var(--font-sans);
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--navy);
  margin: 1.2rem 0 0.6rem;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid var(--border);
}

/* ══════════════════════════════════════════════════════════
   DIFFERENTIAL DIAGNOSIS CARDS
══════════════════════════════════════════════════════════ */
.diff-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.7rem;
}
.diff-card {
  position: relative;
  background: #FFFFFF;
  border: 1px solid #CBD5E0;
  border-radius: var(--radius-md);
  padding: 1rem 1.1rem 0.85rem;
  overflow: hidden;
  transition: border-color 0.2s;
  box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.diff-card:hover { border-color: var(--border-hi); }

/* Rank badge top-right */
.diff-rank-badge {
  position: absolute;
  top: 10px; right: 10px;
  font-family: var(--font-sans);
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--navy);
  background: var(--surface3);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 2px 6px;
}

/* Left accent bar by probability */
.diff-card::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  border-radius: var(--radius-md) 0 0 var(--radius-md);
}
.diff-card.prob-high::before   { background: #DC2626; }
.diff-card.prob-medium::before { background: #D97706; }
.diff-card.prob-low::before    { background: #16A34A; }

.diff-condition {
  font-family: var(--serif);
  font-size: 0.9rem;
  font-weight: 600;
  color: #1A202C;
  margin-bottom: 0.5rem;
  line-height: 1.3;
  padding-right: 36px;
}
.diff-prob-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 0.5rem;
}
.diff-prob-label {
  font-family: var(--font-sans);
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.diff-prob-bar {
  flex: 1;
  height: 3px;
  background: var(--border);
  border-radius: 2px;
  overflow: hidden;
}
.diff-prob-fill {
  height: 100%;
  border-radius: 2px;
}
.diff-desc {
  font-family: var(--serif);
  font-size: 0.78rem;
  color: #4A5568;
  line-height: 1.5;
  margin-top: 0.45rem;
}
.diff-evidence {
  font-family: var(--font-sans);
  font-size: 0.72rem;
  color: #92400E;
  margin-top: 0.4rem;
  line-height: 1.4;
  padding: 4px 8px;
  background: #FEF3C7;
  border-radius: 4px;
  border-left: 2px solid #D97706;
}

/* ══════════════════════════════════════════════════════════
   TREATMENT CARDS
══════════════════════════════════════════════════════════ */
.treatment-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.65rem;
}
.treatment-card {
  background: #FFFFFF;
  border: 1px solid #CBD5E0;
  border-radius: var(--radius-md);
  padding: 0.85rem 1rem;
}
.treatment-card.full-width { grid-column: 1 / -1; }
.tc-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 0.6rem;
}
.tc-icon {
  font-size: 13px;
  width: 24px; height: 24px;
  display: flex; align-items: center; justify-content: center;
  background: var(--surface3);
  border-radius: 6px;
  flex-shrink: 0;
}
.tc-title {
  font-family: var(--font-sans);
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.01em;
  color: var(--navy);
}
.tc-list {
  margin: 0;
  padding-left: 1.15rem;
  font-family: var(--serif);
  font-size: 0.85rem;
  line-height: 1.65;
  color: var(--text);
}
.tc-escalation-text {
  font-family: var(--serif);
  font-size: 0.85rem;
  color: var(--text);
  line-height: 1.55;
  margin: 0;
}
.treatment-card.escalation {
  border-left: 3px solid #D97706;
  background: #FFFBEB;
}

/* ══════════════════════════════════════════════════════════
   RED FLAGS BOX
══════════════════════════════════════════════════════════ */
.redflag-box {
  background: #FFF5F5;
  border: 1px solid #FECACA;
  border-left: 4px solid #DC2626;
  border-radius: 0 var(--radius-md) var(--radius-md) 0;
  padding: 0.9rem 1.1rem;
}
.redflag-title {
  font-family: var(--font-sans);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  color: #DC2626;
  margin-bottom: 0.55rem;
  display: flex;
  align-items: center;
  gap: 6px;
}
.redflag-list {
  margin: 0;
  padding-left: 1.1rem;
  font-family: var(--serif);
  font-size: 0.85rem;
  line-height: 1.65;
  color: #7F1D1D;
}
.redflag-threshold {
  margin-top: 0.55rem;
  font-family: var(--font-sans);
  font-size: 0.75rem;
  font-weight: 600;
  color: #DC2626;
}

/* ══════════════════════════════════════════════════════════
   MENTAL HEALTH BANNER
══════════════════════════════════════════════════════════ */
.mh-banner {
  background: #EFF6FF;
  border: 1px solid #BFDBFE;
  border-left: 3px solid #2563EB;
  border-radius: 0 var(--radius-md) var(--radius-md) 0;
  padding: 0.8rem 1.1rem;
  font-family: var(--serif);
  font-size: 0.88rem;
  line-height: 1.6;
  color: #1E40AF;
}

/* ══════════════════════════════════════════════════════════
   XAI PANEL
══════════════════════════════════════════════════════════ */
.xai-panel {
  background: #F0F9FF;
  border: 1px solid #BAE6FD;
  border-radius: var(--radius-md);
  padding: 0.9rem 1.1rem;
}
.xai-panel p {
  font-family: var(--serif);
  font-size: 0.88rem;
  line-height: 1.65;
  color: #0C4A6E;
  margin: 0;
}

/* ══════════════════════════════════════════════════════════
   BADGE ROW & SOURCE PILLS
══════════════════════════════════════════════════════════ */
.badge-row {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
  margin-top: 0.7rem;
}
.badge {
  font-family: var(--font-sans);
  font-size: 0.68rem;
  letter-spacing: 0.02em;
  padding: 3px 9px;
  border-radius: 4px;
  font-weight: 600;
}
.badge-low    { background: #DCFCE7; color: #16A34A; border: 1px solid #16A34A; }
.badge-medium { background: #FEF3C7; color: #D97706; border: 1px solid #D97706; }
.badge-high   { background: #FEE2E2; color: #DC2626; border: 1px solid #DC2626; }

.sources-wrap { margin-top: 0.6rem; }
.sources-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
  margin-top: 0.35rem;
}
.source-pill {
  font-family: var(--font-sans);
  font-size: 0.72rem;
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2px 10px;
  color: var(--navy);
  letter-spacing: 0.02em;
}

/* ══════════════════════════════════════════════════════════
   EXPANDER OVERRIDES
══════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
  background: #FFFFFF !important;
  border: 1px solid #CBD5E0 !important;
  border-radius: var(--radius-md) !important;
  margin-top: 0.5rem !important;
}
[data-testid="stExpander"] summary {
  font-family: var(--font-sans) !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em !important;
  color: var(--navy) !important;
  padding: 0.55rem 0.8rem !important;
}
[data-testid="stExpander"] summary:hover {
  color: var(--text) !important;
}
[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
  padding: 0.3rem 0.8rem 0.8rem !important;
}

/* ══════════════════════════════════════════════════════════
   CHAT INPUT & BOTTOM CONTAINER
══════════════════════════════════════════════════════════ */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
  background-color: #F2F8F8 !important;
  background: #F2F8F8 !important;
  max-width: 820px !important;
  margin: 0 auto !important;
}
[data-testid="stBottom"] > div {
  max-width: 820px !important;
  margin: 0 auto !important;
  background-color: #F2F8F8 !important;
}
[data-testid="stChatInput"] {
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
  background: transparent !important;
  padding-top: 0.8rem !important;
}
[data-testid="stChatInput"]:focus,
[data-testid="stChatInput"]:focus-within {
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
}
[data-testid="stChatInput"] > div {
  border-radius: 9999px !important;
  border: 1px solid var(--border) !important;
  background: #FFFFFF !important;
  box-shadow: 0 1px 4px rgba(6, 26, 58, 0.04) !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
[data-testid="stChatInput"] > div:focus-within {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 2px var(--teal) !important;
  outline: none !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  outline: none !important;
  color: var(--text) !important;
  font-family: var(--font-sans) !important;
  font-size: 0.95rem !important;
  border-radius: 9999px !important;
  padding: 0.75rem 1.25rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: var(--text-dim) !important;
}

/* ══════════════════════════════════════════════════════════
   STREAMLIT PROGRESS BAR OVERRIDE
══════════════════════════════════════════════════════════ */
[data-testid="stProgress"] > div {
  background: var(--border) !important;
}
[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, var(--teal), var(--accent)) !important;
}

/* ══════════════════════════════════════════════════════════
   EMPTY STATE
══════════════════════════════════════════════════════════ */
.empty-state {
  text-align: center;
  padding: 3.5rem 2rem;
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: 0 1px 4px rgba(6, 26, 58, 0.04);
}
.empty-state-icon {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  opacity: 0.8;
}
.empty-state-title {
  font-family: var(--font-sans);
  font-size: 1.1rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--navy);
  margin-bottom: 0.5rem;
}
.empty-state-langs {
  display: flex;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 1rem;
}
.lang-chip {
  font-family: var(--font-sans);
  font-size: 0.75rem;
  font-weight: 500;
  padding: 4px 12px;
  border: 1px solid var(--border);
  border-radius: 9999px;
  color: var(--teal-dark);
  background: var(--surface2);
}

/* Streamlit spinner override */
[data-testid="stSpinner"] > div {
  border-width: 2px !important;
  border-color: var(--teal) transparent transparent !important;
  animation-duration: 200ms !important;
}
@media (prefers-reduced-motion: reduce) {
  [data-testid="stSpinner"] > div {
    animation: none !important;
    border-color: var(--teal) !important;
    opacity: 0.7 !important;
  }
}

/* ══════════════════════════════════════════════════════════
   SIDEBAR COLLAPSE / EXPAND BUTTONS — make visible on light bg
══════════════════════════════════════════════════════════ */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] {
  visibility: visible !important;
  position: fixed !important;
  top: 12px !important;
  left: 12px !important;
  z-index: 1000001 !important;
  width: 36px !important;
  height: 36px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: #061A3A !important;
  box-shadow: 0 1px 3px rgba(6, 26, 58, 0.08) !important;
  cursor: pointer !important;
  transition: all 0.15s ease !important;
}
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapseButton"]:hover {
  background: var(--surface2) !important;
  border-color: var(--border-hi) !important;
}
[data-testid="stExpandSidebarButton"]:focus-visible,
[data-testid="stSidebarCollapseButton"]:focus-visible {
  outline: 2px solid var(--teal) !important;
  outline-offset: 2px !important;
}
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapseButton"] svg {
  fill: #061A3A !important;
  stroke: #061A3A !important;
  color: #061A3A !important;
  width: 18px !important;
  height: 18px !important;
}

/* Sidebar expand/collapse arrow button (inside sidebar) */
button[kind="header"],
[data-testid="stSidebarNav"] button,
section[data-testid="stSidebar"] button[aria-label="Close sidebar"],
section[data-testid="stSidebar"] button[aria-label="Open sidebar"] {
  background: var(--surface2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] button[aria-label="Close sidebar"] svg,
section[data-testid="stSidebar"] button[aria-label="Open sidebar"] svg {
  fill: #061A3A !important;
  stroke: #061A3A !important;
}

/* Ensure sidebar never has a dark overlay */
section[data-testid="stSidebar"] {
  background: var(--bg) !important;
}
section[data-testid="stSidebar"]::before {
  display: none !important;
}

/* ══════════════════════════════════════════════════════════
   RESPONSIVE MEDIA QUERIES (MOBILE)
══════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
  .block-container {
    padding: 0.8rem 1rem 6rem !important;
  }
  .treatment-grid {
    grid-template-columns: 1fr !important;
  }
  .stButton button,
  [data-testid="stExpandSidebarButton"],
  [data-testid="stSidebarCollapseButton"] {
    min-height: 44px !important;
    min-width: 44px !important;
    width: 44px !important;
    height: 44px !important;
  }
  [data-testid="stSidebar"] {
    max-width: 85vw !important;
  }
  /* Differential cards: stack columns vertically on mobile */
  [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
    gap: 0.6rem !important;
  }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
  [data-testid="stHorizontalBlock"] > [data-testid="column"] {
    min-width: 100% !important;
    flex: 1 1 100% !important;
  }
}

@media (max-width: 480px) {
  .user-bubble,
  .bot-bubble {
    max-width: 90% !important;
  }
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  TRIAGE METADATA
# ════════════════════════════════════════════════════════════════════════════

TRIAGE_META = {
    "SELF_CARE":   urgency_style("SELF_CARE"),
    "SEE_DOCTOR":  urgency_style("SEE_DOCTOR"),
    "URGENT_CARE": urgency_style("URGENT_CARE"),
    "EMERGENCY":   urgency_style("EMERGENCY"),
}

SOCRATES_DIMS = [
    "symptom_or_topic", "onset", "character", "radiation",
    "associations", "timing", "exacerbating", "severity",
]


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar(bot):
    with st.sidebar:
        # Brand
        st.markdown(f"""
<div class="sb-brand">
  <img src="data:image/png;base64,{_LOGO_LIGHT_B64}" class="sb-brand-logo" alt="Care-AI logo" width="180" style="mix-blend-mode: multiply;">
</div>""", unsafe_allow_html=True)

        # ── Detected language — topmost prominent badge ───────
        lang = sanitize_html(getattr(bot, 'user_language', {}).get('language', '—'), preserve_newlines=False)
        st.markdown(f"""
<div class="sb-lang-badge">
  <span class="sb-lang-icon">🌐</span>
  <div>
    <div class="sb-lang-label">Detected Language</div>
    <div class="sb-lang-value">{lang}</div>
  </div>
</div>""", unsafe_allow_html=True)

        # ── Session metrics ───────────────────────────────────
        turns = len([m for m in st.session_state.messages
                     if m["role"] == "user"])
        st.markdown(f"""
<div class="sb-metric">
  <div class="sb-metric-label">Session Turns</div>
  <div class="sb-metric-value sv-teal">{turns}</div>
</div>""", unsafe_allow_html=True)

        triage = getattr(bot, 'last_triage', None)
        if triage and triage in TRIAGE_META:
            m = TRIAGE_META[triage]
            color_map = {
                "SELF_CARE": "sv-green", "SEE_DOCTOR": "sv-yellow",
                "URGENT_CARE": "sv-orange", "EMERGENCY": "sv-red",
            }
            cls = color_map.get(triage, "sv-blue")
            st.markdown(f"""
<div class="sb-metric">
  <div class="sb-metric-label">Triage Level</div>
  <div class="sb-metric-value {cls}">{m['icon']} {m['label']}</div>
</div>""", unsafe_allow_html=True)

        top_text = top_diagnosis_label(getattr(bot, 'xai_logs', None), getattr(bot, 'last_top_condition', ''))
        top = sanitize_html(top_text, preserve_newlines=False)
        if top:
            st.markdown(f"""
<div class="sb-metric">
  <div class="sb-metric-label">Top Diagnosis</div>
  <div class="sb-metric-value sv-teal">{top}</div>
</div>""", unsafe_allow_html=True)

        spec = sanitize_html(getattr(bot, 'last_specialist', ''), preserve_newlines=False)
        if spec:
            st.markdown(f"""
<div class="sb-metric">
  <div class="sb-metric-label">Suggested Specialist</div>
  <div class="sb-metric-value">{spec}</div>
</div>""", unsafe_allow_html=True)

        # ── SOCRATES progress ─────────────────────────────────
        st.markdown('<div class="sb-section">SOCRATES Intake</div>',
                    unsafe_allow_html=True)

        collected = getattr(bot, 'collected_info', {})
        filled_dims = [
            d for d in SOCRATES_DIMS
            if collected.get(d) and str(collected[d]).strip()
            not in ('', '...', 'null', 'None')
        ]
        filled = len(filled_dims)
        total  = len(SOCRATES_DIMS)
        pct    = int(filled / total * 100)

        dots_html = "".join(
            f'<div class="sb-dot{"  filled" if d in filled_dims else ""}"></div>'
            for d in SOCRATES_DIMS
        )

        st.markdown(f"""
<div class="sb-progress-wrap">
  <div class="sb-progress-label">
    <span class="sb-progress-title">Fields Collected</span>
    <span class="sb-progress-count">{filled}/{total}</span>
  </div>
  <div class="sb-progress-bar-bg">
    <div class="sb-progress-bar-fill" style="width:{pct}%"></div>
  </div>
  <div class="sb-socrates-dots">{dots_html}</div>
</div>""", unsafe_allow_html=True)

        # ── Commands ──────────────────────────────────────────
        st.markdown('<div class="sb-section">Commands</div>',
                    unsafe_allow_html=True)

        if st.button("🔄  Reset Session", use_container_width=True):
            bot.reset()
            st.session_state.messages    = []
            st.session_state.last_result = None
            for k in list(st.session_state.keys()):
                if k.startswith("df_"):
                    del st.session_state[k]
            st.rerun()

        st.download_button(
            label="📤  Export Log",
            data=bot.export_log_json(),
            file_name="care_ai_session.json",
            mime="application/json",
            use_container_width=True,
        )
        st.caption("Downloads to your device only. Nothing is saved on the server.")

        # ── Disclaimer ────────────────────────────────────────
        st.markdown("""
<div class="sb-disclaimer">
  ⚠️ Care-AI provides general health information only.
  Always consult a qualified doctor before acting on this advice.
  In an emergency, call <strong>112</strong>.<br><br>
  Refreshing the page clears your chat (privacy).
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  RESULT PANEL RENDERERS
# ════════════════════════════════════════════════════════════════════════════

def render_differential(differential: list):
    if not differential:
        return

    st.markdown('<div class="panel-section-label">Differential Diagnosis</div>',
                unsafe_allow_html=True)

    n = min(len(differential), 3)
    cols = st.columns(n)

    for i, d in enumerate(differential[:n]):
        prob_raw   = str(d.get('probability', 'MEDIUM')).upper()
        pct        = _prob_to_pct(prob_raw)
        css_class  = _prob_css_class(prob_raw)
        fill_color = _prob_fill_color(prob_raw)
        lbl_color  = _prob_label_color(prob_raw)
        description  = d.get('description', '')
        key_evidence = d.get('key_evidence', '')

        desc_html = (
            f'<div class="diff-desc">{sanitize_html(description)}</div>'
            if description else ''
        )
        evidence_html = (
            f'<div class="diff-evidence">🔑 {sanitize_html(key_evidence)}</div>'
            if key_evidence else ''
        )

        with cols[i]:
            st.markdown(f"""
<div class="diff-card {css_class}">
  <div class="diff-rank-badge">#{d.get('rank', i+1)}</div>
  <div class="diff-condition">{sanitize_html(d.get('condition', 'Unknown'), False)}</div>
  <div class="diff-prob-row">
    <span class="diff-prob-label" style="color:{lbl_color}">{prob_raw}</span>
    <div class="diff-prob-bar">
      <div class="diff-prob-fill" style="width:{pct}%;background:{fill_color}"></div>
    </div>
  </div>
  {desc_html}
  {evidence_html}
</div>""", unsafe_allow_html=True)


def render_treatment(treatment: dict):
    if not treatment:
        return

    home  = treatment.get("home_care", [])
    life  = treatment.get("lifestyle_tips", [])
    avoid = treatment.get("what_to_avoid", [])
    esc   = treatment.get("escalation_timeline", "")

    def _li(items):
        return "".join(f"<li>{sanitize_html(i)}</li>" for i in items) if items else "<li>—</li>"

    st.markdown('<div class="treatment-grid">', unsafe_allow_html=True)

    if home:
        st.markdown(f"""
<div class="treatment-card">
  <div class="tc-header">
    <div class="tc-title">Home Care</div>
  </div>
  <ul class="tc-list">{_li(home)}</ul>
</div>""", unsafe_allow_html=True)

    if life:
        st.markdown(f"""
<div class="treatment-card">
  <div class="tc-header">
    <div class="tc-title">Lifestyle Tips</div>
  </div>
  <ul class="tc-list">{_li(life)}</ul>
</div>""", unsafe_allow_html=True)

    if avoid:
        st.markdown(f"""
<div class="treatment-card full-width">
  <div class="tc-header">
    <div class="tc-title">What to Avoid</div>
  </div>
  <ul class="tc-list">{_li(avoid)}</ul>
</div>""", unsafe_allow_html=True)

    if esc:
        st.markdown(f"""
<div class="treatment-card full-width escalation">
  <div class="tc-header">
    <div class="tc-title">What to Do Next / Escalation Timeline</div>
  </div>
  <p class="tc-escalation-text">{sanitize_html(esc)}</p>
</div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_red_flags(red_flags: dict):
    if not red_flags or not red_flags.get("red_flags"):
        return
    flags     = red_flags.get("red_flags", [])
    threshold = red_flags.get("emergency_threshold", "")
    items_html = "".join(f"<li>{sanitize_html(f)}</li>" for f in flags)
    st.markdown(f"""
<div class="redflag-box">
  <div class="redflag-title">🚩 Warning Signs — Seek Immediate Care If You Notice:</div>
  <ul class="redflag-list">{items_html}</ul>
  {f'<div class="redflag-threshold">{sanitize_html(threshold)}</div>' if threshold else ''}
</div>""", unsafe_allow_html=True)


def render_result_panels(result: dict, bot=None):
    """Render all panels below the bot's text response."""
    if not result or result.get("type") not in ("answer",):
        return

    if result.get("low_confidence"):
        st.markdown(
            '<div class="low-confidence-box">⚠️ Low-confidence assessment: the possible conditions below are for discussion with a doctor, not a diagnosis.</div>',
            unsafe_allow_html=True,
        )

    render_differential(result.get("differential", []))

    treatment = result.get("treatment")
    red_flags = result.get("red_flags")
    xai       = result.get("xai_explanation", "")
    mh_msg    = result.get("mental_health_msg", "")
    sources   = result.get("sources", [])
    risk      = result.get("hallucination_risk", "N/A")
    verdict   = result.get("fact_check", "N/A")

    if treatment:
        with st.expander("💊  Treatment Guidance", expanded=False):
            render_treatment(treatment)

    if red_flags and red_flags.get("red_flags"):
        with st.expander("🚩  Red Flag Symptoms", expanded=True):
            render_red_flags(red_flags)

    if xai:
        with st.expander("💡  Why did I say this? (XAI)", expanded=False):
            st.markdown(
                f'<div class="xai-panel"><p>{sanitize_html(xai)}</p></div>',
                unsafe_allow_html=True
            )

    if mh_msg:
        st.markdown(f'<div class="mh-banner">💙 {sanitize_html(mh_msg)}</div>',
                    unsafe_allow_html=True)

    # ── Reliability badge (suppressed when low_confidence is True) ─
    is_low_conf = bool(result.get("low_confidence"))
    _icon, _label, _cls = reliability_badge(
        risk, verdict, low_confidence=is_low_conf
    )

    if _label != "Well-supported" or badge_visible(is_low_conf, verdict):
        st.markdown(
            f'<div class="badge-row"><span class="badge {_cls}">{_icon} {_label}</span></div>',
            unsafe_allow_html=True
        )

    # ── Sources (collapsed by default) ─────────────────────────
    if sources:
        with st.expander("📚 Sources & Evidence", expanded=False):
            pills = "".join(
                f'<span class="source-pill">{sanitize_html(s, False)}</span>' for s in sources
            )
            st.markdown(
                f'<div class="sources-wrap">'
                f'<div class="sources-pills">{pills}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Doctor Finder panel ───────────────────────────────────
    if bot is not None:
        render_doctor_finder(bot, result)


# ════════════════════════════════════════════════════════════════════════════
#  STREAMING — Redirect bot's stdout tokens into Streamlit
# ════════════════════════════════════════════════════════════════════════════

class TokenQueue:
    """Captures tokens written to stdout by CareAI._stream() and yields them."""
    def __init__(self):
        self.q     = queue.Queue()
        self._done = False

    def write(self, text):
        if text:
            self.q.put(text)

    def flush(self):
        pass

    def mark_done(self):
        self._done = True

    def token_gen(self):
        while True:
            try:
                token = self.q.get(timeout=0.05)
                yield token
            except queue.Empty:
                if self._done:
                    break


def run_bot_in_thread(bot, user_message: str, tq: TokenQueue,
                      result_holder: list):
    """Run bot.chat() in a background thread, capturing stdout tokens."""
    import traceback
    old_stdout = sys.stdout
    sys.stdout = tq
    try:
        result = bot.chat(user_message)
        result_holder.append(result)
    except Exception as e:
        tb = traceback.format_exc()
        result_holder.append({
            "type":      "error",
            "message":   f"{type(e).__name__}: {e}",
            "traceback": tb,
        })
    finally:
        sys.stdout = old_stdout
        tq.mark_done()


# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT & RESOURCE CACHING
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Care-AI knowledge base…")
def load_shared_resources():
    """
    Load expensive, immutable resources (knowledge chunks, vector index,
    symptom scorer, LLM router) once and share them across all user sessions.
    """
    router_config = {
        "primary_provider":   _get_setting("LLM_PRIMARY_PROVIDER", "gemini"),
        "openrouter_api_key": _get_setting("OPENROUTER_API_KEY", ""),
        "openrouter_model":   _get_setting("OPENROUTER_MODEL",   "meta-llama/llama-3.3-70b-instruct"),
        "groq_api_key":       _get_setting("GROQ_API_KEY",       ""),
        "groq_model":         _get_setting("GROQ_MODEL",         "llama-3.3-70b-versatile"),
        "gemini_api_key":     _get_setting("GEMINI_API_KEY",     ""),
        "gemini_model":       _get_setting("GEMINI_MODEL",       "gemini-3.5-flash-lite"),
        "ollama_model":       _get_setting("OLLAMA_MODEL",       "qwen2.5:7b"),
        "ollama_host":        _get_setting("OLLAMA_HOST",        "http://localhost:11434"),
    }
    has_cloud = any([
        router_config["openrouter_api_key"],
        router_config["groq_api_key"],
        router_config["gemini_api_key"],
    ])
    if not has_cloud:
        st.error(
            "❌ No LLM provider API key found. Please configure GEMINI_API_KEY or OPENROUTER_API_KEY "
            "in your environment variables or Streamlit Secrets dashboard."
        )
        st.stop()
    all_chunks, severity_map, precaution_map = build_all_chunks(data_dir="data")
    from core import VectorStore, SymptomScorer
    from core.llm_router import LLMRouter
    vs = VectorStore(all_chunks)
    scorer = SymptomScorer(severity_map)
    router = LLMRouter(router_config)
    return {
        "chunks":         all_chunks,
        "severity_map":   severity_map,
        "precaution_map": precaution_map,
        "vector_store":   vs,
        "symptom_scorer": scorer,
        "router":         router,
        "router_config":  router_config,
    }


def get_session_bot() -> CareAI:
    """
    Ensure each user session has its own isolated CareAI instance
    (conversation history, detected language, collected info, triage state),
    while referencing the shared cached knowledge base and vector store.
    """
    if "bot" not in st.session_state:
        res = load_shared_resources()
        st.session_state.bot = CareAI(
            knowledge_chunks=res["chunks"],
            severity_map=res["severity_map"],
            precaution_map=res["precaution_map"],
            vector_store=res["vector_store"],
            symptom_scorer=res["symptom_scorer"],
            router=res["router"],
            router_config=res["router_config"],
        )
    return st.session_state.bot


def load_bot() -> CareAI:
    """Compatibility alias returning the per-session CareAI bot."""
    return get_session_bot()


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "bot" not in st.session_state:
        get_session_bot()


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    bot = get_session_bot()
    init_session()

    # ── Sidebar — unconditional, every run ───────────────────
    render_sidebar(bot)

    # ── Header ────────────────────────────────────────────────
    col_title, col_btn = st.columns([3.5, 1.2])
    with col_title:
        st.markdown(f"""
<div class="main-header">
  <div class="main-header-left">
    <div class="main-title">
      <img src="data:image/png;base64,{_LOGO_MARK_B64}" class="main-title-mark" alt="Care-AI logo" width="32" height="32">
      <span>CARE-AI</span>
    </div>
    <div class="main-subtitle">
      Multilingual · XAI · Hallucination-Controlled Medical Assistant
    </div>
  </div>
</div>""", unsafe_allow_html=True)
    with col_btn:
        st.markdown('<div style="padding-top:0.5rem"></div>',
                    unsafe_allow_html=True)
        if st.button("🔄 New Chat", use_container_width=True, type="secondary"):
            bot.reset()
            st.session_state.messages    = []
            st.session_state.last_result = None
            for k in list(st.session_state.keys()):
                if k.startswith("df_"):
                    del st.session_state[k]
            st.rerun()

    # ── Empty state ───────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("""
<div class="empty-state">
  <div class="empty-state-icon">🩺</div>
  <div class="empty-state-title">Describe your symptoms to begin</div>
  <div style="font-family:var(--serif);font-size:0.84rem;color:var(--text-muted)">
    Ask in any language — Hindi, Telugu, Tamil, or English
  </div>
  <div class="empty-state-langs">
    <span class="lang-chip">English</span>
    <span class="lang-chip">हिन्दी</span>
    <span class="lang-chip">తెలుగు</span>
    <span class="lang-chip">தமிழ்</span>
    <span class="lang-chip">Code-mix</span>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(
                    f'<div class="user-msg-wrap">'
                    f'<div class="user-bubble">{sanitize_html(msg["content"])}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            with st.chat_message("assistant", avatar="assets/images/logo-mark_light.png"):
                triage = msg.get("triage")
                res = msg.get("result") or {}
                is_emergency = (triage == "EMERGENCY") or (res.get("type") == "emergency")

                if is_emergency:
                    st.markdown(
                        f'<div class="emergency-banner-112">'
                        f'<div class="emergency-banner-lead">'
                        f'<span class="emergency-banner-badge">EMERGENCY</span>'
                        f'<span class="emergency-banner-call">CALL <strong class="emergency-banner-number">112</strong> IMMEDIATELY</span>'
                        f'</div>'
                        f'<div class="emergency-banner-text">{sanitize_html(strip_rendered_emoji_prefix(msg["content"]))}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    if triage and triage in TRIAGE_META:
                        m = TRIAGE_META[triage]
                        st.markdown(
                            f'<div class="triage-badge triage-{triage}">'
                            f'{m["icon"]} {m["label"]}</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f'<div class="bot-msg-wrap">'
                        f'<div class="bot-bubble">{sanitize_html(strip_rendered_emoji_prefix(msg["content"]))}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                if msg.get("result"):
                    render_result_panels(msg["result"], bot=bot)

    # ── Chat input ────────────────────────────────────────────
    if user_input := st.chat_input(
        "Describe your symptoms or ask a health question…"
    ):
        with st.chat_message("user"):
            st.markdown(
                f'<div class="user-msg-wrap">'
                f'<div class="user-bubble">{sanitize_html(user_input)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.session_state.messages.append({"role": "user", "content": user_input})

        # ── Bot response ──────────────────────────────────────
        with st.chat_message("assistant", avatar="assets/images/logo-mark_light.png"):
            tq            = TokenQueue()
            result_holder = []

            t = threading.Thread(
                target=run_bot_in_thread,
                args=(bot, user_input, tq, result_holder),
                daemon=True,
            )
            t.start()

            with st.spinner(""):
                stream_placeholder = st.empty()
                streamed_text = ""

                for token in tq.token_gen():
                    if any(token.startswith(p) for p in
                           ["🤔", "📖", "🩺", "🧠", "💊", "🔎", "✅", "🔧", "🔄"]):
                        continue
                    streamed_text += token
                    stream_placeholder.markdown(
                        f'<div class="bot-msg-wrap">'
                        f'<div class="bot-bubble">{sanitize_html(strip_rendered_emoji_prefix(streamed_text))}▌</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                t.join()
                result = result_holder[0] if result_holder else {}

            if result.get("type") == "service_unavailable":
                msg_text = result.get("message", SAFE_UNAVAILABLE_MESSAGE)
                stream_placeholder.markdown(
                    f'<div class="bot-msg-wrap">'
                    f'<div class="bot-bubble">{sanitize_html(strip_rendered_emoji_prefix(msg_text))}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": msg_text,
                    "triage":  None,
                    "result":  result,
                })
                return

            if result.get("type") == "error":
                err_msg = result.get("message", "Technical error")
                err_tb  = result.get("traceback", "")
                sys.stderr.write(f"[Care-AI Hosted Log] Error: {err_msg}\n{err_tb}\n")
                clean_err_msg = SAFE_UNAVAILABLE_MESSAGE
                st.error(clean_err_msg)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": clean_err_msg,
                    "triage":  None,
                    "result":  None,
                })
                return

            final_text = result.get("message", streamed_text)
            triage     = result.get("triage")
            is_emergency = (triage == "EMERGENCY") or (result.get("type") == "emergency")

            if is_emergency:
                stream_placeholder.empty()
                st.markdown(
                    f'<div class="emergency-banner-112">'
                    f'<div class="emergency-banner-lead">'
                    f'<span class="emergency-banner-badge">EMERGENCY</span>'
                    f'<span class="emergency-banner-call">CALL <strong class="emergency-banner-number">112</strong> IMMEDIATELY</span>'
                    f'</div>'
                    f'<div class="emergency-banner-text">{sanitize_html(strip_rendered_emoji_prefix(final_text))}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                if triage and triage in TRIAGE_META:
                    m = TRIAGE_META[triage]
                    st.markdown(
                        f'<div class="triage-badge triage-{triage}">'
                        f'{m["icon"]} {m["label"]}</div>',
                        unsafe_allow_html=True,
                    )

                stream_placeholder.markdown(
                    f'<div class="bot-msg-wrap">'
                    f'<div class="bot-bubble">{sanitize_html(strip_rendered_emoji_prefix(final_text))}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            render_result_panels(result, bot=bot)

        st.session_state.messages.append({
            "role":    "assistant",
            "content": final_text,
            "triage":  triage,
            "result":  result,
        })
        st.session_state.last_result = result


if __name__ == "__main__":
    main()