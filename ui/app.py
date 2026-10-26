"""
Care-AI — Streamlit Browser UI
Place this file at: ui/app.py
Run with: streamlit run ui/app.py
"""

import os
import sys
import queue
import html
import threading
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


def sanitize_html(text: str, preserve_newlines: bool = True) -> str:
    """Safely escape user and model text before inserting into HTML templates."""
    if not text:
        return ""
    escaped = html.escape(str(text))
    if preserve_newlines:
        return escaped.replace("\n", "<br>")
    return escaped


# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Care-AI · Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Light professional medical theme
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,300&display=swap');

/* ── Design tokens ───────────────────────────────────────── */
:root {
  --bg:          #F0F4F8;
  --surface:     #FFFFFF;
  --surface2:    #F7FAFC;
  --surface3:    #EDF2F7;
  --border:      #CBD5E0;
  --border-hi:   #A0AEC0;
  --text:        #1A202C;
  --text-muted:  #4A5568;
  --text-dim:    #718096;
  --green:       #16A34A;
  --green-dim:   #DCFCE7;
  --yellow:      #D97706;
  --yellow-dim:  #FEF3C7;
  --red:         #DC2626;
  --red-dim:     #FEE2E2;
  --orange:      #EA580C;
  --orange-dim:  #FFEDD5;
  --blue:        #2563EB;
  --blue-dim:    #DBEAFE;
  --teal:        #0891B2;
  --teal-dim:    #CFFAFE;
  --accent:      #2563EB;
  --accent-dim:  #DBEAFE;
  --mono:        'IBM Plex Mono', 'Courier New', monospace;
  --serif:       'Source Serif 4', Georgia, serif;
  --radius-sm:   6px;
  --radius-md:   10px;
  --radius-lg:   14px;
}

/* ── Global reset ────────────────────────────────────────── */
html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--serif) !important;
}

/* ── Hide Streamlit chrome ───────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  background: transparent !important;
  padding: 1.6rem 2.2rem 4rem !important;
  max-width: 1160px;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: #CBD5E0;
  border-radius: 3px;
}

/* ══════════════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
  min-width: 248px !important;
  max-width: 300px !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] > div:first-child {
  overflow-y: auto !important;
  padding: 1.1rem 1rem !important;
}

/* Sidebar brand lockup */
.sb-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0.2rem 0 1.2rem;
  margin-bottom: 1rem;
  border-bottom: 1px solid var(--border);
}
.sb-brand-icon {
  width: 36px; height: 36px;
  background: linear-gradient(135deg, var(--teal-dim), var(--accent-dim));
  border: 1px solid rgba(8,145,178,0.2);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
}
.sb-brand-name {
  font-family: var(--mono);
  font-size: 0.9rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  color: var(--teal) !important;
  line-height: 1.1;
}
.sb-brand-ver {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--text-muted) !important;
  letter-spacing: 0.08em;
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
  font-family: var(--mono);
  font-size: 0.58rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--teal) !important;
  margin-bottom: 0.15rem;
}
.sb-lang-value {
  font-family: var(--mono);
  font-size: 1rem;
  font-weight: 700;
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
  font-family: var(--mono);
  font-size: 0.58rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-muted) !important;
  margin-bottom: 0.25rem;
}
.sb-metric-value {
  font-family: var(--mono);
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--text) !important;
}
.sv-teal   { color: var(--teal)   !important; }
.sv-green  { color: var(--green)  !important; }
.sv-yellow { color: var(--yellow) !important; }
.sv-orange { color: var(--orange) !important; }
.sv-red    { color: var(--red)    !important; }
.sv-blue   { color: var(--blue)   !important; }

/* Sidebar section divider */
.sb-section {
  font-family: var(--mono);
  font-size: 0.57rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--text-dim) !important;
  margin: 1rem 0 0.5rem;
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
  font-family: var(--mono);
  font-size: 0.58rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted) !important;
}
.sb-progress-count {
  font-family: var(--mono);
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--teal) !important;
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
  font-family: var(--serif);
  font-size: 0.73rem;
  color: var(--text-muted) !important;
  line-height: 1.55;
  padding: 0.7rem 0.9rem;
  background: rgba(217,119,6,0.05);
  border: 1px solid rgba(217,119,6,0.15);
  border-radius: var(--radius-sm);
  margin-top: 0.5rem;
}

/* Streamlit button overrides inside sidebar */
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
  font-family: var(--mono);
  font-size: 1.25rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  color: var(--teal) !important;
  line-height: 1.1;
  margin: 0;
}
.main-subtitle {
  font-family: var(--serif);
  font-size: 0.82rem;
  color: var(--text-muted);
  margin-top: 0.3rem;
  font-style: italic;
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

/* User bubble — right-aligned */
.user-msg-wrap {
  display: flex;
  justify-content: flex-end;
  margin: 0.6rem 0;
}
.user-bubble {
  background: var(--accent);
  border: none;
  border-radius: 16px 16px 3px 16px;
  padding: 0.8rem 1.1rem;
  max-width: 72%;
  font-family: var(--serif);
  font-size: 0.93rem;
  line-height: 1.6;
  color: #FFFFFF;
}

/* Bot message container */
.bot-msg-wrap {
  display: flex;
  align-items: flex-start;
  gap: 0.8rem;
  margin: 0.6rem 0;
}
.bot-avatar {
  width: 32px; height: 32px;
  background: var(--teal-dim);
  border: 1.5px solid var(--teal);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.9rem;
  flex-shrink: 0;
  margin-top: 3px;
}
.bot-bubble {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 3px 16px 16px 16px;
  padding: 0.9rem 1.2rem;
  max-width: 84%;
  font-family: var(--serif);
  font-size: 0.93rem;
  line-height: 1.65;
  color: var(--text);
  flex: 1;
  box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}

/* Triage banner */
.triage-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--mono);
  font-size: 0.62rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 4px 12px;
  border-radius: 4px;
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
@keyframes triage-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
  50%       { box-shadow: 0 0 0 4px rgba(220,38,38,0.12); }
}

/* ══════════════════════════════════════════════════════════
   RESULT PANELS — section label
══════════════════════════════════════════════════════════ */
.panel-section-label {
  font-family: var(--mono);
  font-size: 0.58rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--text-dim);
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
  font-family: var(--mono);
  font-size: 0.58rem;
  letter-spacing: 0.1em;
  color: var(--text-dim);
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
  font-family: var(--mono);
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.08em;
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
  font-family: var(--mono);
  font-size: 0.67rem;
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
  font-family: var(--mono);
  font-size: 0.6rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
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
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
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
  font-family: var(--mono);
  font-size: 0.75rem;
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
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.06em;
  padding: 3px 9px;
  border-radius: 4px;
  font-weight: 500;
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
  font-family: var(--mono);
  font-size: 0.63rem;
  background: #F7FAFC;
  border: 1px solid #CBD5E0;
  border-radius: 20px;
  padding: 2px 10px;
  color: #4A5568;
  letter-spacing: 0.04em;
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
  font-family: var(--mono) !important;
  font-size: 0.73rem !important;
  letter-spacing: 0.08em !important;
  color: var(--text-muted) !important;
  padding: 0.55rem 0.8rem !important;
}
[data-testid="stExpander"] summary:hover {
  color: var(--text) !important;
}
[data-testid="stExpander"] > div[data-testid="stExpanderDetails"] {
  padding: 0.3rem 0.8rem 0.8rem !important;
}

/* ══════════════════════════════════════════════════════════
   CHAT INPUT
══════════════════════════════════════════════════════════ */
[data-testid="stChatInput"] {
  border-top: 1px solid var(--border) !important;
  padding-top: 0.8rem !important;
}
[data-testid="stChatInput"] textarea {
  background: #FFFFFF !important;
  border: 1.5px solid #CBD5E0 !important;
  color: var(--text) !important;
  font-family: var(--serif) !important;
  font-size: 0.92rem !important;
  border-radius: var(--radius-md) !important;
  padding: 0.75rem 1rem !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px rgba(8,145,178,0.08) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: #718096 !important;
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
  border: 1px solid #CBD5E0;
  border-radius: var(--radius-lg);
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.empty-state-icon {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  opacity: 0.6;
}
.empty-state-title {
  font-family: var(--mono);
  font-size: 0.8rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #1A202C;
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
  font-family: var(--mono);
  font-size: 0.65rem;
  padding: 3px 10px;
  border: 1px solid #0891B2;
  border-radius: 20px;
  color: #0891B2;
  background: #CFFAFE;
}

/* Streamlit spinner override */
[data-testid="stSpinner"] > div {
  border-color: var(--teal) transparent transparent !important;
}

/* ══════════════════════════════════════════════════════════
   SIDEBAR COLLAPSE / EXPAND BUTTON — make it visible on light bg
══════════════════════════════════════════════════════════ */
[data-testid="collapsedControl"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 var(--radius-md) var(--radius-md) 0 !important;
  color: var(--text) !important;
}
[data-testid="collapsedControl"]:hover {
  background: var(--surface3) !important;
  border-color: var(--border-hi) !important;
}
[data-testid="collapsedControl"] svg {
  fill: var(--text) !important;
  stroke: var(--text) !important;
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
  fill: var(--text) !important;
  stroke: var(--text) !important;
}

/* Ensure sidebar never has a dark overlay */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
}
section[data-testid="stSidebar"]::before {
  display: none !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  TRIAGE METADATA
# ════════════════════════════════════════════════════════════════════════════

TRIAGE_META = {
    "SELF_CARE":   {"icon": "●", "label": "SELF-CARE",    "css": "triage-SELF_CARE"},
    "SEE_DOCTOR":  {"icon": "●", "label": "SEE A DOCTOR", "css": "triage-SEE_DOCTOR"},
    "URGENT_CARE": {"icon": "●", "label": "URGENT CARE",  "css": "triage-URGENT_CARE"},
    "EMERGENCY":   {"icon": "▲", "label": "EMERGENCY",    "css": "triage-EMERGENCY"},
}

SOCRATES_DIMS = [
    "symptom_or_topic", "onset", "character", "radiation",
    "associations", "timing", "exacerbating", "severity",
]


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _prob_to_pct(prob_str) -> int:
    """Convert HIGH/MEDIUM/LOW, '75%', 0.75, or 75 → int 0-100."""
    _RANK_MAP = {"HIGH": 78, "MEDIUM": 48, "LOW": 22}
    s = str(prob_str).strip().upper()
    if s in _RANK_MAP:
        return _RANK_MAP[s]
    try:
        s = s.replace('%', '')
        f = float(s)
        return int(f) if f > 1 else int(f * 100)
    except Exception:
        return 50


def _prob_css_class(prob_str: str) -> str:
    p = str(prob_str).strip().upper()
    if p == "HIGH":   return "prob-high"
    if p == "LOW":    return "prob-low"
    return "prob-medium"


def _prob_fill_color(prob_str: str) -> str:
    p = str(prob_str).strip().upper()
    if p == "HIGH":   return "#DC2626"
    if p == "LOW":    return "#16A34A"
    return "#D97706"


def _prob_label_color(prob_str: str) -> str:
    p = str(prob_str).strip().upper()
    if p == "HIGH":   return "#DC2626"
    if p == "LOW":    return "#16A34A"
    return "#D97706"


def _risk_badge_class(risk: str) -> str:
    r = risk.upper()
    if r == "LOW":    return "badge-low"
    if r == "MEDIUM": return "badge-medium"
    return "badge-high"


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar(bot):
    with st.sidebar:
        # Brand
        st.markdown("""
<div class="sb-brand">
  <div class="sb-brand-icon">🏥</div>
  <div>
    <div class="sb-brand-name">CARE-AI</div>
    <div class="sb-brand-ver">v2.0 · MULTILINGUAL XAI</div>
  </div>
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

        top = sanitize_html(getattr(bot, 'last_top_condition', ''), preserve_newlines=False)
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

        if st.button("📤  Export Log", use_container_width=True):
            bot.export_log()
            st.success("Log saved → logs/session_log.json")

        # ── Disclaimer ────────────────────────────────────────
        st.markdown("""
<div class="sb-disclaimer">
  ⚠️ Care-AI provides general health information only.
  Always consult a qualified doctor before acting on this advice.
  In an emergency, call <strong>112</strong>.
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
    <div class="tc-icon">🏠</div>
    <div class="tc-title">Home Care</div>
  </div>
  <ul class="tc-list">{_li(home)}</ul>
</div>""", unsafe_allow_html=True)

    if life:
        st.markdown(f"""
<div class="treatment-card">
  <div class="tc-header">
    <div class="tc-icon">🌿</div>
    <div class="tc-title">Lifestyle Tips</div>
  </div>
  <ul class="tc-list">{_li(life)}</ul>
</div>""", unsafe_allow_html=True)

    if avoid:
        st.markdown(f"""
<div class="treatment-card full-width">
  <div class="tc-header">
    <div class="tc-icon">🚫</div>
    <div class="tc-title">What to Avoid</div>
  </div>
  <ul class="tc-list">{_li(avoid)}</ul>
</div>""", unsafe_allow_html=True)

    if esc:
        st.markdown(f"""
<div class="treatment-card full-width escalation">
  <div class="tc-header">
    <div class="tc-icon">⏱</div>
    <div class="tc-title">Escalation Timeline</div>
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

    # ── Reliability badge ─────────────────────────────────────
    _risk_up    = str(risk).upper()
    _verdict_up = str(verdict).upper()
    if _risk_up == "LOW" and _verdict_up == "VERIFIED":
        _icon, _label, _cls = "✅", "Well-supported", "badge-low"
    elif _risk_up == "HIGH" or _verdict_up in ("UNVERIFIED", "FAILED"):
        _icon, _label, _cls = "⚠️", "Verify with doctor", "badge-high"
    else:
        _icon, _label, _cls = "🟡", "Confirm with doctor", "badge-medium"

    st.markdown(
        f'<div class="badge-row"><span class="badge {_cls}">{_icon} {_label}</span></div>',
        unsafe_allow_html=True
    )

    # ── Sources ───────────────────────────────────────────────
    if sources:
        pills = "".join(
            f'<span class="source-pill">{sanitize_html(s, False)}</span>' for s in sources
        )
        st.markdown(
            f'<div class="sources-wrap">'
            f'<div class="panel-section-label">Sources</div>'
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
    col_title, col_btn = st.columns([6, 1])
    with col_title:
        st.markdown("""
<div class="main-header">
  <div class="main-header-left">
    <div class="main-title">🏥 CARE-AI</div>
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
            with st.chat_message("assistant", avatar="🏥"):
                triage = msg.get("triage")
                if triage and triage in TRIAGE_META:
                    m = TRIAGE_META[triage]
                    st.markdown(
                        f'<div class="triage-badge triage-{triage}">'
                        f'{m["icon"]} {m["label"]}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div class="bot-msg-wrap">'
                    f'<div class="bot-avatar">🏥</div>'
                    f'<div class="bot-bubble">{sanitize_html(msg["content"])}</div>'
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
        with st.chat_message("assistant", avatar="🏥"):
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
                        f'<div class="bot-avatar">🏥</div>'
                        f'<div class="bot-bubble">{sanitize_html(streamed_text)}▌</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                t.join()
                result = result_holder[0] if result_holder else {}

            if result.get("type") == "service_unavailable":
                msg_text = result.get("message", SAFE_UNAVAILABLE_MESSAGE)
                stream_placeholder.markdown(
                    f'<div class="bot-msg-wrap">'
                    f'<div class="bot-avatar">🏥</div>'
                    f'<div class="bot-bubble">{sanitize_html(msg_text)}</div>'
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

            if triage and triage in TRIAGE_META:
                m = TRIAGE_META[triage]
                st.markdown(
                    f'<div class="triage-badge triage-{triage}">'
                    f'{m["icon"]} {m["label"]}</div>',
                    unsafe_allow_html=True,
                )

            stream_placeholder.markdown(
                f'<div class="bot-msg-wrap">'
                f'<div class="bot-avatar">🏥</div>'
                f'<div class="bot-bubble">{sanitize_html(final_text)}</div>'
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