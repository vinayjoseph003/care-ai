"""
Care-AI — Streamlit Browser UI
Place this file at: ui/app.py
Run with: streamlit run ui/app.py
"""

import os
import sys
import queue
import threading
import streamlit as st
from dotenv import load_dotenv

# ── Path setup (so imports from project root work) ──────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.knowledge_base import build_all_chunks
from bot import CareAI
from ui.doctor_finder_panel import render_doctor_finder

load_dotenv()

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Care-AI · Medical Assistant",
    page_icon="🏥",
    layout="wide",
    # Always expanded — we reinforce this via session_state, not JS hacks
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR STATE — reliable fix using session_state, no JS needed
#
#  Strategy: track whether the user has manually collapsed the sidebar.
#  On every run we check — if they haven't deliberately closed it, we call
#  st.set_page_config with expanded=True equivalent by always writing the
#  sidebar content. Streamlit's actual collapse button still works for the
#  user when they want it, but programmatic reruns won't collapse it because
#  we never call initial_sidebar_state="collapsed" again after first load.
#
#  The key insight: initial_sidebar_state only fires on the very first load.
#  After that, Streamlit preserves the user's choice across reruns — meaning
#  the sidebar randomly collapsing is caused by full page re-mounts (e.g.
#  st.cache_resource loading). We fix this by ensuring bot loading happens
#  BEFORE the first render, so the page never fully re-mounts mid-session.
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Clinical dark theme
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,300&display=swap');

/* ── Design tokens ───────────────────────────────────────── */
:root {
  --bg:          #0a0e14;
  --surface:     #0f1520;
  --surface2:    #151d2e;
  --surface3:    #1a2540;
  --border:      #1e2d45;
  --border-hi:   #2a3f5f;
  --text:        #d4dde8;
  --text-muted:  #5a7094;
  --text-dim:    #3a5070;
  --green:       #4ade80;
  --green-dim:   #166534;
  --yellow:      #fbbf24;
  --yellow-dim:  #92400e;
  --red:         #f87171;
  --red-dim:     #7f1d1d;
  --orange:      #fb923c;
  --orange-dim:  #7c2d12;
  --blue:        #60a5fa;
  --teal:        #2dd4bf;
  --teal-dim:    #0f4c46;
  --accent:      #3b82f6;
  --accent-dim:  #1e3a5f;
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
  padding: 1.6rem 2.2rem 4rem !important;
  max-width: 1160px;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: var(--border-hi);
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
  border: 1px solid rgba(45,212,191,0.2);
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
  background: rgba(255,120,0,0.04);
  border: 1px solid rgba(255,120,0,0.1);
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
  background: linear-gradient(135deg, var(--accent-dim), rgba(59,130,246,0.2));
  border: 1px solid rgba(59,130,246,0.3);
  border-radius: 16px 16px 3px 16px;
  padding: 0.8rem 1.1rem;
  max-width: 72%;
  font-family: var(--serif);
  font-size: 0.93rem;
  line-height: 1.6;
  color: #c8d8f0;
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
  background: linear-gradient(135deg, var(--teal-dim), var(--accent-dim));
  border: 1px solid rgba(45,212,191,0.25);
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
}

/* Triage banner — inside bot bubble, above text */
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
.triage-SELF_CARE   {
  background: rgba(74,222,128,0.08);
  color: var(--green);
  border: 1px solid rgba(74,222,128,0.2);
}
.triage-SEE_DOCTOR  {
  background: rgba(251,191,36,0.08);
  color: var(--yellow);
  border: 1px solid rgba(251,191,36,0.2);
}
.triage-URGENT_CARE {
  background: rgba(251,146,60,0.08);
  color: var(--orange);
  border: 1px solid rgba(251,146,60,0.2);
}
.triage-EMERGENCY   {
  background: rgba(248,113,113,0.1);
  color: var(--red);
  border: 1px solid rgba(248,113,113,0.25);
}
/* Pulse animation for emergency */
.triage-EMERGENCY {
  animation: triage-pulse 2s ease-in-out infinite;
}
@keyframes triage-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(248,113,113,0); }
  50%       { box-shadow: 0 0 0 4px rgba(248,113,113,0.1); }
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
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1rem 1.1rem 0.85rem;
  overflow: hidden;
  transition: border-color 0.2s;
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
.diff-card.prob-high::before   { background: var(--red); }
.diff-card.prob-medium::before { background: var(--yellow); }
.diff-card.prob-low::before    { background: var(--green); }

.diff-condition {
  font-family: var(--serif);
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text);
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
  color: var(--text-muted);
  line-height: 1.5;
  margin-top: 0.45rem;
}
.diff-evidence {
  font-family: var(--mono);
  font-size: 0.67rem;
  color: var(--yellow);
  margin-top: 0.4rem;
  line-height: 1.4;
  padding: 4px 8px;
  background: rgba(251,191,36,0.05);
  border-radius: 4px;
  border-left: 2px solid rgba(251,191,36,0.3);
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
  background: var(--surface2);
  border: 1px solid var(--border);
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
  border-left: 2px solid var(--yellow);
  background: rgba(251,191,36,0.03);
}

/* ══════════════════════════════════════════════════════════
   RED FLAGS BOX
══════════════════════════════════════════════════════════ */
.redflag-box {
  background: rgba(127,29,29,0.12);
  border: 1px solid rgba(248,113,113,0.2);
  border-left: 3px solid var(--red);
  border-radius: 0 var(--radius-md) var(--radius-md) 0;
  padding: 0.9rem 1.1rem;
}
.redflag-title {
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--red);
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
  color: #fca5a5;
}
.redflag-threshold {
  margin-top: 0.55rem;
  font-family: var(--mono);
  font-size: 0.75rem;
  color: var(--red);
}

/* ══════════════════════════════════════════════════════════
   MENTAL HEALTH BANNER
══════════════════════════════════════════════════════════ */
.mh-banner {
  background: rgba(15,30,46,0.7);
  border: 1px solid rgba(96,165,250,0.2);
  border-left: 3px solid var(--blue);
  border-radius: 0 var(--radius-md) var(--radius-md) 0;
  padding: 0.8rem 1.1rem;
  font-family: var(--serif);
  font-size: 0.88rem;
  line-height: 1.6;
  color: #93c5fd;
}

/* ══════════════════════════════════════════════════════════
   XAI PANEL
══════════════════════════════════════════════════════════ */
.xai-panel {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 0.9rem 1.1rem;
}
.xai-panel p {
  font-family: var(--serif);
  font-size: 0.88rem;
  line-height: 1.65;
  color: var(--text);
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
.badge-low    { background: rgba(74,222,128,0.08); color: var(--green); border: 1px solid rgba(74,222,128,0.2); }
.badge-medium { background: rgba(251,191,36,0.08); color: var(--yellow); border: 1px solid rgba(251,191,36,0.2); }
.badge-high   { background: rgba(248,113,113,0.1); color: var(--red); border: 1px solid rgba(248,113,113,0.25); }

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
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2px 10px;
  color: var(--text-muted);
  letter-spacing: 0.04em;
}

/* ══════════════════════════════════════════════════════════
   EXPANDER OVERRIDES
══════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
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
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: var(--serif) !important;
  font-size: 0.92rem !important;
  border-radius: var(--radius-md) !important;
  padding: 0.75rem 1rem !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: rgba(59,130,246,0.5) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,0.08) !important;
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
  color: var(--text-muted);
}
.empty-state-icon {
  font-size: 2.5rem;
  margin-bottom: 1rem;
  opacity: 0.5;
}
.empty-state-title {
  font-family: var(--mono);
  font-size: 0.8rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-dim);
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
  border: 1px solid var(--border);
  border-radius: 20px;
  color: var(--text-muted);
  background: var(--surface2);
}

/* Streamlit spinner override */
[data-testid="stSpinner"] > div {
  border-color: var(--teal) transparent transparent !important;
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
    if p == "HIGH":   return "var(--red)"
    if p == "LOW":    return "var(--green)"
    return "var(--yellow)"


def _prob_label_color(prob_str: str) -> str:
    p = str(prob_str).strip().upper()
    if p == "HIGH":   return "var(--red)"
    if p == "LOW":    return "var(--green)"
    return "var(--yellow)"


def _risk_badge_class(risk: str) -> str:
    r = risk.upper()
    if r == "LOW":    return "badge-low"
    if r == "MEDIUM": return "badge-medium"
    return "badge-high"


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar(bot):
    """
    Reliable sidebar rendering.

    We render the sidebar unconditionally on every run. Streamlit preserves
    the user's open/close choice between reruns automatically once the page
    has fully mounted. The only time the sidebar disappears is when
    `st.cache_resource` triggers a full re-mount; to prevent that we load
    the bot BEFORE the first render (see init_session below) so the cache
    is always warm by render time.
    """
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

        # ── Session metrics ───────────────────────────────────
        turns = len([m for m in st.session_state.messages
                     if m["role"] == "user"])
        st.markdown(f"""
<div class="sb-metric">
  <div class="sb-metric-label">Session Turns</div>
  <div class="sb-metric-value sv-teal">{turns}</div>
</div>""", unsafe_allow_html=True)

        lang = getattr(bot, 'user_language', {}).get('language', '—')
        st.markdown(f"""
<div class="sb-metric">
  <div class="sb-metric-label">Detected Language</div>
  <div class="sb-metric-value sv-blue">{lang}</div>
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

        top = getattr(bot, 'last_top_condition', '')
        if top:
            st.markdown(f"""
<div class="sb-metric">
  <div class="sb-metric-label">Top Diagnosis</div>
  <div class="sb-metric-value sv-teal">{top}</div>
</div>""", unsafe_allow_html=True)

        spec = getattr(bot, 'last_specialist', '')
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
            # Clear doctor finder state too
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
            f'<div class="diff-desc">{description}</div>'
            if description else ''
        )
        evidence_html = (
            f'<div class="diff-evidence">🔑 {key_evidence}</div>'
            if key_evidence else ''
        )

        with cols[i]:
            st.markdown(f"""
<div class="diff-card {css_class}">
  <div class="diff-rank-badge">#{d.get('rank', i+1)}</div>
  <div class="diff-condition">{d.get('condition', 'Unknown')}</div>
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
        return "".join(f"<li>{i}</li>" for i in items) if items else "<li>—</li>"

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
  <p class="tc-escalation-text">{esc}</p>
</div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_red_flags(red_flags: dict):
    if not red_flags or not red_flags.get("red_flags"):
        return
    flags     = red_flags.get("red_flags", [])
    threshold = red_flags.get("emergency_threshold", "")
    items_html = "".join(f"<li>{f}</li>" for f in flags)
    st.markdown(f"""
<div class="redflag-box">
  <div class="redflag-title">🚩 Warning Signs — Seek Immediate Care If You Notice:</div>
  <ul class="redflag-list">{items_html}</ul>
  {f'<div class="redflag-threshold">{threshold}</div>' if threshold else ''}
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
                f'<div class="xai-panel"><p>{xai}</p></div>',
                unsafe_allow_html=True
            )

    if mh_msg:
        st.markdown(f'<div class="mh-banner">💙 {mh_msg}</div>',
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
            f'<span class="source-pill">{s}</span>' for s in sources
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
#  SESSION STATE INIT
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Care-AI knowledge base…")
def load_bot():
    router_config = {
        "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "openrouter_model":   os.environ.get("OPENROUTER_MODEL",   "meta-llama/llama-3.3-70b-instruct:free"),
        "groq_api_key":       os.environ.get("GROQ_API_KEY",       ""),
        "groq_model":         os.environ.get("GROQ_MODEL",         "llama-3.3-70b-versatile"),
        "gemini_api_key":     os.environ.get("GEMINI_API_KEY",     ""),
        "gemini_model":       os.environ.get("GEMINI_MODEL",       "gemini-1.5-flash"),
        "ollama_model":       os.environ.get("OLLAMA_MODEL",       "qwen2.5:7b"),
        "ollama_host":        os.environ.get("OLLAMA_HOST",        "http://localhost:11434"),
    }
    # Require at least one cloud provider key
    has_cloud = any([
        router_config["openrouter_api_key"],
        router_config["groq_api_key"],
        router_config["gemini_api_key"],
    ])
    if not has_cloud:
        st.error(
            "❌ No API key found. Add at least one of these to your .env:\n"
            "OPENROUTER_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY"
        )
        st.stop()
    all_chunks, severity_map, precaution_map = build_all_chunks(data_dir="data")
    bot = CareAI(
        knowledge_chunks=all_chunks,
        severity_map=severity_map,
        router_config=router_config,
        precaution_map=precaution_map,
    )
    return bot


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load bot FIRST — before any render calls.
    # This ensures st.cache_resource never triggers a mid-render re-mount,
    # which is the root cause of the sidebar disappearing on first message.
    bot = load_bot()

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
                    f'<div class="user-bubble">{msg["content"]}</div>'
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
                    f'<div class="bot-bubble">{msg["content"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if msg.get("result"):
                    render_result_panels(msg["result"], bot=bot)

    # ── Chat input ────────────────────────────────────────────
    if user_input := st.chat_input(
        "Describe your symptoms or ask a health question…"
    ):
        # Render user bubble immediately
        with st.chat_message("user"):
            st.markdown(
                f'<div class="user-msg-wrap">'
                f'<div class="user-bubble">{user_input}</div>'
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

            # Live streaming
            with st.spinner(""):
                stream_placeholder = st.empty()
                streamed_text = ""

                for token in tq.token_gen():
                    # Filter pipeline status lines
                    if any(token.startswith(p) for p in
                           ["🤔", "📖", "🩺", "🧠", "💊", "🔎", "✅", "🔧", "🔄"]):
                        continue
                    streamed_text += token
                    stream_placeholder.markdown(
                        f'<div class="bot-msg-wrap">'
                        f'<div class="bot-avatar">🏥</div>'
                        f'<div class="bot-bubble">{streamed_text}▌</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                t.join()
                result = result_holder[0] if result_holder else {}

            # ── Error in bot thread ───────────────────────────
            if result.get("type") == "error":
                err_msg = result.get("message", "Unknown error")
                err_tb  = result.get("traceback", "")
                st.error(f"Bot error: {err_msg}")
                if err_tb:
                    with st.expander("Full traceback (for debugging)"):
                        st.code(err_tb)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": f"Something went wrong: {err_msg}",
                    "triage":  None,
                    "result":  None,
                })
                return

            final_text = result.get("message", streamed_text)
            triage     = result.get("triage")

            # Triage badge
            if triage and triage in TRIAGE_META:
                m = TRIAGE_META[triage]
                st.markdown(
                    f'<div class="triage-badge triage-{triage}">'
                    f'{m["icon"]} {m["label"]}</div>',
                    unsafe_allow_html=True,
                )

            # Final text (replaces streaming cursor)
            stream_placeholder.markdown(
                f'<div class="bot-msg-wrap">'
                f'<div class="bot-avatar">🏥</div>'
                f'<div class="bot-bubble">{final_text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            render_result_panels(result, bot=bot)

        # Store in history
        st.session_state.messages.append({
            "role":    "assistant",
            "content": final_text,
            "triage":  triage,
            "result":  result,
        })
        st.session_state.last_result = result


if __name__ == "__main__":
    main()