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

load_dotenv()

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Care-AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
#  CUSTOM CSS — Clinical dark theme, IBM Plex Mono + Source Serif 4
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,300&display=swap');

/* ── Root tokens ─────────────────────────────────────────── */
:root {
  --bg:         #0d1117;
  --surface:    #161b22;
  --surface2:   #1c2333;
  --border:     #30363d;
  --text:       #e6edf3;
  --text-muted: #7d8590;
  --green:      #3fb950;
  --yellow:     #d29922;
  --red:        #f85149;
  --orange:     #fb8500;
  --blue:       #58a6ff;
  --teal:       #39d0d8;
  --accent:     #388bfd;
  --mono:       'IBM Plex Mono', monospace;
  --serif:      'Source Serif 4', Georgia, serif;
}

/* ── Global reset ────────────────────────────────────────── */
html, body, [class*="css"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--serif) !important;
}

/* ── Hide Streamlit chrome ───────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1200px; }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Sidebar brand ───────────────────────────────────────── */
.brand {
  font-family: var(--mono);
  font-size: 1.1rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  color: var(--teal) !important;
  padding: 0.5rem 0 1.2rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.2rem;
}
.brand span { color: var(--text-muted); font-weight: 400; }

/* ── Metric cards (sidebar) ──────────────────────────────── */
.metric-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.7rem 1rem;
  margin-bottom: 0.6rem;
  font-family: var(--mono);
}
.metric-card .mc-label {
  font-size: 0.62rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
}
.metric-card .mc-value {
  font-size: 0.95rem;
  font-weight: 600;
  margin-top: 0.2rem;
  color: var(--text);
}
.mc-green  { color: var(--green)  !important; }
.mc-yellow { color: var(--yellow) !important; }
.mc-red    { color: var(--red)    !important; }
.mc-orange { color: var(--orange) !important; }
.mc-blue   { color: var(--blue)   !important; }
.mc-teal   { color: var(--teal)   !important; }

/* ── Chat area ───────────────────────────────────────────── */
.chat-container {
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
  padding-bottom: 1rem;
}

/* ── User bubble ─────────────────────────────────────────── */
.user-bubble {
  display: flex;
  justify-content: flex-end;
}
.user-bubble .bubble-inner {
  background: var(--accent);
  color: #fff;
  border-radius: 16px 16px 4px 16px;
  padding: 0.75rem 1.1rem;
  max-width: 70%;
  font-family: var(--serif);
  font-size: 0.95rem;
  line-height: 1.55;
}

/* ── Bot bubble ──────────────────────────────────────────── */
.bot-bubble {
  display: flex;
  align-items: flex-start;
  gap: 0.75rem;
}
.bot-avatar {
  width: 34px; height: 34px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem;
  flex-shrink: 0;
  margin-top: 2px;
}
.bot-bubble .bubble-inner {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px 16px 16px 16px;
  padding: 0.85rem 1.2rem;
  max-width: 82%;
  font-family: var(--serif);
  font-size: 0.95rem;
  line-height: 1.65;
  color: var(--text);
}

/* ── Triage banner ───────────────────────────────────────── */
.triage-banner {
  font-family: var(--mono);
  font-size: 0.75rem;
  letter-spacing: 0.1em;
  font-weight: 600;
  padding: 0.45rem 1rem;
  border-radius: 6px;
  margin-bottom: 0.6rem;
  display: inline-block;
}
.triage-SELF_CARE   { background: #1a3a1f; color: var(--green);  border: 1px solid #2d6a35; }
.triage-SEE_DOCTOR  { background: #3a2e0f; color: var(--yellow); border: 1px solid #6a5415; }
.triage-URGENT_CARE { background: #3a1414; color: var(--orange); border: 1px solid #8b3a20; }
.triage-EMERGENCY   { background: #3a0d0d; color: var(--red);    border: 1px solid #8b2020; }

/* ── Diff diagnosis cards ────────────────────────────────── */
.diff-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.75rem;
  margin-top: 0.8rem;
}
.diff-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.85rem 1rem;
  font-family: var(--mono);
}
.diff-rank {
  font-size: 0.6rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--text-muted);
}
.diff-condition {
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--teal);
  margin: 0.3rem 0 0.4rem 0;
  line-height: 1.3;
}
.diff-prob {
  font-size: 0.78rem;
  color: var(--text-muted);
}
.prob-bar {
  height: 3px;
  background: var(--border);
  border-radius: 2px;
  margin-top: 0.5rem;
  overflow: hidden;
}
.prob-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--teal), var(--accent));
  border-radius: 2px;
}
/* [P1-D] Description + key evidence inside differential cards */
.diff-desc {
  font-family: var(--serif);
  font-size: 0.78rem;
  color: var(--text-muted);
  margin-top: 0.55rem;
  line-height: 1.5;
}
.diff-evidence {
  font-family: var(--mono);
  font-size: 0.7rem;
  color: var(--yellow);
  margin-top: 0.4rem;
  line-height: 1.4;
}

/* ── Section labels ──────────────────────────────────────── */
.section-label {
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 1rem 0 0.4rem 0;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid var(--border);
}

/* ── Treatment cards ─────────────────────────────────────── */
.treatment-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.7rem;
  margin-top: 0.5rem;
}
.treatment-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.8rem 1rem;
}
.treatment-card .tc-title {
  font-family: var(--mono);
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 0.5rem;
}
.treatment-card ul {
  margin: 0; padding-left: 1.1rem;
  font-size: 0.87rem;
  line-height: 1.6;
  color: var(--text);
}

/* ── Red flag box ────────────────────────────────────────── */
.redflag-box {
  background: #1e0f0f;
  border: 1px solid #5c1a1a;
  border-left: 3px solid var(--red);
  border-radius: 8px;
  padding: 0.8rem 1rem;
  margin-top: 0.5rem;
}
.redflag-box .rf-title {
  font-family: var(--mono);
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--red);
  margin-bottom: 0.5rem;
}
.redflag-box ul {
  margin: 0; padding-left: 1.1rem;
  font-size: 0.87rem;
  line-height: 1.6;
  color: #f5c6cb;
}
.redflag-threshold {
  margin-top: 0.5rem;
  font-size: 0.82rem;
  color: var(--red);
  font-family: var(--mono);
}

/* ── Mental health banner ────────────────────────────────── */
.mh-banner {
  background: #0f1e2e;
  border: 1px solid #1a4a6e;
  border-left: 3px solid var(--blue);
  border-radius: 8px;
  padding: 0.75rem 1rem;
  margin-top: 0.6rem;
  font-size: 0.9rem;
  line-height: 1.55;
  color: #a8d8f0;
}

/* ── Sources ─────────────────────────────────────────────── */
.sources-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.4rem;
}
.source-pill {
  font-family: var(--mono);
  font-size: 0.68rem;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 0.2rem 0.65rem;
  color: var(--text-muted);
}

/* ── Confidence + fact-check badges ──────────────────────── */
.badge-row {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-top: 0.5rem;
}
.badge {
  font-family: var(--mono);
  font-size: 0.68rem;
  padding: 0.22rem 0.6rem;
  border-radius: 4px;
  font-weight: 500;
}
.badge-conf   { background: #1a2f1a; color: var(--green);  border: 1px solid #2d5c2d; }
.badge-risk-low    { background: #1a2f1a; color: var(--green);  border: 1px solid #2d5c2d; }
.badge-risk-medium { background: #2f2a1a; color: var(--yellow); border: 1px solid #5c4d1a; }
.badge-risk-high   { background: #2f1a1a; color: var(--red);    border: 1px solid #5c1a1a; }

/* ── Input area ──────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: var(--serif) !important;
  border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(56,139,253,0.15) !important;
}

/* ── Expander ────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
  font-family: var(--mono) !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.08em !important;
  color: var(--text-muted) !important;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Streamlit chat message overrides ────────────────────── */
[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS — Triage colors, probability parsing
# ════════════════════════════════════════════════════════════════════════════

TRIAGE_META = {
    "SELF_CARE":   {"icon": "🟢", "label": "SELF-CARE",    "css": "triage-SELF_CARE"},
    "SEE_DOCTOR":  {"icon": "🟡", "label": "SEE A DOCTOR", "css": "triage-SEE_DOCTOR"},
    "URGENT_CARE": {"icon": "🔴", "label": "URGENT CARE",  "css": "triage-URGENT_CARE"},
    "EMERGENCY":   {"icon": "🚨", "label": "EMERGENCY",    "css": "triage-EMERGENCY"},
}

def _prob_to_pct(prob_str) -> int:
    """Convert 'HIGH'/'MEDIUM'/'LOW', '75%', 0.75, or 75 → int 0-100.

    REASONING_PROMPT returns probability as a rank label (HIGH/MEDIUM/LOW),
    not a numeric percentage. Map those to fixed display values so the
    progress bars reflect actual rank separation instead of all showing 50%.
    """
    _RANK_MAP = {"HIGH": 75, "MEDIUM": 45, "LOW": 20}
    s = str(prob_str).strip().upper()
    if s in _RANK_MAP:
        return _RANK_MAP[s]
    try:
        s = s.replace('%', '')
        f = float(s)
        return int(f) if f > 1 else int(f * 100)
    except Exception:
        return 50

def _risk_badge_class(risk: str) -> str:
    r = risk.upper()
    if r == "LOW":    return "badge-risk-low"
    if r == "MEDIUM": return "badge-risk-medium"
    return "badge-risk-high"


# ════════════════════════════════════════════════════════════════════════════
#  RENDER HELPERS
# ════════════════════════════════════════════════════════════════════════════

def render_sidebar(bot):
    with st.sidebar:
        st.markdown('<div class="brand">🏥 CARE-AI <span>v2.0</span></div>',
                    unsafe_allow_html=True)

        # Turn count
        turns = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f'''
        <div class="metric-card">
          <div class="mc-label">Session Turns</div>
          <div class="mc-value mc-teal">{turns}</div>
        </div>''', unsafe_allow_html=True)

        # Language
        lang = getattr(bot, 'user_language', {}).get('language', 'English')
        st.markdown(f'''
        <div class="metric-card">
          <div class="mc-label">Detected Language</div>
          <div class="mc-value mc-blue">{lang}</div>
        </div>''', unsafe_allow_html=True)

        # Triage
        triage = getattr(bot, 'last_triage', None)
        if triage and triage in TRIAGE_META:
            m = TRIAGE_META[triage]
            color = {"SELF_CARE":"mc-green","SEE_DOCTOR":"mc-yellow",
                     "URGENT_CARE":"mc-orange","EMERGENCY":"mc-red"}.get(triage,"mc-blue")
            st.markdown(f'''
            <div class="metric-card">
              <div class="mc-label">Triage Level</div>
              <div class="mc-value {color}">{m["icon"]} {m["label"]}</div>
            </div>''', unsafe_allow_html=True)

        # Top diagnosis
        top = getattr(bot, 'last_top_condition', '')
        if top:
            st.markdown(f'''
            <div class="metric-card">
              <div class="mc-label">Top Diagnosis</div>
              <div class="mc-value mc-teal">{top}</div>
            </div>''', unsafe_allow_html=True)

        # Specialist
        spec = getattr(bot, 'last_specialist', '')
        if spec:
            st.markdown(f'''
            <div class="metric-card">
              <div class="mc-label">Suggested Specialist</div>
              <div class="mc-value">{spec}</div>
            </div>''', unsafe_allow_html=True)

        # SOCRATES progress
        socrates_dims = ["symptom_or_topic","onset","character","radiation",
                         "associations","timing","exacerbating","severity"]
        collected = getattr(bot, 'collected_info', {})
        filled = sum(1 for d in socrates_dims
                     if collected.get(d) and str(collected[d]).strip()
                     not in ('','...','null','None'))
        st.markdown('<div class="section-label">SOCRATES Intake</div>',
                    unsafe_allow_html=True)
        st.progress(filled / len(socrates_dims),
                    text=f"{filled}/{len(socrates_dims)} fields collected")

        st.markdown("---")
        st.markdown('<div class="section-label">Commands</div>',
                    unsafe_allow_html=True)
        if st.button("🔄 Reset Session", use_container_width=True):
            bot.reset()
            st.session_state.messages = []
            st.session_state.last_result = None
            st.rerun()
        if st.button("📤 Export Log", use_container_width=True):
            bot.export_log()
            st.success("Log exported → logs/session_log.json")

        st.markdown("---")
        st.markdown(
            '<div style="font-family:var(--mono);font-size:0.65rem;'
            'color:var(--text-muted);line-height:1.6;">'
            '⚠️ Care-AI is for informational purposes only.<br>'
            'Always consult a qualified medical professional.</div>',
            unsafe_allow_html=True
        )


def render_differential(differential: list):
    if not differential:
        return
    st.markdown('<div class="section-label">Differential Diagnosis</div>',
                unsafe_allow_html=True)

    # Colour the probability label by rank
    _PROB_COLOURS = {"HIGH": "var(--red)", "MEDIUM": "var(--yellow)", "LOW": "var(--green)"}

    cols = st.columns(min(len(differential), 3))
    for i, d in enumerate(differential[:3]):
        prob_raw  = str(d.get('probability', 'MEDIUM')).upper()
        pct       = _prob_to_pct(prob_raw)
        prob_color = _PROB_COLOURS.get(prob_raw, "var(--text-muted)")

        # [P1-D] description + key_evidence fields from REASONING_PROMPT
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
            <div class="diff-card">
              <div class="diff-rank">#{d.get('rank', i+1)} Possibility</div>
              <div class="diff-condition">{d.get('condition','Unknown')}</div>
              <div class="diff-prob" style="color:{prob_color};font-weight:600">
                {prob_raw}
              </div>
              <div class="prob-bar">
                <div class="prob-fill" style="width:{pct}%"></div>
              </div>
              {desc_html}
              {evidence_html}
            </div>""", unsafe_allow_html=True)


def render_treatment(treatment: dict):
    if not treatment:
        return
    home   = treatment.get("home_care", [])
    esc    = treatment.get("escalation_timeline", "")
    life   = treatment.get("lifestyle_tips", [])
    avoid  = treatment.get("what_to_avoid", [])

    def _li(items):
        return "".join(f"<li>{i}</li>" for i in items)

    cols = st.columns(2)
    with cols[0]:
        st.markdown(f"""
        <div class="treatment-card">
          <div class="tc-title">🏠 Home Care</div>
          <ul>{_li(home)}</ul>
        </div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown(f"""
        <div class="treatment-card">
          <div class="tc-title">🌿 Lifestyle Tips</div>
          <ul>{_li(life)}</ul>
        </div>""", unsafe_allow_html=True)
    if avoid:
        st.markdown(f"""
        <div class="treatment-card" style="margin-top:0.5rem">
          <div class="tc-title">🚫 What to Avoid</div>
          <ul>{_li(avoid)}</ul>
        </div>""", unsafe_allow_html=True)
    if esc:
        st.markdown(f"""
        <div class="treatment-card" style="margin-top:0.5rem;
             border-left:3px solid var(--yellow);">
          <div class="tc-title">⏱ Escalation Timeline</div>
          <p style="margin:0;font-size:0.87rem;color:var(--text);line-height:1.55">
            {esc}
          </p>
        </div>""", unsafe_allow_html=True)


def render_red_flags(red_flags: dict):
    if not red_flags or not red_flags.get("red_flags"):
        return
    flags = red_flags.get("red_flags", [])
    threshold = red_flags.get("emergency_threshold", "")
    items_html = "".join(f"<li>{f}</li>" for f in flags)
    st.markdown(f"""
    <div class="redflag-box">
      <div class="rf-title">🚩 Red Flag Symptoms — Seek Immediate Care If You Notice:</div>
      <ul>{items_html}</ul>
      {f'<div class="redflag-threshold">{threshold}</div>' if threshold else ''}
    </div>""", unsafe_allow_html=True)


def render_result_panels(result: dict):
    """Render differential, treatment, red flags, XAI, sources, badges."""
    if not result or result.get("type") not in ("answer",):
        return

    render_differential(result.get("differential", []))

    # ── Treatment + Red Flags in expanders ───────────────────
    treatment  = result.get("treatment")
    red_flags  = result.get("red_flags")
    xai        = result.get("xai_explanation", "")
    mh_msg     = result.get("mental_health_msg", "")
    sources    = result.get("sources", [])
    confidence = result.get("confidence", 0)
    risk       = result.get("hallucination_risk", "N/A")
    verdict    = result.get("fact_check", "N/A")

    if treatment:
        with st.expander("💊 Treatment Guidance", expanded=False):
            render_treatment(treatment)

    if red_flags and red_flags.get("red_flags"):
        with st.expander("🚩 Red Flag Symptoms", expanded=True):
            render_red_flags(red_flags)

    if xai:
        with st.expander("💡 Why did I give this answer? (XAI)", expanded=False):
            st.markdown(
                f'<div style="font-family:var(--serif);font-size:0.9rem;'
                f'line-height:1.65;color:var(--text)">{xai}</div>',
                unsafe_allow_html=True
            )

    if mh_msg:
        st.markdown(
            f'<div class="mh-banner">💙 {mh_msg}</div>',
            unsafe_allow_html=True
        )

    # ── Badges ────────────────────────────────────────────────
    rc = _risk_badge_class(risk)
    st.markdown(f"""
    <div class="badge-row">
      <span class="badge badge-conf">Confidence: {confidence}%</span>
      <span class="badge {rc}">Hallucination Risk: {risk}</span>
      <span class="badge badge-conf">Fact-check: {verdict}</span>
    </div>""", unsafe_allow_html=True)

    # ── Sources ───────────────────────────────────────────────
    if sources:
        pills = "".join(f'<span class="source-pill">{s}</span>' for s in sources)
        st.markdown(
            f'<div class="section-label">Sources</div>'
            f'<div class="sources-row">{pills}</div>',
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════════════════════════════
#  STREAMING — Redirect bot's stdout tokens into Streamlit
# ════════════════════════════════════════════════════════════════════════════

class TokenQueue:
    """Captures tokens written to stdout by CareAI._stream() and yields them."""
    def __init__(self):
        self.q = queue.Queue()
        self._done = False

    def write(self, text):
        if text:
            self.q.put(text)

    def flush(self): pass

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
    old_stdout = sys.stdout
    sys.stdout = tq
    try:
        result = bot.chat(user_message)
        result_holder.append(result)
    except Exception as e:
        result_holder.append({"type": "error", "message": str(e)})
    finally:
        sys.stdout = old_stdout
        tq.mark_done()


# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading Care-AI knowledge base (36,585 chunks)…")
def load_bot():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found. Add it to your .env file.")
        st.stop()
    all_chunks, severity_map, precaution_map = build_all_chunks(data_dir="data")
    bot = CareAI(
        knowledge_chunks=all_chunks,
        severity_map=severity_map,
        api_key=api_key,
        precaution_map=precaution_map,
    )
    return bot


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "pending_panels" not in st.session_state:
        st.session_state.pending_panels = []  # list of result dicts to render


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    init_session()
    bot = load_bot()
    render_sidebar(bot)

    # ── Header row with inline reset button ──────────────────
    col_title, col_btn = st.columns([5, 1])
    with col_title:
        st.markdown(
            '<h1 style="font-family:var(--mono);font-size:1.4rem;'
            'letter-spacing:0.08em;color:var(--teal);margin-bottom:0.2rem">'
            '🏥 CARE-AI</h1>'
            '<p style="font-family:var(--serif);color:var(--text-muted);'
            'font-size:0.9rem;margin-top:0;margin-bottom:1rem">'
            'Multilingual · XAI · Hallucination-Controlled Medical Assistant</p>',
            unsafe_allow_html=True
        )
    with col_btn:
        st.markdown('<div style="padding-top:0.6rem"></div>', unsafe_allow_html=True)
        if st.button("🔄 New Chat", use_container_width=True, type="secondary"):
            bot.reset()
            st.session_state.messages = []
            st.session_state.last_result = None
            st.rerun()

    # ── Render existing chat history ──────────────────────────
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(
                    f'<div class="user-bubble"><div class="bubble-inner">'
                    f'{msg["content"]}</div></div>',
                    unsafe_allow_html=True
                )
        else:
            with st.chat_message("assistant", avatar="🏥"):
                # Triage banner
                triage = msg.get("triage")
                if triage and triage in TRIAGE_META:
                    m = TRIAGE_META[triage]
                    st.markdown(
                        f'<div class="triage-banner triage-{triage}">'
                        f'{m["icon"]} {m["label"]}</div>',
                        unsafe_allow_html=True
                    )
                st.markdown(
                    f'<div class="bot-bubble"><div class="bubble-inner">'
                    f'{msg["content"]}</div></div>',
                    unsafe_allow_html=True
                )
                # Render stored result panels
                if msg.get("result"):
                    render_result_panels(msg["result"])

    # ── Chat input ────────────────────────────────────────────
    if user_input := st.chat_input("Describe your symptoms or ask a medical question…"):

        # Render user bubble immediately
        with st.chat_message("user"):
            st.markdown(
                f'<div class="user-bubble"><div class="bubble-inner">'
                f'{user_input}</div></div>',
                unsafe_allow_html=True
            )
        st.session_state.messages.append({"role": "user", "content": user_input})

        # ── Bot response ──────────────────────────────────────
        with st.chat_message("assistant", avatar="🏥"):
            tq            = TokenQueue()
            result_holder = []

            # Run bot in background thread
            t = threading.Thread(
                target=run_bot_in_thread,
                args=(bot, user_input, tq, result_holder),
                daemon=True
            )
            t.start()

            # Stream tokens live into Streamlit
            with st.spinner(""):
                response_placeholder = st.empty()
                streamed_text = ""

                for token in tq.token_gen():
                    # Filter out pipeline status lines (emoji prefixes)
                    if any(token.startswith(p) for p in
                           ["🤔", "📖", "🩺", "🧠", "💊", "🔎", "✅", "🔧", "🔄"]):
                        continue
                    streamed_text += token
                    response_placeholder.markdown(
                        f'<div class="bot-bubble"><div class="bubble-inner">'
                        f'{streamed_text}▌</div></div>',
                        unsafe_allow_html=True
                    )

                t.join()
                result = result_holder[0] if result_holder else {}

            # Final render without cursor
            final_text = result.get("message", streamed_text)
            triage     = result.get("triage")

            if triage and triage in TRIAGE_META:
                m = TRIAGE_META[triage]
                st.markdown(
                    f'<div class="triage-banner triage-{triage}">'
                    f'{m["icon"]} {m["label"]}</div>',
                    unsafe_allow_html=True
                )

            response_placeholder.markdown(
                f'<div class="bot-bubble"><div class="bubble-inner">'
                f'{final_text}</div></div>',
                unsafe_allow_html=True
            )

            # Render panels (diff dx, treatment, red flags, XAI, sources)
            render_result_panels(result)

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