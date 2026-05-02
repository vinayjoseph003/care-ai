# ================================================================
#  ui/doctor_finder_panel.py — Doctor Finder UI for Care-AI
#
#  Premium redesign: styled hospital cards with distance badges,
#  grid layout, colour-coded distance, OSM map links.
#
#  Requires:  pip install streamlit-js-eval
# ================================================================

import streamlit as st
import urllib.parse

try:
    from streamlit_js_eval import get_geolocation
    _HAS_JS_EVAL = True
except ImportError:
    _HAS_JS_EVAL = False


# ── Triage levels where doctor finder is relevant ─────────────
_SHOW_FOR_TRIAGE = {"SEE_DOCTOR", "URGENT_CARE", "EMERGENCY"}

# ── CSS ────────────────────────────────────────────────────────
_CSS = """
<style>
/* ── Doctor Finder Section Header ─────────────────────────── */
.df-section {
    margin: 1.8rem 0 0;
    padding-top: 1.2rem;
    border-top: 1px solid rgba(57,208,216,0.15);
}
.df-section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.35rem;
}
.df-section-icon {
    width: 32px; height: 32px;
    background: rgba(57,208,216,0.1);
    border: 1px solid rgba(57,208,216,0.25);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px;
    flex-shrink: 0;
}
.df-section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #39d0d8;
}
.df-section-sub {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 0.82rem;
    color: rgba(125,133,144,0.9);
    margin-bottom: 1rem;
    line-height: 1.5;
    padding-left: 42px;
}

/* ── Hospital card grid ─────────────────────────────────────── */
.df-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 10px;
    margin-bottom: 0.6rem;
}
.df-card {
    position: relative;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 14px 16px 12px;
    background: #161b22;
    transition: border-color 0.2s, background 0.2s, transform 0.15s;
    text-decoration: none !important;
    color: inherit !important;
    display: block;
    overflow: hidden;
}
.df-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #39d0d8, #388bfd);
    opacity: 0;
    transition: opacity 0.2s;
    border-radius: 12px 12px 0 0;
}
.df-card:hover {
    border-color: rgba(57,208,216,0.4);
    background: #1c2333;
    transform: translateY(-2px);
}
.df-card:hover::before { opacity: 1; }

/* Map icon top-right */
.df-card-map-icon {
    position: absolute;
    top: 12px; right: 12px;
    font-size: 14px;
    opacity: 0.35;
    transition: opacity 0.2s;
}
.df-card:hover .df-card-map-icon { opacity: 0.8; }

/* Facility type badge */
.df-card-type-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 4px;
    margin-bottom: 7px;
    background: rgba(56,139,253,0.1);
    color: #58a6ff;
    border: 1px solid rgba(56,139,253,0.2);
}
.df-card-type-badge.badge-hospital {
    background: rgba(57,208,216,0.1);
    color: #39d0d8;
    border-color: rgba(57,208,216,0.2);
}
.df-card-type-badge.badge-clinic {
    background: rgba(56,139,253,0.1);
    color: #58a6ff;
    border-color: rgba(56,139,253,0.2);
}
.df-card-type-badge.badge-pharmacy {
    background: rgba(63,185,80,0.1);
    color: #3fb950;
    border-color: rgba(63,185,80,0.2);
}

/* Name */
.df-card-name {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 0.92rem;
    font-weight: 600;
    color: #e6edf3;
    line-height: 1.35;
    margin-bottom: 10px;
    padding-right: 20px;
    word-break: break-word;
}

/* Distance pill */
.df-dist-row {
    display: flex;
    align-items: center;
    gap: 8px;
}
.df-dist-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
}
/* Green ≤ 3km */
.df-dist-near {
    background: rgba(63,185,80,0.12);
    color: #3fb950;
    border: 1px solid rgba(63,185,80,0.25);
}
/* Amber > 3km */
.df-dist-far {
    background: rgba(210,153,34,0.12);
    color: #d29922;
    border: 1px solid rgba(210,153,34,0.25);
}
.df-dist-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}
.df-dist-near .df-dist-dot { background: #3fb950; }
.df-dist-far  .df-dist-dot { background: #d29922; }

/* ── Count + radius note ────────────────────────────────────── */
.df-results-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.08em;
    color: rgba(125,133,144,0.6);
    margin-top: 0.4rem;
    padding: 0 2px;
}

/* ── Disclaimer ─────────────────────────────────────────────── */
.df-disclaimer {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 0.78rem;
    color: rgba(125,133,144,0.7);
    margin-top: 0.7rem;
    padding: 8px 12px;
    border-left: 2px solid rgba(57,208,216,0.2);
    line-height: 1.55;
    background: rgba(57,208,216,0.03);
    border-radius: 0 6px 6px 0;
}

/* ── Error / empty state ────────────────────────────────────── */
.df-alert {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 0.85rem;
    color: rgba(245,158,11,0.9);
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid rgba(245,158,11,0.25);
    background: rgba(245,158,11,0.06);
    line-height: 1.5;
}

/* ── Manual search links ────────────────────────────────────── */
.df-manual-links {
    display: flex;
    gap: 10px;
    margin-top: 0.8rem;
    flex-wrap: wrap;
}
.df-manual-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 6px 14px;
    border-radius: 8px;
    border: 1px solid #30363d;
    background: #161b22;
    color: #58a6ff !important;
    text-decoration: none !important;
    transition: border-color 0.15s, background 0.15s;
}
.df-manual-link:hover {
    border-color: rgba(56,139,253,0.4);
    background: #1c2333;
}

/* ── Info box ───────────────────────────────────────────────── */
.df-info {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 0.84rem;
    color: #7d8590;
    padding: 8px 12px;
    border: 1px solid #30363d;
    border-radius: 8px;
    background: rgba(28,35,51,0.5);
    margin-bottom: 0.5rem;
}
</style>
"""


# ── Helpers ─────────────────────────────────────────────────────

def _facility_badge_class(facility_type: str) -> str:
    t = facility_type.lower()
    if "hospital" in t:  return "badge-hospital"
    if "clinic"   in t:  return "badge-clinic"
    if "pharmacy" in t:  return "badge-pharmacy"
    return "badge-clinic"


def _distance_info(distance_str: str) -> tuple[str, bool]:
    """Returns (pill_class, is_near)."""
    try:
        if "m" in distance_str and "km" not in distance_str:
            return "df-dist-near", True          # metres → always near
        val = float(distance_str.replace(" km", "").strip())
        return ("df-dist-near", True) if val <= 3.0 else ("df-dist-far", False)
    except Exception:
        return "df-dist-near", True


def _render_cards(results: list) -> str:
    cards = []
    for r in results:
        dist_str    = r.get("distance_str", "?")
        pill_class, _ = _distance_info(dist_str)
        ftype       = r.get("facility_type", "Clinic")
        badge_class = _facility_badge_class(ftype)
        name        = r.get("name", "Unknown")
        lat, lon    = r.get("lat", 0), r.get("lon", 0)
        osm_url     = (
            f"https://www.openstreetmap.org/?mlat={lat}"
            f"&mlon={lon}#map=17/{lat}/{lon}"
        )

        cards.append(f"""
<a href="{osm_url}" target="_blank" class="df-card">
  <span class="df-card-map-icon">🗺️</span>
  <div class="df-card-type-badge {badge_class}">{ftype}</div>
  <div class="df-card-name">{name}</div>
  <div class="df-dist-row">
    <span class="df-dist-pill {pill_class}">
      <span class="df-dist-dot"></span>{dist_str}
    </span>
  </div>
</a>""")

    return f'<div class="df-card-grid">{"".join(cards)}</div>'


def _manual_links(specialist: str):
    q_practo = urllib.parse.quote(specialist)
    q_gmaps  = urllib.parse.quote(f"{specialist} near me")
    st.markdown(f"""
<div class="df-manual-links">
  <a class="df-manual-link"
     href="https://www.practo.com/search/doctors?results_type=doctor&q={q_practo}&city="
     target="_blank">🔗 Search on Practo</a>
  <a class="df-manual-link"
     href="https://www.google.com/maps/search/{q_gmaps}"
     target="_blank">🗺️ Google Maps</a>
</div>""", unsafe_allow_html=True)


# ── MAIN PUBLIC FUNCTION ─────────────────────────────────────────

def render_doctor_finder(bot, response: dict):
    """
    Render the Doctor Finder panel in the Streamlit app.
    Call once per response after the main answer panels.
    """
    triage     = response.get("triage", "SELF_CARE")
    specialist = response.get("specialist", "")

    if triage not in _SHOW_FOR_TRIAGE:
        return
    if not specialist:
        return

    # Inject CSS once
    if "df_css_injected" not in st.session_state:
        st.markdown(_CSS, unsafe_allow_html=True)
        st.session_state["df_css_injected"] = True

    # Section header
    st.markdown(f"""
<div class="df-section">
  <div class="df-section-header">
    <div class="df-section-icon">🏥</div>
    <div class="df-section-title">Nearby {specialist}s</div>
  </div>
  <div class="df-section-sub">
    Based on your symptoms, we recommend seeing a <strong
    style="color:#e6edf3">{specialist}</strong>.
    Find verified facilities near you — tap any card to open on the map.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── No streamlit-js-eval installed ──────────────────────────
    if not _HAS_JS_EVAL:
        st.warning(
            "📦 Install `streamlit-js-eval` to enable live location: "
            "`pip install streamlit-js-eval`"
        )
        _manual_links(specialist)
        return

    # ── State keys ───────────────────────────────────────────────
    STATE   = "df_state"
    RESULTS = "df_results"

    if STATE not in st.session_state:
        st.session_state[STATE] = "idle"

    current = st.session_state[STATE]

    # ── IDLE — show Find button ──────────────────────────────────
    if current == "idle":
        col1, col2 = st.columns([2, 3])
        with col1:
            if st.button("📍 Find Doctors Near Me",
                         key="df_find_btn", use_container_width=True):
                st.session_state[STATE] = "locating"
                st.rerun()
        with col2:
            st.caption("Uses your device location · No data stored")

    # ── LOCATING — request geolocation via streamlit-js-eval ────
    elif current == "locating":
        st.markdown(
            '<div class="df-info">📍 Allow location access in the browser popup…</div>',
            unsafe_allow_html=True
        )
        loc = get_geolocation(component_key="df_geo")

        if loc is None:
            st.caption("⏳ Waiting for location permission…")
            return

        coords = loc.get("coords") if isinstance(loc, dict) else None

        if coords:
            lat = coords.get("latitude")
            lon = coords.get("longitude")
            if lat and lon:
                st.session_state["df_lat"] = float(lat)
                st.session_state["df_lon"] = float(lon)
                st.session_state[STATE]    = "fetching"
                st.rerun()
            else:
                st.session_state[STATE] = "error"
                st.session_state["df_geo_error"] = "Could not read coordinates."
                st.rerun()
        else:
            err  = loc.get("error", {}) if isinstance(loc, dict) else {}
            code = err.get("code", 0) if isinstance(err, dict) else 0
            msgs = {
                1: "Location access was denied. Please allow it in browser settings.",
                2: "Could not determine your location. Check GPS / network.",
                3: "Location request timed out. Try again.",
            }
            st.session_state["df_geo_error"] = msgs.get(
                code, "Location unavailable. Use manual search below."
            )
            st.session_state[STATE] = "error"
            st.rerun()

    # ── FETCHING — call OSM via bot ──────────────────────────────
    elif current == "fetching":
        lat = st.session_state.get("df_lat")
        lon = st.session_state.get("df_lon")
        if lat and lon:
            with st.spinner(f"Finding nearby {specialist}s…"):
                result = bot.find_nearby_doctors(lat, lon)
            st.session_state[RESULTS] = result
            st.session_state[STATE]   = "done"
            st.rerun()
        else:
            st.session_state[STATE] = "idle"
            st.rerun()

    # ── DONE — show results ──────────────────────────────────────
    elif current == "done":
        result    = st.session_state.get(RESULTS, {})
        error     = result.get("error")
        results   = result.get("results", [])
        radius_km = result.get("radius_used_km", 5)

        if error:
            st.markdown(
                f'<div class="df-alert">ℹ️ {error}</div>',
                unsafe_allow_html=True
            )
        elif results:
            st.markdown(_render_cards(results), unsafe_allow_html=True)
            st.markdown(
                f'<div class="df-results-meta">'
                f'{len(results)} results · within {radius_km} km · '
                f'Data © OpenStreetMap contributors</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="df-alert">No {specialist}s found within '
                f'{radius_km} km of your location.</div>',
                unsafe_allow_html=True,
            )
            _manual_links(specialist)

        st.markdown(
            '<div class="df-disclaimer">'
            '⚠️ These results are nearby options to explore — not a personal recommendation. '
            'Always verify credentials and availability before visiting. '
            'In an emergency, call <strong>112</strong> immediately.'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.button("🔄 Search Again", key="df_reset_btn"):
            for k in (STATE, RESULTS, "df_lat", "df_lon", "df_geo_error"):
                st.session_state.pop(k, None)
            st.rerun()

    # ── ERROR ────────────────────────────────────────────────────
    elif current == "error":
        err_msg = st.session_state.get("df_geo_error", "Location unavailable.")
        st.markdown(
            f'<div class="df-alert">📍 {err_msg}</div>',
            unsafe_allow_html=True
        )
        _manual_links(specialist)

        if st.button("↩️ Try Again", key="df_retry_btn"):
            for k in (STATE, "df_geo_error"):
                st.session_state.pop(k, None)
            st.rerun()