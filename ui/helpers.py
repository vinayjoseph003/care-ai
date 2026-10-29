"""
ui/helpers.py — Pure presentation and data helpers for Care-AI UI.
Contains ZERO Streamlit imports to guarantee fast and isolated unit testing.
"""

import html
import logging
import re
from typing import Any, Callable

logger = logging.getLogger(__name__)

LOAD_FAILED_MESSAGE = (
    "The knowledge base could not be loaded. Please refresh the page in a minute. "
    "If this is a medical emergency, call 112 now."
)


def safe_load(loader: Callable[[], Any]) -> tuple[Any, str | None]:
    """
    Safely execute a resource loader function.
    Catches all standard exceptions (including MemoryError) without catching BaseException.
    Logs only the exception type name to avoid leaking traceback details or user/secret data.
    Returns (result, None) on success or (None, LOAD_FAILED_MESSAGE) on failure.
    """
    try:
        return loader(), None
    except Exception as e:
        logger.error("Knowledge base loading failed: %s", type(e).__name__)
        return None, LOAD_FAILED_MESSAGE


def strip_rendered_emoji_prefix(text: str) -> str:
    """Strip leading hospital/doctor emoji from rendered text without altering stored messages."""
    if not text:
        return ""
    return re.sub(r'^[🏥🩺\s]+', '', str(text))


def sanitize_html(text: str, preserve_newlines: bool = True) -> str:
    """Safely escape user and model text before inserting into HTML templates."""
    if not text:
        return ""
    escaped = html.escape(str(text))
    if preserve_newlines:
        return escaped.replace("\n", "<br>")
    return escaped


def top_diagnosis_label(xai_logs: list | None, last_top: str | None) -> str:
    """Return 'Not determined' if latest assessment was low_confidence, else last_top."""
    is_low_conf = bool(xai_logs[-1].get("low_confidence")) if xai_logs else False
    if is_low_conf:
        return "Not determined"
    return str(last_top).strip() if last_top else ""


def reliability_badge(risk: str | None, verdict: str | None, low_confidence: bool = False) -> tuple[str, str, str]:
    """Return (_icon, _label, _cls) for hallucination risk / verification status."""
    _risk_up    = str(risk).upper()
    _verdict_up = str(verdict).upper()
    if _risk_up == "HIGH" or _verdict_up in ("UNVERIFIED", "FAILED"):
        return ("⚠️", "Verify with doctor", "badge-high")
    if low_confidence:
        return ("🟡", "Confirm with doctor", "badge-medium")
    if _risk_up == "LOW" and _verdict_up == "VERIFIED":
        return ("✅", "Well-supported", "badge-low")
    return ("🟡", "Confirm with doctor", "badge-medium")


def badge_visible(low_confidence: bool, verified: bool | str = False) -> bool:
    """
    Determine whether the 'Well-supported' badge should be visible.
    When low_confidence is True, the badge is always suppressed (returns False).
    """
    if low_confidence:
        return False
    if isinstance(verified, str):
        return verified.upper() == "VERIFIED"
    return bool(verified)


def urgency_style(level: str | None) -> dict:
    """
    Return styling metadata and WCAG AA-compliant palette for an urgency/triage level.
    Supported levels: SELF_CARE, SEE_DOCTOR, URGENT_CARE, EMERGENCY.
    Returns safe fallback dict for None or unknown level.
    """
    norm = str(level).strip().upper() if level else ""
    styles = {
        "SELF_CARE": {
            "label": "Self-Care",
            "icon": "●",
            "color": "#15803D",
            "bg": "#DCFCE7",
            "border": "#16A34A",
            "css": "triage-SELF_CARE",
        },
        "SEE_DOCTOR": {
            "label": "See a Doctor",
            "icon": "●",
            "color": "#92400E",
            "bg": "#FEF3C7",
            "border": "#D97706",
            "css": "triage-SEE_DOCTOR",
        },
        "URGENT_CARE": {
            "label": "Urgent Care",
            "icon": "●",
            "color": "#9A3412",
            "bg": "#FFEDD5",
            "border": "#EA580C",
            "css": "triage-URGENT_CARE",
        },
        "EMERGENCY": {
            "label": "Emergency",
            "icon": "▲",
            "color": "#991B1B",
            "bg": "#FEE2E2",
            "border": "#DC2626",
            "css": "triage-EMERGENCY",
        },
    }
    if norm in styles:
        return styles[norm]
    return {
        "label": str(level) if level else "General Information",
        "icon": "●",
        "color": "#061A3A",
        "bg": "#E7EFEF",
        "border": "#CBD5E0",
        "css": "triage-DEFAULT",
    }


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


def format_specialist_heading(specialist: str | None) -> str:
    """
    Format specialist name/label for display in doctor finder heading and cards:
    - Strips any stray trailing 's' after parentheses (e.g. '(GP)s' -> '(GP)').
    - Keeps medical acronyms such as GP and ENT uppercase.
    - Handles None and empty strings safely, returning an empty string.
    """
    if not specialist:
        return ""
    text = str(specialist).strip()
    # Strip stray trailing 's' after closing parenthesis (e.g. '(GP)s' -> '(GP)')
    text = re.sub(r'\)\s*s\b', ')', text, flags=re.IGNORECASE)
    # Ensure acronyms stay uppercase (GP, ENT)
    for acronym in ("GP", "ENT", "ER", "ICU", "OB/GYN"):
        text = re.sub(rf'\b{re.escape(acronym)}\b', acronym, text, flags=re.IGNORECASE)
    return text

