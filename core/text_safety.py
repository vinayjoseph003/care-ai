"""
core/text_safety.py — Pure text safety utilities for Care-AI.

Normalizes emergency numbers (replacing 911/999 with 112) for India localization,
guarding against false positives for dosages, measurement units, dates, and non-emergency digits.
"""

import re

# Match 911 or 999 with digit lookarounds, excluding numbers followed by
# optional whitespace and a medical/measurement unit (case-insensitive).
_UNIT_PATTERN = r'(?!\s*(?:(?:mg|mcg|g|ml|kg|cm|mm|iu|units)\b|%))'
_EMERGENCY_RE = re.compile(
    rf'(?<!\d)(?:911|999)(?!\d){_UNIT_PATTERN}',
    re.IGNORECASE
)


def normalize_emergency_numbers(val):
    r"""
    Replace standalone 911 and 999 with 112 in text using (?<!\d)(?:911|999)(?!\d),
    unless followed by optional whitespace and a unit (mg, mcg, g, ml, kg, cm, mm, iu, %, units).
    None or non-str is returned unchanged.
    112 and 108 are untouched.
    Recursively processes lists and dicts without mutating the original structures.
    """
    if isinstance(val, str):
        return _EMERGENCY_RE.sub("112", val)
    if isinstance(val, list):
        return [normalize_emergency_numbers(item) for item in val]
    if isinstance(val, dict):
        return {k: normalize_emergency_numbers(v) for k, v in val.items()}
    return val
