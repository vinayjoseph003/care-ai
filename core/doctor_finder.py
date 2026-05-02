# ================================================================
#  core/doctor_finder.py — Nearby Doctor Finder via OpenStreetMap
#
#  Uses the free Overpass API (no key needed) to find nearby
#  healthcare facilities based on specialist type and user location.
#
#  Flow:
#    1. Map Care-AI specialist name → OSM healthcare tags
#    2. Query Overpass API with a radius around user's lat/lon
#    3. Compute haversine distance for each result
#    4. Rank by distance, return top-N as clean dicts
# ================================================================

import math
import urllib.request
import urllib.parse
import json
import time

# ── Specialist → OSM tag mapping ──────────────────────────────
# Maps Care-AI's specialist labels to the most relevant OSM
# healthcare tags. Each entry is a list of (key, value) pairs
# tried in order — first match wins if results are found.
#
# OSM healthcare tags reference:
#   amenity=doctors         — general GP / outpatient clinic
#   amenity=hospital        — full hospital (fallback for any specialist)
#   healthcare=doctor       — tagged doctor/clinic
#   healthcare:speciality   — specific speciality (when tagged)
#
# Strategy: search for speciality-tagged nodes first, then
# fall back to general hospitals if < 3 results come back.

SPECIALIST_OSM_MAP = {
    # ── General ───────────────────────────────────────────────
    "General Practitioner":     [("amenity", "doctors"), ("healthcare", "doctor")],
    "General Physician":        [("amenity", "doctors"), ("healthcare", "doctor")],
    "Family Medicine":          [("amenity", "doctors"), ("healthcare", "doctor")],

    # ── Internal medicine / common specialities ───────────────
    "Cardiologist":             [("healthcare:speciality", "cardiology"),
                                  ("amenity", "hospital")],
    "Pulmonologist":            [("healthcare:speciality", "pulmonology"),
                                  ("amenity", "hospital")],
    "Gastroenterologist":       [("healthcare:speciality", "gastroenterology"),
                                  ("amenity", "hospital")],
    "Neurologist":              [("healthcare:speciality", "neurology"),
                                  ("amenity", "hospital")],
    "Nephrologist":             [("healthcare:speciality", "nephrology"),
                                  ("amenity", "hospital")],
    "Endocrinologist":          [("healthcare:speciality", "endocrinology"),
                                  ("amenity", "hospital")],
    "Rheumatologist":           [("healthcare:speciality", "rheumatology"),
                                  ("amenity", "hospital")],
    "Haematologist":            [("healthcare:speciality", "haematology"),
                                  ("amenity", "hospital")],
    "Hematologist":             [("healthcare:speciality", "haematology"),
                                  ("amenity", "hospital")],
    "Infectious Disease":       [("amenity", "hospital")],

    # ── Surgery ───────────────────────────────────────────────
    "General Surgeon":          [("healthcare:speciality", "surgery"),
                                  ("amenity", "hospital")],
    "Orthopaedic Surgeon":      [("healthcare:speciality", "orthopaedics"),
                                  ("amenity", "hospital")],
    "Orthopedic Surgeon":       [("healthcare:speciality", "orthopaedics"),
                                  ("amenity", "hospital")],
    "Neurosurgeon":             [("healthcare:speciality", "neurosurgery"),
                                  ("amenity", "hospital")],

    # ── ENT / Head-Neck ───────────────────────────────────────
    "ENT Specialist":           [("healthcare:speciality", "otolaryngology"),
                                  ("healthcare:speciality", "ent"),
                                  ("amenity", "hospital")],
    "Otolaryngologist":         [("healthcare:speciality", "otolaryngology"),
                                  ("amenity", "hospital")],

    # ── Eye ───────────────────────────────────────────────────
    "Ophthalmologist":          [("healthcare:speciality", "ophthalmology"),
                                  ("amenity", "hospital")],
    "Optometrist":              [("healthcare", "optometrist")],

    # ── Skin ──────────────────────────────────────────────────
    "Dermatologist":            [("healthcare:speciality", "dermatology"),
                                  ("amenity", "hospital")],

    # ── Women's health ────────────────────────────────────────
    "Gynaecologist":            [("healthcare:speciality", "gynaecology"),
                                  ("amenity", "hospital")],
    "Gynecologist":             [("healthcare:speciality", "gynaecology"),
                                  ("amenity", "hospital")],
    "Obstetrician":             [("healthcare:speciality", "obstetrics"),
                                  ("amenity", "hospital")],

    # ── Child health ──────────────────────────────────────────
    "Paediatrician":            [("healthcare:speciality", "paediatrics"),
                                  ("amenity", "hospital")],
    "Pediatrician":             [("healthcare:speciality", "paediatrics"),
                                  ("amenity", "hospital")],

    # ── Mental health ─────────────────────────────────────────
    "Psychiatrist":             [("healthcare:speciality", "psychiatry"),
                                  ("amenity", "hospital")],
    "Psychologist":             [("healthcare", "psychologist"),
                                  ("amenity", "hospital")],

    # ── Musculoskeletal / Physio ──────────────────────────────
    "Physiotherapist":          [("healthcare", "physiotherapist"),
                                  ("amenity", "hospital")],
    "Orthopaedist":             [("healthcare:speciality", "orthopaedics"),
                                  ("amenity", "hospital")],

    # ── Dental ────────────────────────────────────────────────
    "Dentist":                  [("amenity", "dentist")],

    # ── Urology ───────────────────────────────────────────────
    "Urologist":                [("healthcare:speciality", "urology"),
                                  ("amenity", "hospital")],

    # ── Oncology ──────────────────────────────────────────────
    "Oncologist":               [("healthcare:speciality", "oncology"),
                                  ("amenity", "hospital")],
}

# Radius in metres to search. Starts at PRIMARY_RADIUS,
# expands to FALLBACK_RADIUS if < MIN_RESULTS found.
PRIMARY_RADIUS  = 5_000    # 5 km
FALLBACK_RADIUS = 15_000   # 15 km
MIN_RESULTS     = 3        # expand radius if fewer than this
MAX_RESULTS     = 8        # cap displayed results

# Overpass API endpoints — tried in order if one fails
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


# ── Haversine distance ─────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres between two points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Overpass query builder ─────────────────────────────────────

def _build_query(lat: float, lon: float, radius: int, osm_key: str, osm_val: str) -> str:
    """Build an Overpass QL query for nodes + ways matching a tag within radius."""
    return (
        f"[out:json][timeout:10];"
        f"("
        f"  node[\"{osm_key}\"=\"{osm_val}\"](around:{radius},{lat},{lon});"
        f"  way[\"{osm_key}\"=\"{osm_val}\"](around:{radius},{lat},{lon});"
        f");"
        f"out center tags;"
    )


def _query_overpass(query: str) -> list:
    """POST query to Overpass API, return list of OSM elements or []."""
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            data = urllib.parse.urlencode({"data": query}).encode()
            req  = urllib.request.Request(
                endpoint, data=data,
                headers={"User-Agent": "CareAI-DoctorFinder/1.0"},
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                result = json.loads(resp.read().decode())
                return result.get("elements", [])
        except Exception:
            continue   # try next endpoint
    return []


# ── Element → clean dict ───────────────────────────────────────

def _parse_element(el: dict, user_lat: float, user_lon: float) -> dict | None:
    """
    Convert a raw OSM element to a clean result dict.
    Returns None if lat/lon can't be determined.
    """
    tags = el.get("tags", {})

    # Get coordinates — nodes have lat/lon directly, ways have centre
    if el.get("type") == "node":
        lat = el.get("lat")
        lon = el.get("lon")
    else:
        centre = el.get("center", {})
        lat = centre.get("lat")
        lon = centre.get("lon")

    if lat is None or lon is None:
        return None

    # Build a human-readable name
    name = (
        tags.get("name")
        or tags.get("name:en")
        or tags.get("operator")
        or tags.get("brand")
        or None
    )
    if not name:
        return None   # skip unnamed facilities

    # Determine facility type label
    amenity = tags.get("amenity", "")
    healthcare = tags.get("healthcare", "")
    speciality = tags.get("healthcare:speciality", "")

    if amenity == "hospital":
        facility_type = "Hospital"
    elif amenity == "doctors":
        facility_type = "Clinic"
    elif amenity == "dentist":
        facility_type = "Dental Clinic"
    elif healthcare == "physiotherapist":
        facility_type = "Physiotherapy"
    elif healthcare == "psychologist":
        facility_type = "Psychology Clinic"
    elif healthcare == "optometrist":
        facility_type = "Optometry"
    elif speciality:
        facility_type = speciality.replace("_", " ").title() + " Clinic"
    else:
        facility_type = "Healthcare Facility"

    distance_km = _haversine_km(user_lat, user_lon, lat, lon)

    return {
        "name":          name,
        "facility_type": facility_type,
        "distance_km":   round(distance_km, 2),
        "distance_str":  f"{distance_km:.1f} km" if distance_km >= 1.0
                         else f"{int(distance_km * 1000)} m",
        "lat":           lat,
        "lon":           lon,
        "osm_id":        el.get("id"),
    }


# ── PUBLIC FUNCTION ────────────────────────────────────────────

def find_nearby_doctors(
    specialist: str,
    user_lat: float,
    user_lon: float,
    max_results: int = MAX_RESULTS,
) -> dict:
    """
    Find nearby doctors/clinics for a given specialist type.

    Args:
        specialist:  Care-AI specialist label (e.g. "Cardiologist")
        user_lat:    User's latitude from browser geolocation
        user_lon:    User's longitude from browser geolocation
        max_results: Max number of results to return

    Returns:
        {
            "specialist":    str,
            "results":       [ {name, facility_type, distance_str, ...}, ... ],
            "radius_used_km": int,
            "error":         str or None,
        }
    """
    # Normalise specialist name — strip trailing details like "e.g. ..."
    specialist_clean = specialist.split("(")[0].strip().title()

    # Look up OSM tag list; fall back to general hospital search
    tag_pairs = SPECIALIST_OSM_MAP.get(
        specialist_clean,
        [("amenity", "hospital")],
    )

    all_results: list[dict] = []
    radius_used = PRIMARY_RADIUS

    # Try primary radius, then expand if needed
    for radius in (PRIMARY_RADIUS, FALLBACK_RADIUS):
        radius_used = radius
        raw_elements: list = []

        for osm_key, osm_val in tag_pairs:
            query    = _build_query(user_lat, user_lon, radius, osm_key, osm_val)
            elements = _query_overpass(query)
            raw_elements.extend(elements)
            time.sleep(0.3)   # be polite to Overpass

        # Deduplicate by OSM id
        seen_ids = set()
        for el in raw_elements:
            osm_id = el.get("id")
            if osm_id in seen_ids:
                continue
            seen_ids.add(osm_id)
            parsed = _parse_element(el, user_lat, user_lon)
            if parsed:
                all_results.append(parsed)

        if len(all_results) >= MIN_RESULTS:
            break   # enough results — no need to expand radius

    if not all_results:
        return {
            "specialist":     specialist_clean,
            "results":        [],
            "radius_used_km": radius_used // 1000,
            "error": (
                "No nearby facilities found. "
                "Try searching on Google Maps or Practo for "
                f"'{specialist_clean} near me'."
            ),
        }

    # Sort by distance, deduplicate by name (keep closest)
    seen_names: dict[str, dict] = {}
    for r in sorted(all_results, key=lambda x: x["distance_km"]):
        key = r["name"].lower().strip()
        if key not in seen_names:
            seen_names[key] = r

    ranked = list(seen_names.values())[:max_results]

    return {
        "specialist":     specialist_clean,
        "results":        ranked,
        "radius_used_km": radius_used // 1000,
        "error":          None,
    }