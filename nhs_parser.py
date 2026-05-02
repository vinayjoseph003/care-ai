import json
import re
from pathlib import Path

RAW_PATH = Path("data/nhs_raw.json")
OUT_PATH = Path("data/nhs_chunks.json")


# ── SECTION TAGGING (P4) ─────────────────────────────────────

_SECTION_PATTERNS = {
    "causes":     re.compile(r"(?i)(cause[sd]?|why.{0,20}happen|risk factor)"),
    "treatments": re.compile(r"(?i)(treatment|treated|management|therapy|remedy)"),
    "symptoms":   re.compile(r"(?i)(symptom[s]?|sign[s]?|you may (feel|have|notice))"),
    "diagnosis":  re.compile(r"(?i)(diagnos|test[s]?|examined|scan|blood test)"),
    "prevention": re.compile(r"(?i)(prevent|avoid|reduce risk|vaccination)"),
}


def tag_chunk_sections(text: str) -> list[str]:
    """Return section tags found in chunk text."""
    return [
        tag for tag, pattern in _SECTION_PATTERNS.items()
        if pattern.search(text)
    ]


# ── CATEGORY CLASSIFIER ─────────────────────────────────────

def categorize(text):
    text = text.lower()

    if any(x in text for x in ["mouth", "gum", "tooth", "palate"]):
        return "Oral/Dental"
    if any(x in text for x in ["ear", "nose", "throat", "sinus"]):
        return "ENT"
    if any(x in text for x in ["bone", "joint", "muscle", "knee", "shoulder"]):
        return "Musculoskeletal"
    if any(x in text for x in ["skin", "rash", "itch"]):
        return "Dermatology"

    return "General"


# ── MAIN PARSER ─────────────────────────────────────────────

def main():
    print("🔄 Parsing NHS raw data...")

    if not RAW_PATH.exists():
        print("❌ nhs_raw.json not found")
        return

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = []

    for item in data:
        text = item.get("text", "").strip()
        title = item.get("title", "").strip()

        if not text or len(text) < 100:
            continue

        chunk = {
            "text": f"Condition: {title}. {text}",
            "source": "NHS",
            "category": categorize(text),
        }

        # ✅ NEW: section tagging
        chunk["section_tags"] = tag_chunk_sections(chunk["text"])

        chunks.append(chunk)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"\n✅ Parsed {len(chunks)} NHS condition chunks")
    print(f"💾 Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()