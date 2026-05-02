"""
rebuild_cache.py
================
Deletes the stale vector index cache and triggers a full rebuild
by importing VectorStore (which auto-builds on first load).

Run from project root AFTER integrating NHS data:
    python rebuild_cache.py

This will:
  1. Delete cache/vector_index.pkl (759 MB)
  2. Load all knowledge sources (including NHS)
  3. Build a new TF-IDF index and save it to cache/

Expect ~3-5 minutes on first run depending on machine speed.
"""

import os
import sys

CACHE_FILE = os.path.join("cache", "vector_index.pkl")


def main():
    # ── Step 1: delete old cache ──────────────────────────────
    if os.path.exists(CACHE_FILE):
        size_mb = os.path.getsize(CACHE_FILE) / (1024 * 1024)
        print(f"🗑️  Deleting stale cache: {CACHE_FILE} ({size_mb:.0f} MB)")
        os.remove(CACHE_FILE)
        print("✅ Cache deleted.")
    else:
        print(f"ℹ️  No cache found at {CACHE_FILE} — nothing to delete.")

    # ── Step 2: trigger rebuild via build_all_chunks ──────────
    print("\n📚 Rebuilding knowledge base (this will take a few minutes)...")
    from dotenv import load_dotenv
    load_dotenv()

    from core.knowledge_base import build_all_chunks
    from bot import CareAI

    all_chunks, sev_map, prec_map = build_all_chunks(data_dir="data")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("⚠️  GROQ_API_KEY not set — skipping bot init (cache still built).")
        # Build VectorStore directly to save cache
        from core.vector_store import VectorStore
        vs = VectorStore(all_chunks)
        print(f"\n✅ VectorStore built with {len(all_chunks)} chunks.")
    else:
        bot = CareAI(
            knowledge_chunks=all_chunks,
            severity_map=sev_map,
            api_key=api_key,
            precaution_map=prec_map,
        )
        print(f"\n✅ Care-AI ready with {len(all_chunks)} chunks.")

    print(f"\n💾 New cache saved → {CACHE_FILE}")
    print("🚀 You can now run: streamlit run ui/app.py")


if __name__ == "__main__":
    main()