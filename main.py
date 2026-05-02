"""
Care-AI — Entry Point
Run with: python main.py
"""

import os
from dotenv import load_dotenv

from core.knowledge_base import build_all_chunks
from bot  import CareAI
from utils import display_response, display_symptom_score

load_dotenv()


def main():
    print("=" * 58)
    print("  🏥 Care-AI — Multilingual XAI Medical Chatbot")
    print("=" * 58)
    print("  Features  : Conversational | Multilingual | XAI | RAG")
    print("  Languages : Auto-detected — type in any language")
    print("  Commands  : reset | log | export | severity | quit")
    print("=" * 58 + "\n")

    # ── Load ALL knowledge sources ──────────────────────────
    # build_all_chunks now returns 3 values (precaution_map is new)
    all_chunks, severity_map, precaution_map = build_all_chunks(data_dir="data")

    # ── Initialize bot ───────────────────────────────────────
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
    has_cloud = any([
        router_config["openrouter_api_key"],
        router_config["groq_api_key"],
        router_config["gemini_api_key"],
    ])
    if not has_cloud:
        print("❌ No API key found. Add at least one of these to your .env:")
        print("   OPENROUTER_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY")
        return

    bot = CareAI(
        knowledge_chunks=all_chunks,
        severity_map=severity_map,
        router_config=router_config,
        precaution_map=precaution_map,
    )

    # ── Greeting ─────────────────────────────────────────────
    print("Care-AI: Hello! I'm Care-AI, your medical information assistant.")
    print("         I'll ask a few questions before giving you information.")
    print("         You can talk to me in any language.")
    print("         How can I help you today?\n")

    # ── Main loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! Stay healthy. 👋")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "quit":
            print("Goodbye! Stay healthy. 👋")
            break
        if cmd == "reset":
            bot.reset()
            print("Care-AI: Hello again! How can I help you?\n")
            continue
        if cmd == "log":
            import json
            print(json.dumps(bot.xai_logs[-1] if bot.xai_logs else {}, indent=2))
            continue
        if cmd == "export":
            bot.export_log()
            continue
        if cmd == "severity":
            if bot.severity_result:
                display_symptom_score(bot.severity_result)
            else:
                print("No symptom scores yet. Ask about symptoms first.\n")
            continue

        result = bot.chat(user_input)
        display_response(result)


if __name__ == "__main__":
    main()