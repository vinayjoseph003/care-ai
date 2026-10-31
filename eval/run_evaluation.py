"""
eval/run_evaluation.py — Evaluates Care-AI on eval/test_cases.json

Harness Rules:
1. Run order: Emergency/red-flag cases first (TC016-TC030), then the rest (TC001-TC015).
2. Enable router fallback chain (Gemini, OpenRouter, Groq).
3. Drain turns rule: After scripted turns, if no triage was produced, send the fixed reply
   "No, that is all I can tell you." up to 3 times, recording drain_turns_used per case.
4. Record provider_used per case.
5. If every provider is rate-limited, mark remaining cases as "not_run" and halt.
   Rate limits are not scored as model failures.
6. Saves raw per-case outputs to eval/results_current.json incrementally with resume capability.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from dotenv import load_dotenv

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

load_dotenv(os.path.join(WORKSPACE_ROOT, ".env"))

from core.knowledge_base import build_all_chunks
from bot.careai import CareAI

TRIAGE_ORDER = {
    "SELF_CARE": 0,
    "SEE_DOCTOR": 1,
    "URGENT_CARE": 2,
    "EMERGENCY": 3,
}

DRAIN_REPLY = "No, that is all I can tell you."
MAX_DRAIN_TURNS = 3
DELAY_SECONDS = 3.5  # Polite delay between pipeline calls to reduce rate limit frequency


def get_git_commit():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=WORKSPACE_ROOT)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def main():
    test_cases_path = os.path.join(WORKSPACE_ROOT, "eval", "test_cases.json")
    test_eval_path = os.path.join(WORKSPACE_ROOT, "eval", "test_cases_evaluated.json")
    results_path = os.path.join(WORKSPACE_ROOT, "eval", "results_current.json")

    if not os.path.exists(test_cases_path):
        print(f"Error: {test_cases_path} does not exist.")
        sys.exit(1)

    with open(test_cases_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)["cases"]

    with open(test_eval_path, "r", encoding="utf-8") as f:
        eval_cases = {c["id"]: c for c in json.load(f)["cases"]}

    # Reorder cases: TC016-TC030 first, then TC001-TC015
    emergency_cases = [c for c in all_cases if c["id"] >= "TC016"]
    other_cases = [c for c in all_cases if c["id"] < "TC016"]
    ordered_cases = emergency_cases + other_cases

    # Initialize results container
    results_data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "commit": get_git_commit(),
        "cases": [],
    }

    completed_ids = set()
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                if isinstance(old_data, dict) and "cases" in old_data:
                    results_data = old_data
                    completed_ids = {c["id"] for c in results_data["cases"] if c.get("status") in ("completed", "provider_error")}
                    print(f"Resuming: found {len(completed_ids)} already evaluated cases.")
        except Exception as e:
            print(f"Could not read existing results, starting fresh: {e}")

    # Build knowledge chunks once
    print("Initializing knowledge chunks...")
    chunks, sev_map, prec_map = build_all_chunks(os.path.join(WORKSPACE_ROOT, "data"))

    # Configure multi-provider router with fallback
    router_cfg = {
        "primary_provider": os.getenv("LLM_PRIMARY_PROVIDER", "gemini"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "openrouter_model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct"),
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "groq_model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "ollama_host": "http://localhost:99999",
    }

    bot = CareAI(
        knowledge_chunks=chunks,
        severity_map=sev_map,
        router_config=router_cfg,
        precaution_map=prec_map,
    )

    print(f"\nStarting evaluation of {len(ordered_cases)} cases (Run order: TC016-TC030 first, then TC001-TC015)...\n")

    rate_limit_halt = False

    for idx, case in enumerate(ordered_cases):
        cid = case["id"]
        if cid in completed_ids:
            print(f"[{idx+1}/{len(ordered_cases)}] {cid}: already completed, skipping.")
            continue

        if rate_limit_halt:
            print(f"[{idx+1}/{len(ordered_cases)}] Marking {cid} as 'not_run' due to prior provider rate-limit.")
            results_data["cases"].append({
                "id": cid,
                "status": "not_run",
                "reason": "rate_limit_halt",
            })
            continue

        meta = eval_cases.get(cid, {})
        exp_triage = meta.get("expected_triage", [])
        if isinstance(exp_triage, str):
            exp_triage = [exp_triage]
        exp_lang = meta.get("expected_language", "English")

        print(f"[{idx+1}/{len(ordered_cases)}] Running {cid} ({len(case['turns'])} scripted turns)...")
        bot.reset()

        case_record = {
            "id": cid,
            "status": "in_progress",
            "scripted_turns": case["turns"],
            "expected_triage": exp_triage,
            "expected_language": exp_lang,
            "final_triage": None,
            "final_language": None,
            "triage_match": False,
            "language_match": False,
            "triage_direction": "n/a",
            "provider_used": None,
            "drain_turns_used": 0,
            "error": None,
            "responses": [],
        }

        got_triage = False

        try:
            # Phase 1: Scripted turns
            for turn_idx, turn_text in enumerate(case["turns"]):
                resp = bot.chat(turn_text)
                resp_type = resp.get("type", "unknown")
                case_record["responses"].append({
                    "turn": turn_idx + 1,
                    "kind": "scripted",
                    "input": turn_text,
                    "type": resp_type,
                    "message": resp.get("message", "")[:200],
                })

                if resp_type == "emergency":
                    case_record["final_triage"] = "EMERGENCY"
                    user_lang = getattr(bot, "user_language", {}) or {}
                    case_record["final_language"] = user_lang.get("language", "English")
                    case_record["provider_used"] = "deterministic_emergency"
                    got_triage = True
                    break
                elif resp_type == "service_unavailable":
                    case_record["error"] = "service_unavailable: all providers rate-limited or failed"
                    case_record["status"] = "not_run"
                    rate_limit_halt = True
                    break
                elif resp_type == "answer":
                    case_record["final_triage"] = resp.get("triage", bot.last_triage)
                    user_lang = getattr(bot, "user_language", {}) or {}
                    case_record["final_language"] = resp.get("language") or user_lang.get("language", "English")
                    case_record["provider_used"] = getattr(bot.router, "_sticky", "unknown") or "unknown"
                    got_triage = True
                    break

                time.sleep(DELAY_SECONDS)

            # Phase 2: Drain turns if no triage produced yet
            if not got_triage and not rate_limit_halt:
                for drain_idx in range(MAX_DRAIN_TURNS):
                    case_record["drain_turns_used"] = drain_idx + 1
                    time.sleep(DELAY_SECONDS)
                    resp = bot.chat(DRAIN_REPLY)
                    resp_type = resp.get("type", "unknown")
                    case_record["responses"].append({
                        "turn": len(case["turns"]) + drain_idx + 1,
                        "kind": "drain",
                        "input": DRAIN_REPLY,
                        "type": resp_type,
                        "message": resp.get("message", "")[:200],
                    })

                    if resp_type == "emergency":
                        case_record["final_triage"] = "EMERGENCY"
                        user_lang = getattr(bot, "user_language", {}) or {}
                        case_record["final_language"] = user_lang.get("language", "English")
                        case_record["provider_used"] = "deterministic_emergency"
                        got_triage = True
                        break
                    elif resp_type == "service_unavailable":
                        case_record["error"] = "service_unavailable: all providers rate-limited or failed"
                        case_record["status"] = "not_run"
                        rate_limit_halt = True
                        break
                    elif resp_type == "answer":
                        case_record["final_triage"] = resp.get("triage", bot.last_triage)
                        user_lang = getattr(bot, "user_language", {}) or {}
                        case_record["final_language"] = resp.get("language") or user_lang.get("language", "English")
                        case_record["provider_used"] = getattr(bot.router, "_sticky", "unknown") or "unknown"
                        got_triage = True
                        break

            # Set status and evaluation
            if rate_limit_halt:
                case_record["status"] = "not_run"
            else:
                case_record["status"] = "completed"
                # If still no triage after drain turns, final_triage remains bot.last_triage or None
                if not case_record["final_triage"]:
                    case_record["final_triage"] = bot.last_triage
                    user_lang = getattr(bot, "user_language", {}) or {}
                    case_record["final_language"] = user_lang.get("language", "English")
                    case_record["provider_used"] = getattr(bot.router, "_sticky", "unknown") or "unknown"

                act_tr = case_record["final_triage"]
                if act_tr and act_tr in exp_triage:
                    case_record["triage_match"] = True
                    case_record["triage_direction"] = "correct"
                elif act_tr and act_tr in TRIAGE_ORDER:
                    min_exp = min(TRIAGE_ORDER[t] for t in exp_triage if t in TRIAGE_ORDER)
                    max_exp = max(TRIAGE_ORDER[t] for t in exp_triage if t in TRIAGE_ORDER)
                    act_val = TRIAGE_ORDER[act_tr]
                    if act_val < min_exp:
                        case_record["triage_direction"] = "too_low"
                    elif act_val > max_exp:
                        case_record["triage_direction"] = "too_high"
                    else:
                        case_record["triage_direction"] = "correct"
                        case_record["triage_match"] = True
                else:
                    case_record["triage_direction"] = "error"
                    case_record["triage_match"] = False

                act_lang = case_record["final_language"]
                if act_lang and act_lang.lower() == exp_lang.lower():
                    case_record["language_match"] = True

        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)[:250]}"
            print(f"    ERROR on {cid}: {err_msg}")
            case_record["error"] = err_msg
            if any(term in err_msg.lower() for term in ["rate limit", "429", "quota", "resource_exhausted"]):
                case_record["status"] = "not_run"
                rate_limit_halt = True
            else:
                case_record["status"] = "provider_error"

        results_data["cases"].append(case_record)
        completed_ids.add(cid)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"    -> Status: {case_record['status']} | Provider: {case_record.get('provider_used')} | Triage: {case_record.get('final_triage')} (Exp: {exp_triage}) | Match: {case_record.get('triage_match')} | Drain turns: {case_record.get('drain_turns_used')}")

        time.sleep(DELAY_SECONDS)

    print(f"\nEvaluation loop completed. Results saved to {results_path}.")


if __name__ == "__main__":
    main()
