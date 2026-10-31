"""
eval/run_evaluation.py — Evaluates Care-AI on eval/test_cases.json

Runs each case through the real CareAI pipeline using a single configured provider (Gemini).
Saves results per case to eval/results_current.json, with resume capability.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# Ensure root is in sys.path
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

DELAY_SECONDS = 4.5  # Bounded pause between pipeline calls to prevent rate limits


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
        cases_input = json.load(f)["cases"]

    with open(test_eval_path, "r", encoding="utf-8") as f:
        eval_cases = {c["id"]: c for c in json.load(f)["cases"]}

    # Load existing results if resuming
    results_data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "commit": get_git_commit(),
        "provider": "gemini",
        "model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "cases": [],
    }

    completed_ids = set()
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                if isinstance(old_data, dict) and "cases" in old_data:
                    results_data = old_data
                    completed_ids = {c["id"] for c in results_data["cases"]}
                    print(f"Resuming: found {len(completed_ids)} already evaluated cases.")
        except Exception as e:
            print(f"Could not read existing results, starting fresh: {e}")

    # Build knowledge chunks once
    print("Initializing knowledge chunks...")
    chunks, sev_map, prec_map = build_all_chunks(os.path.join(WORKSPACE_ROOT, "data"))

    # Configure strictly single provider: Gemini
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        print("Error: GEMINI_API_KEY not found in .env")
        sys.exit(1)

    router_cfg = {
        "primary_provider": "gemini",
        "gemini_api_key": gemini_key,
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "openrouter_api_key": "",
        "groq_api_key": "",
        "ollama_host": "http://localhost:99999",
    }

    bot = CareAI(
        knowledge_chunks=chunks,
        severity_map=sev_map,
        router_config=router_cfg,
        precaution_map=prec_map,
    )

    print(f"\nStarting evaluation of {len(cases_input)} cases using {results_data['provider']} ({results_data['model']})...\n")

    for idx, case in enumerate(cases_input):
        cid = case["id"]
        if cid in completed_ids:
            print(f"[{idx+1}/{len(cases_input)}] {cid}: already completed, skipping.")
            continue

        meta = eval_cases.get(cid, {})
        exp_triage = meta.get("expected_triage", [])
        if isinstance(exp_triage, str):
            exp_triage = [exp_triage]
        exp_lang = meta.get("expected_language", "English")

        print(f"[{idx+1}/{len(cases_input)}] Running {cid} ({len(case['turns'])} turns)...")
        bot.reset()

        case_record = {
            "id": cid,
            "turns": case["turns"],
            "expected_triage": exp_triage,
            "expected_language": exp_lang,
            "final_triage": None,
            "final_language": None,
            "triage_match": False,
            "language_match": False,
            "triage_direction": "n/a",
            "error": None,
            "responses": [],
        }

        halt_on_rate_limit = False
        try:
            for turn_idx, turn_text in enumerate(case["turns"]):
                resp = bot.chat(turn_text)
                resp_type = resp.get("type", "unknown")
                case_record["responses"].append({
                    "turn": turn_idx + 1,
                    "input": turn_text,
                    "type": resp_type,
                    "message": resp.get("message", "")[:200],
                })

                if resp_type == "emergency":
                    case_record["final_triage"] = "EMERGENCY"
                    user_lang = getattr(bot, "user_language", {}) or {}
                    case_record["final_language"] = user_lang.get("language", "English")
                    break
                elif resp_type == "service_unavailable":
                    case_record["error"] = "service_unavailable"
                    halt_on_rate_limit = True
                    break
                elif resp_type == "answer":
                    case_record["final_triage"] = resp.get("triage", bot.last_triage)
                    user_lang = getattr(bot, "user_language", {}) or {}
                    case_record["final_language"] = resp.get("language") or user_lang.get("language", "English")
                    break
                elif resp_type == "followup_check":
                    # Pausing before next turn
                    time.sleep(DELAY_SECONDS)
                else:
                    time.sleep(DELAY_SECONDS)

            if not case_record["final_triage"] and not case_record["error"]:
                case_record["final_triage"] = bot.last_triage
                user_lang = getattr(bot, "user_language", {}) or {}
                case_record["final_language"] = user_lang.get("language", "English")

            # Determine triage match & direction
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

            # Determine language match
            act_lang = case_record["final_language"]
            if act_lang and act_lang.lower() == exp_lang.lower():
                case_record["language_match"] = True

        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)[:250]}"
            print(f"    ERROR on {cid}: {err_msg}")
            case_record["error"] = err_msg
            if any(term in err_msg.lower() for term in ["rate limit", "429", "quota", "resource_exhausted"]):
                halt_on_rate_limit = True

        # Append and save per case
        results_data["cases"].append(case_record)
        completed_ids.add(cid)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"    -> Triage: {case_record['final_triage']} (Expected: {exp_triage}) | Match: {case_record['triage_match']} | Lang: {case_record['final_language']}")

        if halt_on_rate_limit:
            print("\n⚠️ Rate limit encountered. Saved progress. Halting run to allow resumption later.")
            break

        time.sleep(DELAY_SECONDS)

    print(f"\nEvaluation finished. Total completed cases in {results_path}: {len(results_data['cases'])}")


if __name__ == "__main__":
    main()
