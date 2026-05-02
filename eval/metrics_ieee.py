"""
eval/metrics_ieee.py — Care-AI IEEE Metrics Analyzer
======================================================
Standalone script: reads any results_*.json and produces full
IEEE-grade metrics + Table I / II / III text output.

Use this to:
  - Regenerate tables after adding more results
  - Compare partial results across days
  - Compute final paper numbers from mixed-provider results
  - Get error analysis breakdown

Usage:
    # Analyze Groq results:
    python eval/metrics_ieee.py --provider groq

    # Analyze all available result files:
    python eval/metrics_ieee.py --all

    # Analyze a specific results JSON:
    python eval/metrics_ieee.py --file eval/results_groq.json

    # Print full error breakdown (debug mode):
    python eval/metrics_ieee.py --provider groq --errors

    # Export metrics as JSON (for further analysis):
    python eval/metrics_ieee.py --provider groq --json
"""

import os, sys, json, argparse, math
from datetime import datetime
from collections import defaultdict

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if EVAL_DIR == os.path.dirname(os.path.abspath(".")):
    EVAL_DIR = "eval"   # fallback if run from project root

TRIAGE_ORDER  = {"SELF_CARE": 0, "SEE_DOCTOR": 1, "URGENT_CARE": 2, "EMERGENCY": 3}
TRIAGE_LEVELS = ["SELF_CARE", "SEE_DOCTOR", "URGENT_CARE", "EMERGENCY"]

CAT_LABELS = {
    "common_low_triage":      "Common / Low Triage",
    "neuro_red_flags":        "Neuro Red Flags",
    "cardiac_red_flags":      "Cardiac Red Flags",
    "multi_turn_socrates":    "Multi-turn SOCRATES",
    "language_tests":         "Language Tests",
    "hallucination_injection": "Hallucination Injection",
}
CAT_ORDER = list(CAT_LABELS.keys())


# ── HELPERS ──────────────────────────────────────────────────

def _pct(n, d):
    return round(n / d * 100, 1) if d and d > 0 else 0.0

def _mean(lst):
    lst = [x for x in lst if x is not None]
    return round(sum(lst) / len(lst), 2) if lst else None

def _load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Supports both raw list and {"results": [...]} formats
    if isinstance(data, list):
        return data, None
    return data.get("results", []), data.get("provider", os.path.basename(path))


# ── CORE METRICS ENGINE ───────────────────────────────────────

def compute_all_metrics(results, provider="unknown"):
    total   = len(results)
    valid   = [r for r in results if not r.get("error")]
    errors  = [r for r in results if r.get("error")]
    n_valid = len(valid)

    # ── OVERALL ──────────────────────────────────────────────
    # Classify errors into execution (infrastructure) vs reasoning (pipeline)
    execution_errors = [r for r in errors if r.get("error_type") in
                        ("rate_limit", "timeout", None)]  # None for legacy results
    reasoning_errors = [r for r in errors if r.get("error_type") in
                        ("parse_error", "pipeline_error", "socrates_stuck")]
    # Legacy fallback: if no error_type field, classify by error text
    for r in errors:
        if r.get("error_type") is None and r.get("error"):
            err_lower = r["error"].lower()
            if any(s in err_lower for s in ["rate limit", "429", "quota", "unavailable"]):
                execution_errors.append(r)
            else:
                reasoning_errors.append(r)
    # Deduplicate
    exec_ids = {r["id"] for r in execution_errors}
    reason_ids = {r["id"] for r in reasoning_errors}

    overall = {
        "provider":               provider,
        "total_cases":            total,
        "completed_cases":        n_valid,
        "error_count":            len(errors),
        "error_rate_pct":         _pct(len(errors), total),
        # Execution metrics (over ALL cases)
        "execution_reliability_pct":  _pct(total - len(exec_ids), total),
        "execution_error_count":      len(exec_ids),
        "reasoning_error_count":      len(reason_ids),
        # Clinical metrics (over ALL cases — raw)
        "triage_accuracy_pct":    _pct(sum(r.get("triage_correct", False)   for r in results), total),
        "language_accuracy_pct":  _pct(sum(r.get("language_correct", False) for r in results), total),
        "hallucination_ok_pct":   _pct(sum(r.get("hallucination_ok", False) for r in results), total),
        # Clinical metrics (over COMPLETED cases only — adjusted)
        "triage_accuracy_completed_pct":   _pct(sum(r.get("triage_correct", False) for r in valid), n_valid),
        "language_accuracy_completed_pct": _pct(sum(r.get("language_correct", False) for r in valid), n_valid),
        "hallucination_ok_completed_pct":  _pct(sum(r.get("hallucination_ok", False) for r in valid), n_valid),
        # Safety
        "under_triage_count":     sum(1 for r in results if r.get("triage_direction") == "too_low"),
        "over_triage_count":      sum(1 for r in results if r.get("triage_direction") == "too_high"),
        "avg_followup_turns":     _mean([r.get("followup_turns", 0) for r in results]),
        "avg_response_time_s":    _mean([r["response_time_s"] for r in results if r.get("response_time_s")]),
        "xai_coverage_pct":       _pct(sum(r.get("xai_explanation_present", False) for r in valid), n_valid),
        "xai_reasoning_pct":      _pct(sum(r.get("xai_has_reasoning", False) for r in valid), n_valid),
    }

    # ── BY CATEGORY ───────────────────────────────────────────
    cats = defaultdict(lambda: {
        "total":0, "triage_correct":0, "lang_correct":0, "hal_ok":0,
        "errors":0, "under_triage":0, "over_triage":0, "xai_present":0,
        "followup_turns":[], "response_times":[], "hal_risks":[]
    })
    for r in results:
        c = cats[r["category"]]
        c["total"] += 1
        if r.get("triage_correct"):   c["triage_correct"] += 1
        if r.get("language_correct"): c["lang_correct"] += 1
        if r.get("hallucination_ok"): c["hal_ok"] += 1
        if r.get("error"):            c["errors"] += 1
        if r.get("triage_direction") == "too_low":  c["under_triage"] += 1
        if r.get("triage_direction") == "too_high": c["over_triage"] += 1
        if r.get("xai_explanation_present"):        c["xai_present"] += 1
        c["followup_turns"].append(r.get("followup_turns", 0))
        if r.get("response_time_s"): c["response_times"].append(r["response_time_s"])
        if r.get("hallucination_risk"): c["hal_risks"].append(r["hallucination_risk"])

    by_category = {}
    for cat in CAT_ORDER:
        if cat not in cats: continue
        c = cats[cat]; n = c["total"]
        # Hallucination risk distribution
        hal_dist = defaultdict(int)
        for h in c["hal_risks"]: hal_dist[h] += 1
        by_category[cat] = {
            "total":               n,
            "triage_correct":      c["triage_correct"],
            "triage_accuracy_pct": _pct(c["triage_correct"], n),
            "lang_correct":        c["lang_correct"],
            "lang_accuracy_pct":   _pct(c["lang_correct"], n),
            "hal_ok":              c["hal_ok"],
            "hal_ok_pct":          _pct(c["hal_ok"], n),
            "under_triage":        c["under_triage"],
            "over_triage":         c["over_triage"],
            "errors":              c["errors"],
            "xai_coverage_pct":    _pct(c["xai_present"], n),
            "avg_followup_turns":  _mean(c["followup_turns"]),
            "avg_response_time_s": _mean(c["response_times"]),
            "hal_risk_dist":       dict(hal_dist),
        }

    # ── TRIAGE CLASSIFICATION (P/R/F1) ───────────────────────
    triage_clf = {}
    for level in TRIAGE_LEVELS:
        tp = fp = fn = tn = 0
        for r in results:
            if r.get("error"): continue
            pred = r.get("final_triage")
            correct = r.get("triage_correct", False)
            if pred == level:
                if correct: tp += 1
                else:       fp += 1
            else:
                if not correct and r.get("triage_direction") != "n/a": fn += 1
                else: tn += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        triage_clf[level] = {
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "support": sum(1 for r in results if r.get("final_triage") == level),
            "tp": tp, "fp": fp, "fn": fn,
        }
    lvls = [l for l in TRIAGE_LEVELS if triage_clf[l]["support"] > 0]
    total_sup = sum(triage_clf[l]["support"] for l in lvls)
    triage_clf["macro_avg"] = {
        "precision": round(_mean([triage_clf[l]["precision"] for l in lvls]) or 0, 3),
        "recall":    round(_mean([triage_clf[l]["recall"]    for l in lvls]) or 0, 3),
        "f1":        round(_mean([triage_clf[l]["f1"]        for l in lvls]) or 0, 3),
    }
    triage_clf["weighted_avg"] = {
        "f1": round(
            sum(triage_clf[l]["f1"] * triage_clf[l]["support"] for l in lvls) / total_sup
            if total_sup else 0, 3
        )
    }

    # ── XAI METRICS ───────────────────────────────────────────
    xai = {
        "responses_with_xai":        sum(r.get("xai_explanation_present", False) for r in valid),
        "xai_with_reasoning":         sum(r.get("xai_has_reasoning", False) for r in valid),
        "xai_coverage_pct":           _pct(sum(r.get("xai_explanation_present", False) for r in valid), n_valid),
        "xai_reasoning_quality_pct":  _pct(sum(r.get("xai_has_reasoning", False) for r in valid), n_valid),
        "total_valid_answers":        n_valid,
    }

    # ── CONFIDENCE CALIBRATION ────────────────────────────────
    buckets = {"high":{"correct":0,"total":0}, "medium":{"correct":0,"total":0}, "low":{"correct":0,"total":0}}
    for r in valid:
        try:
            c = float(r.get("confidence") or 0)
        except (TypeError, ValueError):
            continue
        b = "high" if c >= 0.75 else ("medium" if c >= 0.45 else "low")
        buckets[b]["total"] += 1
        if r.get("triage_correct"): buckets[b]["correct"] += 1
    calibration = {
        b: {"count": d["total"], "accuracy_pct": _pct(d["correct"], d["total"]) if d["total"] else None}
        for b, d in buckets.items()
    }

    # ── SAFETY METRICS ─────────────────────────────────────────
    under_cases = [r for r in results if r.get("triage_direction") == "too_low"]
    ut_by_cat = defaultdict(int)
    for r in under_cases: ut_by_cat[r["category"]] += 1
    safety = {
        "under_triage_total":  len(under_cases),
        "zero_under_triage":   len(under_cases) == 0,
        "under_triage_by_cat": dict(ut_by_cat),
        "under_triage_cases":  [
            {"id": r["id"], "category": r["category"],
             "predicted": r["final_triage"], "description": r["description"]}
            for r in under_cases
        ],
    }

    # ── ERROR ANALYSIS ─────────────────────────────────────────
    error_types = defaultdict(list)
    for r in errors:
        err = r.get("error", "")
        if "rate limit" in err.lower() or "429" in err or "quota" in err.lower():
            error_types["rate_limit"].append(r["id"])
        elif "stuck" in err.lower():
            error_types["socrates_stuck"].append(r["id"])
        elif "timeout" in err.lower():
            error_types["timeout"].append(r["id"])
        else:
            error_types["other"].append(r["id"])
    error_analysis = {
        "total_errors": len(errors),
        "by_type": dict(error_types),
        "by_category": {cat: sum(1 for r in errors if r["category"] == cat)
                        for cat in CAT_ORDER if any(r["category"] == cat for r in errors)},
    }

    return {
        "overall":                overall,
        "by_category":            by_category,
        "triage_classification":  triage_clf,
        "xai_coverage":           xai,
        "confidence_calibration": calibration,
        "safety_metrics":         safety,
        "error_analysis":         error_analysis,
    }


# ── IEEE TABLE PRINTER ────────────────────────────────────────

def print_ieee_tables(metrics, provider, show_errors=False):
    o   = metrics["overall"]
    bc  = metrics["by_category"]
    tc  = metrics["triage_classification"]
    xai = metrics["xai_coverage"]
    cal = metrics["confidence_calibration"]
    saf = metrics["safety_metrics"]
    ea  = metrics["error_analysis"]

    lines = []
    def L(s=""):
        lines.append(s)
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode("ascii", errors="replace").decode("ascii"))

    L("=" * 76)
    L(f"CARE-AI IEEE METRICS  |  Provider: {provider.upper()}")
    L(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L(f"Cases: {o['completed_cases']}/{o['total_cases']} completed, "
      f"{o['error_count']} errors ({o['error_rate_pct']}%)")
    L("=" * 76)

    # TABLE I
    L()
    L("TABLE I — PER-CATEGORY CLINICAL PERFORMANCE")
    L("-" * 76)
    L(f"{'Category':<26} {'N':>3} {'Triage%':>8} {'Lang%':>6} {'Hal-OK%':>8} "
      f"{'UnderT':>7} {'XAI%':>6} {'AvgFU':>6} {'AvgRT':>7}")
    L("-" * 76)
    for cat in CAT_ORDER:
        if cat not in bc: continue
        c = bc[cat]
        label = CAT_LABELS[cat][:25]
        ut_str = f"{'⚠' if c['under_triage'] > 0 else ''}{c['under_triage']}"
        avg_rt = f"{c['avg_response_time_s']}s" if c['avg_response_time_s'] else "N/A"
        L(f"{label:<26} {c['total']:>3} {c['triage_accuracy_pct']:>8} "
          f"{c['lang_accuracy_pct']:>6} {c['hal_ok_pct']:>8} "
          f"{ut_str:>7} {c['xai_coverage_pct']:>6} "
          f"{str(c['avg_followup_turns']):>6} {avg_rt:>7}")
    L("-" * 76)
    avg_rt_str = f"{o['avg_response_time_s']}s" if o['avg_response_time_s'] else "N/A"
    ut_str = f"{'⚠' if o['under_triage_count'] > 0 else ''}{o['under_triage_count']}"
    L(f"{'OVERALL':<26} {o['total_cases']:>3} {o['triage_accuracy_pct']:>8} "
      f"{o['language_accuracy_pct']:>6} {o['hallucination_ok_pct']:>8} "
      f"{ut_str:>7} {o['xai_coverage_pct']:>6} "
      f"{str(o['avg_followup_turns']):>6} {avg_rt_str:>7}")
    L()
    L()
    ut_label = "\u2705 ZERO SAFETY FAILURES" if saf["zero_under_triage"] else "\u26a0 {} SAFETY FAILURES".format(saf["under_triage_total"])
    L("  Under-triage (too_low): " + ut_label)
    L()
    L(f"  Execution reliability: {o.get('execution_reliability_pct', 'N/A')}% "
      f"({o.get('execution_error_count', '?')} infra errors, "
      f"{o.get('reasoning_error_count', '?')} reasoning errors)")
    L(f"  Clinical accuracy (completed only): "
      f"Triage={o.get('triage_accuracy_completed_pct', 'N/A')}% | "
      f"Lang={o.get('language_accuracy_completed_pct', 'N/A')}% | "
      f"Hal-OK={o.get('hallucination_ok_completed_pct', 'N/A')}%")

    # Show under-triage details if any
    if saf["under_triage_cases"]:
        L()
        L("  UNDER-TRIAGE CASES (must review):")
        for uc in saf["under_triage_cases"]:
            L(f"    ⚠ {uc['id']} [{uc['category']}]: "
              f"predicted {uc['predicted']} | {uc['description'][:55]}")

    # TABLE II
    L()
    L("TABLE II — TRIAGE CLASSIFICATION: PRECISION / RECALL / F1")
    L("-" * 64)
    L(f"{'Triage Level':<16} {'Precision':>10} {'Recall':>8} {'F1-Score':>9} {'Support':>8}")
    L("-" * 64)
    for level in TRIAGE_LEVELS:
        m = tc.get(level, {})
        L(f"{level:<16} {m.get('precision',0):>10.3f} {m.get('recall',0):>8.3f} "
          f"{m.get('f1',0):>9.3f} {m.get('support',0):>8}")
    L("-" * 64)
    ma = tc.get("macro_avg",    {})
    wa = tc.get("weighted_avg", {})
    L(f"{'Macro avg':<16} {ma.get('precision',0):>10.3f} {ma.get('recall',0):>8.3f} "
      f"{ma.get('f1',0):>9.3f} {o['completed_cases']:>8}")
    L(f"{'Weighted avg':<16} {'':>10} {'':>8} {wa.get('f1',0):>9.3f} {o['completed_cases']:>8}")
    L()

    # XAI
    L("XAI EXPLAINABILITY COVERAGE")
    L("-" * 50)
    L(f"  Responses with XAI:        {xai['responses_with_xai']}/{xai['total_valid_answers']} "
      f"({xai['xai_coverage_pct']}%)")
    L(f"  XAI with causal reasoning: {xai['xai_with_reasoning']}/{xai['total_valid_answers']} "
      f"({xai['xai_reasoning_quality_pct']}%)")
    L()

    # Confidence calibration
    L("CONFIDENCE CALIBRATION")
    L("-" * 44)
    L(f"  {'Bucket':<10} {'Count':>6} {'Triage Accuracy':>16}")
    for b in ["high", "medium", "low"]:
        c = cal.get(b, {})
        acc = f"{c['accuracy_pct']}%" if c.get("accuracy_pct") is not None else "N/A"
        L(f"  {b.capitalize():<10} {c.get('count',0):>6} {acc:>16}")
    L()

    # Error analysis
    L("ERROR / SKIP ANALYSIS")
    L("-" * 44)
    L(f"  Total errors:      {ea['total_errors']} / {o['total_cases']}")
    for etype, ids in ea["by_type"].items():
        L(f"  {etype:<18} {len(ids)}  ({', '.join(ids[:5])}{'...' if len(ids) > 5 else ''})")
    L()
    if ea["total_errors"] > 0 and ea["total_errors"] == sum(1 for r in [] if r.get("error")):
        L("  NOTE: All errors are rate-limit skips — pipeline itself has no failures.")
        L("  Re-run these cases tomorrow with: python eval/run_eval_v4.py --provider " +
          provider)
    L()

    # Paper-ready summary
    L("=" * 76)
    L("PAPER-READY RESULTS SENTENCE")
    L("=" * 76)
    L()
    L(f"Care-AI achieved {o['triage_accuracy_pct']}% triage accuracy ({o['completed_cases']}")
    L(f"of {o['total_cases']} cases evaluated) with a macro-F1 of {ma.get('f1','N/A')} across")
    L(f"four triage levels. The system recorded {saf['under_triage_total']} under-triage")
    L(f"events (safety-critical failures), a language detection accuracy of")
    L(f"{o['language_accuracy_pct']}%, and XAI explanation coverage of {o['xai_coverage_pct']}%.")
    L(f"Hallucination risk was within acceptable bounds (LOW or MEDIUM) in")
    L(f"{o['hallucination_ok_pct']}% of responses, with a mean response latency of")
    L(f"{o['avg_response_time_s']}s across all categories.")
    L()

    return "\n".join(lines)


# ── MAIN ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Care-AI IEEE Metrics Analyzer")
    p.add_argument("--provider", default=None,
                   choices=["groq", "openrouter", "gemini", "ollama"],
                   help="Load results_{provider}.json from eval/")
    p.add_argument("--file", default=None,
                   help="Load a specific results JSON file")
    p.add_argument("--all", action="store_true",
                   help="Analyze all available results_*.json files")
    p.add_argument("--errors", action="store_true",
                   help="Show detailed error breakdown")
    p.add_argument("--json", action="store_true",
                   help="Export full metrics as JSON")
    return p.parse_args()


def main():
    args = parse_args()

    files_to_analyze = []

    if args.file:
        files_to_analyze.append((args.file, os.path.splitext(os.path.basename(args.file))[0]))
    elif args.provider:
        path = os.path.join(EVAL_DIR, f"results_{args.provider}.json")
        if not os.path.exists(path):
            print(f"No results found at {path}")
            print(f"Run: python eval/run_eval_v4.py --provider {args.provider}")
            sys.exit(1)
        files_to_analyze.append((path, args.provider))
    elif args.all:
        import glob
        for path in sorted(glob.glob(os.path.join(EVAL_DIR, "results_*.json"))):
            provider = os.path.basename(path).replace("results_", "").replace(".json", "")
            files_to_analyze.append((path, provider))
        if not files_to_analyze:
            print(f"No results_*.json files found in {EVAL_DIR}")
            sys.exit(1)
    else:
        # Default: try groq
        path = os.path.join(EVAL_DIR, "results_groq.json")
        if os.path.exists(path):
            files_to_analyze.append((path, "groq"))
        else:
            print("No results found. Run eval first:")
            print("  python eval/run_eval_v4.py --provider groq --batch 6")
            sys.exit(1)

    for fpath, provider in files_to_analyze:
        results, file_provider = _load(fpath)
        if not results:
            print(f"No results in {fpath}")
            continue
        pname = file_provider or provider
        metrics = compute_all_metrics(results, pname)
        text = print_ieee_tables(metrics, pname, show_errors=args.errors)

        # Save tables
        out_path = os.path.join(EVAL_DIR, f"ieee_tables_{pname}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n  Saved -> {out_path}")

        if args.json:
            jpath = os.path.join(EVAL_DIR, f"ieee_metrics_{pname}.json")
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Saved -> {jpath}")


if __name__ == "__main__":
    main()
