"""
eval/score.py — Standalone metrics calculator for Care-AI evaluation results.

Computes metrics exclusively from eval/results_current.json:
  - n (total cases)
  - errors count
  - triage accuracy with Wilson 95% confidence interval
  - recall on EMERGENCY cases
  - under-triage and over-triage counts with case IDs
  - language accuracy per language
  - per-class precision, recall, and F1 only for classes with support >= 5
A failed or errored case counts as incorrect.
"""

import os
import sys
import json
import math
from collections import defaultdict


def wilson_score_interval(k: int, n: int, confidence: float = 0.95):
    """Compute the Wilson score 95% confidence interval for proportion k / n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    # z for 95% two-sided confidence
    z = 1.959963984540054
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + (z2 / (4 * n * n)))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return p, lower, upper


def score_results(results_path: str):
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        sys.exit(1)

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data.get("cases", [])
    provider = data.get("provider", "unknown")
    model = data.get("model", "unknown")
    date_str = data.get("date", "unknown")
    commit_str = data.get("commit", "unknown")

    n = len(cases)
    if n == 0:
        print("No cases found in results file.")
        return

    # Counts
    errors = []
    triage_correct_count = 0
    under_triage_ids = []
    over_triage_ids = []

    # Emergency recall tracking
    emergency_cases = []
    emergency_tp = 0

    # Language tracking: {lang: {"total": int, "correct": int}}
    lang_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    # Class-level tracking for precision/recall/F1
    # classes: SELF_CARE, SEE_DOCTOR, URGENT_CARE, EMERGENCY
    classes = ["SELF_CARE", "SEE_DOCTOR", "URGENT_CARE", "EMERGENCY"]
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    class_support = defaultdict(int)

    for c in cases:
        cid = c.get("id", "")
        exp_triage = c.get("expected_triage", [])
        if isinstance(exp_triage, str):
            exp_triage = [exp_triage]
        act_triage = c.get("final_triage")
        exp_lang = c.get("expected_language", "English")
        act_lang = c.get("final_language", "")
        err = c.get("error")
        tdir = c.get("triage_direction", "")

        # Errors
        if err or not act_triage:
            errors.append((cid, err or "No triage output"))

        # Triage accuracy (errored cases count as incorrect)
        if not err and act_triage and (act_triage in exp_triage):
            triage_correct_count += 1

        # Under-triage / over-triage
        if tdir == "too_low":
            under_triage_ids.append(cid)
        elif tdir == "too_high":
            over_triage_ids.append(cid)

        # Emergency recall
        if "EMERGENCY" in exp_triage:
            emergency_cases.append(cid)
            if not err and act_triage == "EMERGENCY":
                emergency_tp += 1

        # Language tracking
        lang_stats[exp_lang]["total"] += 1
        if not err and act_lang and (act_lang.lower() == exp_lang.lower()):
            lang_stats[exp_lang]["correct"] += 1

        # Per-class P/R/F1 accounting
        # Support is counted if class is in expected_triage
        for cls in classes:
            if cls in exp_triage:
                class_support[cls] += 1

        if not err and act_triage in classes:
            if act_triage in exp_triage:
                class_tp[act_triage] += 1
            else:
                class_fp[act_triage] += 1

        # False negative: case expected this class, but output was not in expected_triage
        for cls in exp_triage:
            if cls in classes:
                if err or act_triage not in exp_triage:
                    class_fn[cls] += 1

    # Triage accuracy Wilson CI
    p_acc, ci_low, ci_high = wilson_score_interval(triage_correct_count, n)

    # Output formatted report
    print("=" * 65)
    print("  CARE-AI CURRENT EVALUATION REPORT")
    print("=" * 65)
    print(f"Date          : {date_str}")
    print(f"Commit        : {commit_str}")
    print(f"Provider/Model: {provider} ({model})")
    print(f"Total Cases (n): {n}")
    print(f"Errors        : {len(errors)}" + (f" {errors}" if errors else ""))
    print("-" * 65)
    print(f"Triage Accuracy: {triage_correct_count}/{n} ({p_acc * 100:.1f}%)")
    print(f"Wilson 95% CI  : [{ci_low * 100:.1f}%, {ci_high * 100:.1f}%]")
    print("-" * 65)

    em_recall = (emergency_tp / len(emergency_cases)) if emergency_cases else 0.0
    print(f"Emergency Recall: {emergency_tp}/{len(emergency_cases)} ({em_recall * 100:.1f}%)")
    print(f"Under-triage Count: {len(under_triage_ids)} cases -> {under_triage_ids}")
    print(f"Over-triage Count : {len(over_triage_ids)} cases -> {over_triage_ids}")
    print("-" * 65)

    print("Language Accuracy by Language:")
    for lang, s in sorted(lang_stats.items()):
        l_acc = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0.0
        print(f"  • {lang:10s}: {s['correct']}/{s['total']} ({l_acc:.1f}%)")
    print("-" * 65)

    print("Per-Class Metrics (Classes with Support >= 5 only):")
    print(f"{'Class':<14} {'Support':<9} {'Precision':<11} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 54)
    for cls in classes:
        sup = class_support[cls]
        if sup >= 5:
            tp = class_tp[cls]
            fp = class_fp[cls]
            fn = class_fn[cls]
            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            print(f"{cls:<14} {sup:<9} {prec * 100:>6.1f}%    {rec * 100:>6.1f}%   {f1:>7.3f}")
        else:
            print(f"{cls:<14} {sup:<9} (excluded: support < 5)")
    print("=" * 65)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results_current.json")
    score_results(path)
