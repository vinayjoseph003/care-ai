"""
eval/score.py — Standalone metrics calculator for Care-AI evaluation results.

Computes metrics exclusively from eval/results_current.json:
  - Total cases, completed cases, provider-error cases, not_run cases
  - Triage accuracy with Wilson 95% CI (completed cases only; no triage after drain turns counts as incorrect)
  - Recall on EMERGENCY cases (completed emergency cases)
  - Under-triage and over-triage counts with case IDs (completed cases)
  - Language accuracy per language (completed cases)
  - Per-class precision, recall, F1 only for classes with support >= 5 (completed cases)
  - Provider breakdown (how many cases each provider answered)
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
    z = 1.959963984540054  # 95% two-sided
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
    date_str = data.get("date", "unknown")
    commit_str = data.get("commit", "unknown")

    total_n = len(cases)
    if total_n == 0:
        print("No cases found in results file.")
        return

    # Categorize cases
    completed_cases = []
    provider_error_cases = []
    not_run_cases = []

    for c in cases:
        status = c.get("status")
        # If status is explicit
        if status == "not_run":
            not_run_cases.append(c)
        elif status == "provider_error":
            provider_error_cases.append(c)
        elif status == "completed":
            completed_cases.append(c)
        else:
            # Fallback categorization if status not set
            if c.get("error") and "rate limit" in str(c.get("error")).lower():
                not_run_cases.append(c)
            elif c.get("error"):
                provider_error_cases.append(c)
            else:
                completed_cases.append(c)

    n_comp = len(completed_cases)
    n_err = len(provider_error_cases)
    n_not_run = len(not_run_cases)

    # Provider breakdown
    provider_counts = defaultdict(int)
    for c in completed_cases:
        p = c.get("provider_used") or "unknown"
        provider_counts[p] += 1

    # Metrics on completed cases ONLY
    triage_correct_count = 0
    under_triage_ids = []
    over_triage_ids = []
    no_triage_ids = []

    emergency_cases = []
    emergency_tp = 0

    lang_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    classes = ["SELF_CARE", "SEE_DOCTOR", "URGENT_CARE", "EMERGENCY"]
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    class_support = defaultdict(int)

    for c in completed_cases:
        cid = c.get("id", "")
        exp_triage = c.get("expected_triage", [])
        if isinstance(exp_triage, str):
            exp_triage = [exp_triage]
        act_triage = c.get("final_triage")
        exp_lang = c.get("expected_language", "English")
        act_lang = c.get("final_language", "")
        tdir = c.get("triage_direction", "")

        # A case that finishes with no triage after drain turns counts as incorrect
        if not act_triage:
            no_triage_ids.append(cid)

        is_correct = bool(act_triage and (act_triage in exp_triage))
        if is_correct:
            triage_correct_count += 1

        if tdir == "too_low":
            under_triage_ids.append(cid)
        elif tdir == "too_high":
            over_triage_ids.append(cid)

        # Emergency recall
        if "EMERGENCY" in exp_triage:
            emergency_cases.append(cid)
            if act_triage == "EMERGENCY":
                emergency_tp += 1

        # Language tracking
        lang_stats[exp_lang]["total"] += 1
        if act_lang and (act_lang.lower() == exp_lang.lower()):
            lang_stats[exp_lang]["correct"] += 1

        # Per-class accounting
        for cls in classes:
            if cls in exp_triage:
                class_support[cls] += 1

        if act_triage in classes:
            if act_triage in exp_triage:
                class_tp[act_triage] += 1
            else:
                class_fp[act_triage] += 1

        for cls in exp_triage:
            if cls in classes:
                if not act_triage or act_triage not in exp_triage:
                    class_fn[cls] += 1

    p_acc, ci_low, ci_high = wilson_score_interval(triage_correct_count, n_comp)
    em_recall = (emergency_tp / len(emergency_cases)) if emergency_cases else 0.0

    # Print Formatted Report
    print("=" * 68)
    print("  CARE-AI EVALUATION METRICS REPORT")
    print("=" * 68)
    print(f"Date                 : {date_str}")
    print(f"Commit               : {commit_str}")
    print(f"Total Cases Defined  : {total_n}")
    print(f"Completed Cases (n)  : {n_comp}")
    print(f"Provider Error Cases : {n_err} -> {[c.get('id') for c in provider_error_cases]}")
    print(f"Not Run Cases        : {n_not_run} -> {[c.get('id') for c in not_run_cases]}")
    print("-" * 68)
    print("Provider Distribution (Completed Cases):")
    for p, count in sorted(provider_counts.items()):
        print(f"  - {p:<25}: {count} cases")
    print("-" * 68)
    print(f"Triage Accuracy      : {triage_correct_count}/{n_comp} ({p_acc * 100:.1f}%)")
    print(f"Wilson 95% CI        : [{ci_low * 100:.1f}%, {ci_high * 100:.1f}%]")
    if no_triage_ids:
        print(f"No Triage After Drain: {len(no_triage_ids)} cases (counted incorrect) -> {no_triage_ids}")
    print("-" * 68)
    print(f"Emergency Recall     : {emergency_tp}/{len(emergency_cases)} ({em_recall * 100:.1f}%) [Support: {len(emergency_cases)}]")
    print(f"Under-triage Count   : {len(under_triage_ids)} cases -> {under_triage_ids}")
    print(f"Over-triage Count    : {len(over_triage_ids)} cases -> {over_triage_ids}")
    print("-" * 68)
    print("Language Accuracy by Language:")
    for lang, s in sorted(lang_stats.items()):
        l_acc = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0.0
        print(f"  - {lang:10s}: {s['correct']}/{s['total']} ({l_acc:.1f}%)")
    print("-" * 68)
    print("Per-Class Metrics (Support >= 5 only):")
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
    print("=" * 68)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results_current.json")
    score_results(path)
