# CARE-AI Evaluation Suite & Integrity Audit

> **Status Notice: Withdrawn Historical Claims**
> The evaluation metrics, tables, and summary figures contained in this directory represent **unverified, exploratory historical artifacts**. They were not reliably established and **must not be used as evidence of clinical accuracy, reliability, or safety**.
>
> All raw files are preserved in this directory for archival and audit transparency.

---

## 1. Directory Contents and Artifact Inventory

| Artifact | Format | Description | Integrity & Verification Status |
|---|---|---|---|
| `CARE_AI_results_updated.xlsx` | Spreadsheet (31 rows) | Case-level breakdown of 30 test cases (`TC001`–`TC030`) | Exploratory 30-case log across mixed cloud providers; not reproducible offline; contains recorded under-triage in `TC023`. |
| `CARE_AI_tables.txt` | Plain text | Formatted summary tables (Tables I–V) matching historical README claims | Unverified summary; contradicts `summary_final.csv` on under-triage count. |
| `summary_final.csv` | CSV | 6-metric summary table | **Contradictory**: Reports `Under-triage: 0`, conflicting with `CARE_AI_tables.txt` (`Under-triage: 1`). |
| `test_cases_evaluated.json` | JSON (30 cases) | JSON export of 30 test cases with summary block | **Contradictory**: Reports `hallucination_ok_rate_pct: 90.0%` (vs. 66.7% in tables) and `under_triage_count: 0` (vs. 1 in tables). |
| `run_eval_v5.py` | Python script (1,572 lines) | Evaluation runner harness | Incomplete provenance: expects `test_cases.json` (missing from repository) and requires live external paid APIs; cannot execute offline. |
| `metrics_ieee.py` | Python script (484 lines) | Standalone IEEE table formatter | Formatter script that computes metrics from JSON results files. |

---

## 2. Integrity Audit Findings

An internal evaluation-readiness audit identified critical methodological and data-integrity defects:

### A. Incomplete 100-Case Evaluation & Synthesized Metrics
- **Finding:** Development documentation planned a 100-case automated clinical test suite (including hallucination injection tests `TC046`–`TC050` referenced in `run_eval_v5.py`).
- **Disclosed Cause:** The intended 100-case evaluation was halted due to upstream cloud LLM API rate limits. Summary evaluation numbers were subsequently synthesized rather than computed from a complete, reproducible 100-case execution.
- **Retained Reality:** Only 30 cases (`TC001`–`TC030`) exist in the repository.

### B. Mislabeled Explainability Scoring
- **Finding:** Public-facing documentation described the "Mean Explainability Score (2.38 / 3.0)" as an *"independent human rating of Cause → Reasoning → Recommendation clarity"*.
- **Verified Code:** `run_eval_v5.py` lines 10–11 and its internal scoring function define `xai_quality_score` as a programmatic regex/keyword heuristic awarding 0 to 3 points for the presence of structural text headings and causal transition words.
- **Correction:** The metric was entirely automated code output, **not** an independent human rating.

### C. Contradictory Safety and Under-Triage Claims
- **Finding:** Public documentation claimed "zero under-triage across all severe emergency test presentations" and `summary_final.csv` reported `Under-triage: 0`.
- **Verified Code & Artifacts:**
  - In `eval/CARE_AI_results_updated.xlsx`, test case `TC023` (stiff neck + fever / suspected meningitis) resulted in `URGENT_CARE` instead of `EMERGENCY` (`triage_direction: too_low`, recorded as an under-triage).
  - `CARE_AI_tables.txt` Table I records `UnderT = 1` and Table V records `Under-triage: 1`.
  - Claiming "zero under-triage" was directly contradicted by the project's own retained 30-case artifact.

### D. Missing Test Battery Input & Offline Non-Reproducibility
- **Finding:** `run_eval_v5.py` requires `eval/test_cases.json` as its primary input (`TEST_CASES_PATH = os.path.join(EVAL_DIR, "test_cases.json")`). This file is not present in the repository.
- **API Dependency:** The script cannot run offline or without live commercial API credentials for Groq, OpenRouter, and Gemini.

---

## 3. Preserved Artifact Policy

In accordance with scientific and software documentation integrity standards:
1. **No Artifact Deletion:** Retained files (`CARE_AI_results_updated.xlsx`, `test_cases_evaluated.json`, `CARE_AI_tables.txt`, `summary_final.csv`) have **not** been deleted or altered to hide historical claims.
2. **Explicit Disqualification:** These figures are formally disqualified from serving as validated benchmarks of Care-AI's operational or clinical performance.
3. **Research Classification:** Care-AI is strictly an academic research prototype. It is not clinically validated, and no clinical efficacy or safety claims are substantiated by these historical files.
