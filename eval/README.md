# Care-AI Evaluation

This directory contains evaluation inputs, execution utilities, and raw test outputs for Care-AI.

## Evaluation Files

- **`test_cases.json`**: Reconstructed test case inputs (author-written, not clinician-reviewed).
- **`results_current.json`**: Raw per-case outputs from evaluating the current build using multi-provider router fallback (emergency-first run order, drain turns enabled).
- **`results_attempt1.json`**: Archived per-case outputs from initial test attempt 1 (halted early by provider quota exhaustion).
- **`score.py`**: Standalone scoring script that computes total cases (n), pipeline errors, triage accuracy with Wilson 95% confidence intervals, emergency recall, under/over-triage counts, language accuracy, and per-class classification metrics directly from `results_current.json`.

## Running the Evaluation and Scoring

To score the current results file:

```bash
python eval/score.py
```

To run the evaluation pipeline against the configured provider:

```bash
python eval/run_evaluation.py
```

Other files in this folder are archived output from an earlier build and are superseded by the results in the main README.
