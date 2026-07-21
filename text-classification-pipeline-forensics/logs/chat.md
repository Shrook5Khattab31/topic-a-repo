# AI Usage Log

Per the assignment's Academic Integrity section, this log records how AI tools were
used and how outputs were verified.

## Entries

### 2026-07-17 to 2026-07-22 - Fatma Hadad - pipeline scaffolding
Tool: Claude / Codex with code execution
Prompt/ask: requested a CPU-only TF-IDF + Logistic Regression project pipeline,
data-splitting utilities using the fixed `split_indices.json`, reusable
preprocessing/model helpers, and artifact-export helpers matching the required
schemas.
Output used: `pipeline/data.py`, `pipeline/model.py`,
`pipeline/preprocess.py`, `pipeline/artifacts.py`, and the experiment scripts in
`experiments/`.
Verification: reran all scripts with `/opt/anaconda3/bin/python`; confirmed split
sizes were train/dev/heldout = 4567/1523/1523, no split overlap, no duplicate ids,
and Ticket 1 held-out F1 = 0.7576, matching the project contract reference 0.7574
within tolerance.

### 2026-07-17 to 2026-07-22 - Fatma Hadad - ticket analysis and validation
Tool: Claude / Codex with code execution
Prompt/ask: requested five ticket investigations covering baseline discrepancy,
normalization, shortcut features, decision-rule/model choice, and data-quality
forensics.
Output used: `tickets/ticket-1-baseline.md` through
`tickets/ticket-5-data-quality.md`, plus validation scripts for determinism,
bootstrap confidence intervals, McNemar tests, tokenizer/vocabulary tracing,
shortcut inspection, threshold oracle diagnostics, and label-fix validation.
Verification: every numeric claim in the ticket files was checked by rerunning
the matching `experiments/ticket*_*.py` script. Mechanism claims were checked
against concrete flipped examples, tokenizer output, fitted vocabulary diffs, and
manual inspection of high-purity keyword examples and hard-negative examples.

### 2026-07-22 - Fatma Hadad - final submission cleanup
Tool: Codex
Prompt/ask: requested a final no-model-logic cleanup of deliverables and
machine-checkable artifacts.
Output used: regenerated `predictions/heldout_predictions.csv`,
`results/summary.csv`, `results/threshold_sweep.csv`,
`results/data_quality_audit.csv`, completed `logs/chat.md`, updated `README.md`,
and exported `report.pdf`.
Verification: reran the full project with `/opt/anaconda3/bin/python`, confirmed
all scripts exited successfully, prediction artifacts have the expected row counts
and no duplicate keys, and `results/summary.csv` includes all five tickets.
