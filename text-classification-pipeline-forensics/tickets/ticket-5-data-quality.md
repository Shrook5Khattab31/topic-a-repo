# Ticket 5: Data-Quality and Error Ticket

## Hypothesis
Exact-duplicate tweets in `train_ids` with conflicting labels are likely
annotation noise, and correcting them to a majority-vote label should be a safe,
strictly-helpful fix — a small, well-targeted data cleaning step with low downside
risk.

## Intended lever
A training-label correction: for every exact-text duplicate group in `train_ids`
with disagreeing labels, relabel all copies to the majority-vote label, leaving
dev/heldout completely untouched, then retrain the unchanged reference-repro model
and compare.

## Controlled experiment
The 12-row majority-vote fix (from the audit findings below) was applied to
`train_ids` only. The unchanged reference-repro pipeline (same vectorizer
settings, same seed, same `C`) was retrained from scratch on the fixed training
data and evaluated on both dev and heldout.

## Audit findings (context for the controlled experiment)

**Exact-duplicate label conflicts (train only):** 11 distinct texts appear more
than once in `train_ids` with disagreeing labels, affecting 12 individual rows
whose label disagrees with their group's majority. Full list with proposed labels
is in `results/data_quality_audit.csv` (`issue_type=exact_duplicate_label_conflict`);
both the original and proposed label are preserved in the `evidence` column, per
the assignment's audit-table requirement — nothing was silently overwritten.

**Near-duplicate templated conflicts (all splits, reported only, never fixed):** 78
distinct normalized-text groups (URL/mention/case/punctuation stripped) show label
disagreement, spanning 287 rows across train/dev/heldout — these look like
templated or copy-pasted tweets (identical wording, different embedded URL) that
were labeled inconsistently, likely across different annotators or scraping
passes.

**Heldout hard negatives (reported only, never edited or removed):** 295 of 1523
heldout examples (19.4%) are misclassified by the reference-repro model.

## Dev evidence

| | Dev F1 |
|---|---|
| No fix (baseline) | 0.7514 |
| After majority-vote fix | 0.7545 |
| Delta | +0.0031 |

On dev alone, the fix looks like a clean win — this is exactly why dev evidence
by itself is not sufficient to trust a data-quality intervention.

## Final held-out evidence

| | Heldout F1 |
|---|---|
| No fix (baseline) | 0.7576 |
| After majority-vote fix | 0.7516 |
| Delta | **−0.0060** |

**The fix helps on dev but hurts on heldout.** This is the opposite of "safe,
strictly helpful" — the hypothesis was wrong, or at least incomplete.

## Is that heldout drop real, or noise?
Checked with McNemar's exact test on the paired heldout predictions:
```
no_fix-only-right: 10   label_fix-only-right: 1
both right: 1218   both wrong: 294
McNemar exact test p-value: 0.0117
```
**This is statistically significant** at the conventional α=0.05 threshold, and in
the direction that hurts the model: 10 examples the unfixed model got right become
wrong after the "fix," against only 1 the other way. Unlike the threshold and
normalization tickets, this is not a "noise, can't tell" result — the label fix
measurably and significantly *degraded* heldout performance, despite improving dev
F1 and despite the fix being a defensible-looking correction (majority vote on
literal exact-text duplicates, not a guess).

## Why might a "safe" fix backfire?
Only 12 of 4567 training rows changed — too few to meaningfully shift the decision
boundary in a way that should generalize, and reweighting 12 examples' worth of
gradient signal in a regularized linear model can easily just move the decision
boundary in an idiosyncratic direction that happens to help the specific 1523 dev
examples (which is what dev selection would reward) without helping — or while
hurting — a different 1523-example heldout sample. This is a small-sample
overfitting-to-dev story playing out at the level of a single data-cleaning
decision, not just at the level of model hyperparameters (compare to Ticket 4's
threshold selection, which generalized well specifically *because* the effect
being tuned was much larger and more stable than 12 relabeled rows out of 4567).

## Concrete examples
The highest-confidence heldout hard negatives include cases like id=5903 ("You can
never escape me... I know pain...", true label=1, predicted score=0.031) and
id=10795 ("Israel wrecked my home. Now it wants my land.", true label=1, predicted
score=0.083) — both plausibly ambiguous between literal disaster language and
figurative/political speech, which is itself useful evidence about where this
task's label boundary gets genuinely hard, independent of any data defect. On the
label-fix side: of the 12 relabeled train rows, the McNemar breakdown shows the
fix's cost outweighed its benefit on heldout (10 previously-correct predictions
broke vs. only 1 newly fixed) — see `results/data_quality_audit.csv` for the full
list of relabeled ids with original and proposed labels.

## Dispositions (per `results/data_quality_audit.csv`)
Using the required vocabulary (`fix`, `keep_but_flag`, `ambiguous`,
`reject_false_positive`):
- Exact-duplicate rows whose original label matches the majority vote →
  `keep_but_flag`
- Exact-duplicate rows whose original label disagrees with the majority vote →
  `fix` (proposed correction; **the controlled experiment above shows this
  "fix" should be applied with caution, not blindly** — the audit table's
  `disposition=fix` reflects the majority-vote logic, but the accompanying
  evidence column and this ticket both make clear the correction was tested and
  did not reliably help)
- Near-duplicate conflicts in dev/heldout → `ambiguous`
- Hard-negative heldout errors → `ambiguous` (heldout labels/examples are never
  edited or removed, per the assignment rules)

## Limitation
Only 12 rows were affected by the fix, which is both the finding's strength (a
small, precisely-targeted intervention) and its weakness (too small a sample to
draw a general conclusion about whether majority-vote correction of duplicate
labels is a good idea in general — it might generalize better on a dataset with
more duplicate conflicts, or with a less regularized model). This result should be
read as "this specific fix, on this specific 12-row set, measurably hurt" rather
than "label-correction is bad in general."

## AI usage note
Tool: Claude (via Claude.ai chat with code execution).
Prompt/ask: find exact/near-duplicate label conflicts and heldout hard negatives,
then actually test whether the proposed training-label correction helps rather
than assuming it does.
Output used: `experiments/ticket5_data_quality.py` (audit detection, artifact
export) and `experiments/ticket5_validation.py` (the controlled before/after
retraining experiment and McNemar significance test).
Verification: the "fix hurts heldout" claim was checked twice — once as a raw F1
delta, then confirmed with a paired significance test rather than trusting a
single heldout F1 number, exactly as required for a controlled experiment rather
than an anecdote.
