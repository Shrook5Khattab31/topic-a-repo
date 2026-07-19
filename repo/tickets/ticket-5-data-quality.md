# Ticket 5: Data-Quality and Error Ticket

## Hypothesis
TODO: state your hypothesis before looking at results.

## Method
Three checks, run by `experiments/ticket5_data_quality.py`:
1. Exact-text duplicates in `train_ids` with conflicting labels.
2. Near-duplicate (normalized) text across all splits with label disagreement
   (reported only for dev/heldout -- never fixed or removed there).
3. Highest-confidence wrong predictions on `heldout_ids` from the Ticket 1
   reference-repro model ("hard negatives"), reported only.

## Findings

**Exact-duplicate label conflicts (train):** 11 distinct texts appear more than
once in `train_ids` with disagreeing labels (e.g. one copy labeled 1, another
labeled 0 for the identical string). Full list with proposed majority-vote labels
is in `results/data_quality_audit.csv` (`issue_type=exact_duplicate_label_conflict`).
Both the original and proposed label are preserved in the `evidence` column, per
the assignment's audit-table requirement -- nothing was silently overwritten.

**Near-duplicate templated conflicts (all splits, reported only):** 78 distinct
normalized-text groups (URL/mention/case/punctuation stripped) show label
disagreement, spanning 287 rows total across train/dev/heldout. These look like
templated or copy-pasted tweets (same wording, different embedded URL) that
different annotators or scraping passes labeled inconsistently.

**Heldout hard negatives (reported only, not edited):** 295 of 1523 heldout
examples (19.4%) are misclassified by the reference-repro model. The 8
highest-confidence wrong predictions (model score far from 0.5, yet still wrong)
include cases like id=5903 ("You can never escape me... I know pain...", true
label=1, predicted score=0.031) and id=10795 ("Israel wrecked my home. Now it
wants my land.", true label=1, predicted score=0.083) -- both plausibly ambiguous
between literal disaster language and figurative/political speech.

## Dispositions
Per `results/data_quality_audit.csv`, using the required disposition vocabulary
(`fix`, `keep_but_flag`, `ambiguous`, `reject_false_positive`):
- Exact-duplicate rows whose label matches the majority vote -> `keep_but_flag`
- Exact-duplicate rows whose label disagrees with the majority vote -> `fix`
  (train-only; proposed correction, original label preserved in evidence)
- Near-duplicate conflicts in dev/heldout -> `ambiguous`
- Hard-negative heldout errors -> `ambiguous` (heldout labels/examples are never
  edited or removed, per the assignment rules)

## Concrete examples
TODO: for at least 2 of the exact-duplicate conflicts and 2 of the hard-negative
examples, quote the text and explain your own judgment call: mislabel, genuinely
ambiguous, or a case the model should just be expected to get wrong (hard negative
that isn't a data problem at all)?

## Limitation
TODO: e.g. "confidence" in the hard-negative table is a proxy (distance of the
predicted score from 0.5) rather than a calibrated probability, so it should not be
read as "the model was N% sure" -- only as a ranking of which errors were most
surprising to the model.

## AI usage note
TODO.
