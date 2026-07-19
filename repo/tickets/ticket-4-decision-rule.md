# Ticket 4: Decision-Rule and Model Ticket

## Hypothesis
TODO: state your hypothesis before looking at results.

## Intended lever
Decision threshold, `class_weight`, and a second CPU classifier (linear SVM via SGD,
`modified_huber` loss), compared against the Ticket 1 reference-repro Logistic
Regression at default threshold 0.5.

## Controlled experiment: threshold sweep (dev)

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.6144 | 0.8611 | 0.7171 |
| 0.35 | 0.6717 | 0.8183 | 0.7378 |
| 0.40 | 0.7151 | 0.7740 | 0.7434 |
| **0.45** | **0.7756** | **0.7389** | **0.7568** |
| 0.50 (default) | 0.8121 | 0.6992 | 0.7514 |
| 0.55 | 0.8418 | 0.6580 | 0.7386 |
| 0.60 | 0.8744 | 0.5954 | 0.7084 |
| 0.65 | 0.9067 | 0.5344 | 0.6724 |
| 0.70 | 0.9345 | 0.4794 | 0.6337 |

This is the full precision-recall tradeoff curve, not just the best F1 point --
note precision climbs from 0.61 to 0.93 across the sweep while recall falls from
0.86 to 0.48. Whether 0.45 (best F1) is actually the "right" operating point depends
on whether missed disasters (false negatives) or false alarms (false positives) are
costlier for the downstream use case.

## Other candidates compared on dev

| Model | Dev F1 |
|---|---|
| logreg, thr=0.5 | 0.7514 |
| logreg, thr=0.45 | **0.7568** (winner) |
| logreg, class_weight=balanced, thr=0.5 | 0.7568 (tied) |
| linear SVM (SGD), thr=0.5 | 0.7306 |

Threshold tuning at 0.45 and `class_weight='balanced'` land on the same dev F1
(0.7568) via different mechanisms -- worth checking in your writeup whether they
also produce the *same* predictions or arrive at that score differently.

## Held-out evidence
Winner (`logreg_thr0.45`) heldout F1 = 0.7666 vs reference-repro (thr=0.5) heldout
F1 = 0.7576. Error decomposition: **30 false negatives fixed, 0 false positives
fixed; 34 new false positives introduced, 0 new false negatives** (`results/summary.csv`,
ticket=4 row).

## Concrete examples
TODO: pull specific ids from `predictions/heldout_predictions.csv` (ticket==4) where
`y_pred` flipped relative to the ticket==1 reference-repro predictions, and look at
2-3 of the newly-introduced false positives. Do they look like reasonable
"almost-disaster" language that a lower threshold would predictably catch, or
outright unrelated tweets?

## Precision-recall tradeoff discussion
TODO: this is the actual required analysis -- explain in your own words why the
mechanism here (30 FN fixed for 34 FP added) is roughly a 1-for-1 trade rather than
a clean win, and what that implies about whether "F1 went up" is a good enough
justification on its own for adopting this threshold in a real deployment.

## Limitation
TODO: e.g. threshold was frozen using a single dev-split F1 sweep; it was not
validated for stability with a different random seed or resampled dev set, so it's
unclear if 0.45 is a genuinely better operating point or is fit to this particular
dev split's idiosyncrasies.

## AI usage note
TODO.
