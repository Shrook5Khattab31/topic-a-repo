# Ticket 4: Decision-Rule and Model Ticket

## Hypothesis
The default 0.5 decision threshold is an arbitrary artifact of treating this as a
plain binary classification problem, not a value tuned to this task's actual
error costs - so sweeping the threshold should reveal a better operating point,
and a second linear CPU classifier (SVM) might offer a different precision/recall
shape worth comparing against Logistic Regression.

## Intended lever
Decision threshold, `class_weight`, and a second CPU classifier (linear SVM via
SGD, `modified_huber` loss), all compared against the Ticket 1 reference-repro
Logistic Regression at default threshold 0.5.

## Controlled experiment
Threshold sweep, `class_weight` comparison, and a second CPU classifier (linear
SVM via SGD), all trained on `train_ids` with the fixed reference-repro
vectorizer/regularization settings from Ticket 1, compared on `dev_ids` before any
heldout evaluation.

## Dev evidence

**Threshold sweep:**

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

This is the full precision-recall tradeoff curve, not just the best F1 point.
Precision climbs from 0.61 to 0.93 across the sweep while recall falls from 0.86
to 0.48 - a roughly linear-looking tradeoff over this range, with F1's peak at 0.45
sitting close to where precision and recall cross (0.78/0.74).

**Other candidates compared on dev:**

| Model | Dev P | Dev R | Dev F1 | Decision note |
|---|---:|---:|---:|---|
| logreg, thr=0.5 | 0.8121 | 0.6992 | 0.7514 | reference operating point |
| logreg, thr=0.45 | 0.7756 | 0.7389 | **0.7568** | winner/tie-breaker |
| logreg, class_weight=balanced, thr=0.5 | 0.7756 | 0.7389 | 0.7568 | same dev P/R/F1 as threshold move |
| linear SVM (SGD), thr=0.5 | 0.7595 | 0.7038 | 0.7306 | second CPU classifier, rejected |

Threshold tuning at 0.45 and `class_weight='balanced'` land on the identical dev
precision/recall/F1 point (0.7756/0.7389/0.7568). The frozen choice is the threshold
move because it is the narrower decision-rule intervention: it reuses the already
validated reference-repro probabilities and changes only the operating point,
whereas class weighting retrains the classifier and changes the fitted coefficients.
The linear SVM underperforms both. This satisfies the second-classifier requirement
while keeping the final decision tied to the controlled lever that is easiest to
audit.

## Final held-out evidence
Winner (`logreg_thr0.45`) heldout F1 = **0.7666** vs reference-repro (thr=0.5)
heldout F1 = 0.7576. Error decomposition: **30 false negatives fixed, 0 false
positives fixed; 34 new false positives introduced, 0 new false negatives**
(`results/summary.csv`, ticket=4 row).

## Is the threshold change actually significant, or generalizing well?
Two separate questions, checked separately:

**Significance (McNemar's exact test, thr=0.5 vs thr=0.45 on heldout):**
```
thr0.5-only-right: 34   thr0.45-only-right: 30
McNemar exact test p-value: 0.7080
```
Not significant - same pattern as Tickets 1–3: at n=1523 heldout, a swing of a few
dozen flipped predictions in either direction is well within noise.

**Generalization (oracle threshold check, diagnostic only, never used to make the
actual decision):** sweeping the threshold directly *on heldout* to find its true
optimum gives **0.44 (F1=0.7688)** - almost exactly the dev-chosen 0.45
(F1=0.7666), a gap of only +0.0022. This is a genuinely reassuring result: the
dev-based threshold selection wasn't a fluke that happened to overfit dev noise -
it landed within a hair of the heldout-optimal value despite never seeing heldout
during selection. **These two findings aren't in conflict**: the threshold effect
is small and not statistically distinguishable from a 0.5 baseline in absolute
terms, but *within that small effect*, dev-based selection reliably finds close to
the best available operating point rather than a random one.

## Concrete examples: the precision-recall tradeoff, not just F1
Lowering the threshold to 0.45 makes the model fire "disaster" more readily. The
30 fixed false negatives are heldout tweets the model was previously too
conservative on - genuine disaster tweets it now catches. The 34 new false
positives are the cost: tweets the stricter 0.5 threshold correctly rejected that
the looser 0.45 threshold now wrongly flags.

Representative fixed false negatives:

| id | true | score | why the lower threshold helps |
|---|---:|---:|---|
| 509 | 1 | 0.493 | Salvation Army disaster-relief context just below 0.5 |
| 1973 | 1 | 0.467 | bush-fire mention was too weak for the default threshold |
| 1985 | 1 | 0.491 | public-health trauma after bush fires sits near the boundary |

Representative new false positives:

| id | true | score | why the lower threshold hurts |
|---|---:|---:|---|
| 67 | 0 | 0.487 | figurative "streets ablaze" becomes a false alarm |
| 936 | 0 | 0.465 | real-estate listing with "Cannon Beach" is pushed positive |
| 1488 | 0 | 0.483 | band/show name "The Body Bags" triggers disaster-like terms |

The trade is close to 1-for-1 (30 fixed vs. 34 new), which is exactly why F1 barely
moves (0.7576 → 0.7666) despite 64 total predictions changing. **F1 near a
tradeoff's crossover point is not very informative about whether the tradeoff is
"worth it"**; that depends on whether missing a real disaster tweet (false
negative) or raising a false alarm (false positive) is more costly for whatever
this classifier would actually be used for downstream, which the assignment's
evaluation setup does not specify and which no F1-based analysis alone can answer.

## Limitation
The linear SVM (SGD) comparison used default `max_iter=2000` with no separate
learning-rate or epoch tuning, while the Logistic Regression baseline uses
`lbfgs`, a solver much better suited to small, well-conditioned problems like
this one out of the box. This is not a fair apples-to-apples comparison of the
*model families* - it mainly shows that our default SGD-SVM setup underperforms
our tuned LogReg setup, not that linear SVMs are inherently worse for this task.
A fairer comparison would sweep SGD's `alpha`/learning-rate schedule the same way
we swept the LogReg threshold.


## Ticket 4 completion checklist
- Hypothesis stated: yes.
- Intended levers covered: threshold, class weighting, and second CPU classifier.
- Precision-recall tradeoff shown: yes, full threshold table 0.30-0.70.
- Dev-only selection respected: yes.
- Heldout evidence supplied after freezing: yes.
- FP/FN movement supplied: yes, 30 FNs fixed and 34 FPs introduced.
- Concrete examples supplied: yes, fixed FNs and new FPs with ids and scores.
- Significance and generalization separated: yes, McNemar p=0.7080 plus oracle
  threshold diagnostic.
- Limitation supplied: yes, SVM family was not exhaustively tuned.

## AI usage note
Tool: Claude
AI assistance was used only to re-run threshold sweeps and significance tests for confirmation.
 All model training, comparison logic, and held‑out evaluation were done manually.
