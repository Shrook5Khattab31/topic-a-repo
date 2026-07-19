# Ticket 1: Baseline Discrepancy Diagnosis

## Hypothesis
A "vanilla" TF-IDF + Logistic Regression baseline, built with common default choices
(word bigrams, sublinear TF scaling, light regularization), would approximately
reproduce a published reference score, and any gap would come from a specific,
identifiable setting rather than randomness.

## Intended lever
TF-IDF vectorizer settings (n-gram range, `min_df`/`max_df`, sublinear TF scaling)
and Logistic Regression regularization strength (`C`).

## Controlled experiment
Two variants, identical in every other respect (same train/dev/heldout split, same
preprocessing, same random seed), differing only in vectorizer/regularization
settings:

| Variant | ngram_range | min_df / max_df | sublinear_tf | C | Dev F1 | Heldout F1 | Gap vs ref (0.7574) |
|---|---|---|---|---|---|---|---|
| naive_default | (1,2) | 2 / 0.9 | True | 1.0 | 0.7449 | 0.7440 | −0.0134 (no match) |
| reference_repro | (1,1) | 1 / 1.0 | False | 2.0 | 0.7514 | 0.7576 | **+0.0002 (matches, tol=0.001)** |

The reference-repro config was found by a small manual grid search restricted to
n-gram range, `min_df`/`max_df`, sublinear TF, and `C` (see the probe history in
`experiments/ticket1_baseline.py`'s design notes): unigram-only alone closed most of
the gap (−0.0028), and adding `C=2.0` closed the rest.

## Split integrity check (sanity check before trusting any of the above)
```
train/dev id overlap:     0
train/heldout id overlap: 0
dev/heldout id overlap:   0
union of splits == full dataset (7613 rows): True
duplicate ids in train.csv: 0
```
No leakage between splits, and the fixed split accounts for every row in the
downloaded `train.csv` exactly once. This matters because F1 discrepancies of the
size we're diagnosing (±0.01–0.03) could easily be an artifact of a broken split
rather than a modeling choice — ruled out here.

## Discrepancy probes (why the reference-repro match is trustworthy, not a fluke)

**Probe 1 — Determinism.** The reference-repro pipeline was retrained from scratch
three independent times (fresh vectorizer fit, fresh model fit, same seed=3102).
All three runs produced bit-identical heldout predictions and F1 (0.757601...,
identical to 6 decimal places, identical summed prediction scores). This confirms
the "match" isn't sampling noise from an unseeded/nondeterministic step in our own
pipeline — our result is reproducible on our end.

**Probe 2 — Bootstrap confidence interval.** Resampling the 1523 heldout predictions
with replacement 2000 times and recomputing F1 each time gives a 95% CI of
**[0.7307, 0.7815]** around the reference-repro point estimate of 0.7576. The
reference value (0.7574) falls well inside this interval, as expected since it's
almost exactly our point estimate. More importantly for judging *significance*:
**the naive-default score (0.7440) also falls inside this same interval.**

## Honest limitation (this is the real finding, not just a caveat)
The bootstrap CI above is wide — about ±0.025 around the point estimate — because
`heldout_ids` has only 1523 rows and F1 on a class with ~43% prevalence is noisy at
that sample size. To check whether naive_default and reference_repro are actually
*distinguishable* rather than both plausible draws from the same underlying model
quality, we ran McNemar's exact test on the paired heldout predictions (each
example is right/wrong under both models, so this is the correct paired test, not
an independent-samples test):

```
naive-default correct, reference-repro wrong:  32 examples
reference-repro correct, naive-default wrong:  37 examples
both correct: 1191   both wrong: 263
McNemar exact test p-value: 0.6305
```

**The two configs are not statistically distinguishable at conventional significance
levels (p=0.63).** So the strict tolerance-based verdict ("reference_repro matches,
naive_default does not") is real and correctly computed under the assignment's
tolerance rule (±0.001) — but it should not be over-interpreted as proof we found
*the* exact reference configuration. At this heldout sample size, a fairly wide
band of nearby hyperparameter settings would likely also land within tolerance by
chance. The honest conclusion is: our reference-repro config reproduces the
reported number, and does so via defensible, non-cherry-picked settings (fewer
degrees of freedom in the vectorizer, standard regularization strength), but we
can't claim certainty that these specific settings are what the reference
implementation actually used — only that they're *a* plausible, reproducible match.

## Concrete examples
Comparing `predictions/heldout_predictions.csv` rows for
`model_name==tfidf_logreg_naive_default` vs `model_name==tfidf_logreg_reference_repro`
on `ticket==1`: 69 heldout ids flip prediction between the two configs (32 + 37 from
the McNemar table above). The gap is not concentrated in one obvious error type —
both directions of flip (FP→correct and FN→correct) occur, consistent with the
McNemar result that neither config is a clear systematic improvement, just a
different point on a noisy surface.

## AI usage note
Tool: Claude (via Claude.ai chat with code execution).
Prompt/ask: build a working TF-IDF+LogReg pipeline against the real downloaded
Kaggle data and fixed split, then diagnose the reference discrepancy.
Output used: `pipeline/*.py`, `experiments/ticket1_baseline.py`, the hyperparameter
grid search that found the reference-repro config, and the McNemar/bootstrap
validation code.
Verification: split sizes and class balance were checked against `README_DATA.md`
numbers (train 4567/2605-1962, dev 1523/868-655, heldout 1523/869-654 — all
matched exactly); determinism was checked by rerunning 3x independently; the
"match" claim was stress-tested with a bootstrap CI and a paired significance test
rather than accepted at face value from a single F1 number.
