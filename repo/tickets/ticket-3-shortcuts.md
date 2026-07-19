# Ticket 3: Feature and Shortcut Audit

## Hypothesis
TODO: state your hypothesis before looking at results (e.g. "keyword alone should
carry some signal since it's curated from the tweet, but far less than full text").

## Intended lever
Compare three models trained on isolated feature sets: `keyword` field only,
character/word length only, and full TF-IDF text (the Ticket 1 reference-repro
model), all trained on `train_ids`, evaluated on `dev_ids` then `heldout_ids`.

## Controlled experiment / Dev evidence

| Feature set | Dev F1 | Dev P | Dev R |
|---|---|---|---|
| keyword_only | 0.6599 | 0.6796 | 0.6412 |
| length_only | 0.4358 | 0.5344 | 0.3679 |
| text_only (full TF-IDF) | 0.7514 | 0.8121 | 0.6992 |

## Held-out evidence

| Feature set | Heldout F1 |
|---|---|
| keyword_only | 0.6905 |
| length_only | 0.4835 |
| text_only | 0.7576 |

`keyword` alone reaches F1=0.69 -- roughly 91% of the full model's F1 -- using a
single categorical field with no text modeling at all. Length alone is much weaker
(F1=0.48, barely above chance-adjacent) but not useless.

221 distinct keywords exist in train; 21 of them (9.5%) have >=5 examples and are
>=95% single-class (e.g. `derailment` is 100% positive across 27 rows, `aftershock`
is 0% positive across 23 rows -- see script output for the full purity table).

## Concrete examples
TODO: pick 2-3 of the near-pure keywords (e.g. `derailment`, `aftershock`,
`body%20bags`) and look at a handful of their actual tweet texts. Do they look like
genuinely disaster-associated words (legitimate signal) or like an artifact of how
the Kaggle dataset was scraped/curated (e.g. templated tweets, retweet chains)?

## Is this signal legitimate, an artifact, or mixed?
TODO: this is the actual required judgment call for this ticket -- don't just report
the numbers. Consider: would `keyword` still predict well on tweets collected from a
different time period or platform, or is 0.69 F1 partly because the Kaggle keyword
field was assigned in a way that leaks target-correlated information (e.g. searched-for
keyword lists that were disaster/non-disaster curated in the first place)?

## Limitation
TODO: e.g. keyword_only and text_only were not combined into a joint feature model,
so we can't yet say how much of keyword's signal is redundant with what TF-IDF
already captures from the tweet body vs. additive.

## AI usage note
TODO.
