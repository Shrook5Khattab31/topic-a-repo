# Ticket 3: Feature and Shortcut Audit

## Hypothesis
`keyword` is a curated single field taken from the tweet itself, so it should carry
real but limited signal on its own - clearly less than the full tweet text, but
non-trivially above chance - and any strong keyword→target correlation is worth
checking for whether it reflects genuine disaster-related meaning or an artifact of
how this specific dataset happens to be composed.

## Intended lever
Feature set: `keyword` alone, tweet length alone, or the full tweet text -
isolating each as its own predictor to measure how much signal comes from
shallow/metadata fields versus the actual text.

## Controlled experiment
Three isolated feature sets, each trained as its own Logistic Regression model on
`train_ids`: `keyword` alone (one-hot), character/word length alone, and the full
TF-IDF text model (Ticket 1's reference-repro config). No preprocessing or
model-selection choices are tuned here - the point is measurement, not
optimization.

## Dev evidence

| Feature set | Dev F1 | Dev P | Dev R |
|---|---|---|---|
| keyword_only | 0.6599 | 0.6796 | 0.6412 |
| length_only | 0.4358 | 0.5344 | 0.3679 |
| text_only (full TF-IDF, reference config) | 0.7514 | 0.8121 | 0.6992 |

## Final held-out evidence

| Feature set | Heldout F1 |
|---|---|
| keyword_only | 0.6905 |
| length_only | 0.4835 |
| text_only | 0.7576 |

`keyword` alone reaches **91% of the full text model's F1** using nothing but a
single categorical field and no text modeling at all. `length` alone is much
weaker (F1=0.48) but still clearly above a majority-class floor (majority-class F1
would be 0, since the floor model never predicts the minority class - see
`experiments/ticket1_baseline.py`'s floor-model utility).

## Is keyword redundant with text, or does it add real information?
Measured directly rather than assumed, on heldout:

- **Redundancy**: of the 1125 heldout examples `keyword_only` gets right, **1010
  (89.8%) are also correctly classified by `text_only`.** Keyword is mostly telling
  the model something it would have figured out from the tweet body anyway.
- **Incremental value**: a combined text+keyword model (TF-IDF features
  concatenated with one-hot keyword, same regularization) reaches heldout F1 =
  **0.7653**, up from text_only's 0.7576.
- **But is that gain real?** McNemar's exact test on text_only vs. text+keyword
  gives an exactly symmetric result: **58 examples text_only gets right that the
  combined model misses, and 58 examples the reverse - p=1.0.** The F1 number goes
  up, but the two models are not distinguishable in how many predictions they get
  right; the gain comes from a precision/recall rebalancing (which examples are
  right), not from strictly more correct predictions. This is a caution against
  reading "F1 improved by 0.008" as unambiguous evidence of a better model without
  checking what actually changed.

## Concrete examples and legitimacy judgment (the actual required analysis)
221 distinct keywords exist in `train_ids`; **21 (9.5%) are "near-pure"** (≥5
training examples, ≥95% single-class). Inspecting the actual tweets behind the top
five by count shows a real split between legitimate signal and dataset artifact:

**Genuinely disaster-associated (legitimate task information):**
- `derailment` (100% positive, n=27) - every sampled tweet is real train-derailment
  news coverage ("MP trains derailment: 'It's the freakiest of freak accidents'...").
  This word is disaster-specific in ordinary English; its purity looks like real
  signal that should generalize.

**Purity driven by non-disaster idiom/slang usage (likely artifact for this word):**
- `aftershock` (0% positive, n=23) - despite being a genuine disaster term, every
  sampled tweet uses it as a proper noun: a Twitter handle (`@afterShock_DeLo`), a
  dubstep/EDM track title (`320 [IR] ICEMOON [AFTERSHOCK]`). The word's dictionary
  meaning is disaster-related, but its *usage distribution in this dataset* is
  dominated by an unrelated pop-culture sense. A model trained on this dataset
  would learn "aftershock → not a disaster," which is backwards from the word's
  literal meaning and would likely mislead on a differently-sampled tweet stream.
- `wrecked` (3.8% positive, n=26) - all sampled uses are casual idiom ("wrecked an
  hour on YouTube", "sleeping schedule is wrecked") rather than literal disaster
  damage. Same failure mode as `aftershock`.
- `ruin` (3.6% positive, n=28) - same pattern: idiomatic/emotional usage
  ("don't let that ruin your year") rather than literal disaster ruin.
- `body%20bags` (0% positive, n=23) - the literal `%20` (URL-encoded space) still
  present in the field value is itself a strong sign of a scraping/query artifact
  rather than a clean content label. Sampled tweets are about literal shoulder
  bags/handbags, not disaster body bags - likely a keyword-search false-positive
  from a term-matching collection process rather than content the annotators
  intended to associate with this term.

## Discussion: legitimate, artifact, or mixed?
**Mixed, and the split is systematic rather than random.** The near-pure keywords
that stayed disaster-associated in this data (`derailment`, `debris`, `rescuers`,
`typhoon`, `outbreak`) are words with little competing non-disaster usage in
everyday English/Twitter slang. The near-pure keywords that flipped to
non-disaster purity (`aftershock`, `wrecked`, `ruin`, `hellfire`) are words that
have a common idiomatic or pop-culture meaning that dominates their actual usage on
Twitter, even though the word itself is disaster-flavored. This is consistent with
`keyword` having originally been used as a **search/collection term** for building
this dataset (the field values read like a fixed vocabulary list, and `body%20bags`
retaining literal URL encoding supports this) rather than being independently
assigned by annotators reading each tweet - a tweet matching the search term
`aftershock` was pulled into the dataset regardless of whether the tweet actually
used the word in a disaster sense, and the label was then assigned by an annotator
per its own content. That means `keyword`'s predictive power partly reflects **this
dataset's specific sampling process**, not a stable property of the words
themselves - the hidden stress-test grading mentioned in the handout (which
perturbs metadata-like shortcuts) is exactly the kind of check that would likely
expose this: shuffling or masking `keyword` should hurt `keyword_only` far more
than it hurts `text_only`, because `text_only` is grounded in the actual content of
the tweet rather than in which search term the dataset's original authors happened
to have used to find it.

## Limitation
This analysis inspected only the top 5 of 21 near-pure keywords by manual reading,
which is a small, judgment-based sample - a systematic classification of all 21 (or
all 222) keywords into "generalizes" vs. "artifact of collection" was not
attempted, and would need either external validation data or a clearer theory of
what counts as idiomatic-vs-literal usage to do rigorously rather than by eyeballing
a handful of tweets per keyword.

## AI usage note
Tool: Claude
AI helped with reading tweet samples for keyword purity and re-running the redundancy checks. 
The feature-isolation models and combined-model experiments were implemented and validated manually.
