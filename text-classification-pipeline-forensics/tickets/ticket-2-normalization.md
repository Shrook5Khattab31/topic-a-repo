# Ticket 2: Text Normalization Lever

## Hypothesis
Keeping literal hashtag tokens (e.g. `#earthquake` as its own distinct feature)
would help the model, because a word used as a hashtag might carry different
signal than the same word used in running text (e.g. `#earthquake` as a topic tag
vs. "earthquake" mentioned in passing) - so unwrapping hashtags to bare words could
be throwing away a useful distinction.

## Intended lever
Normalization config, chosen from `pipeline/preprocess.py: CONFIGS` (`raw`,
`baseline`, `aggressive`, `keep_hashtags_raw`, `keep_case`) - the only axis varied
here is whether `#word` is unwrapped to `word` or left as literal `#word` before
casing/URL/mention handling (which is held constant across all 5 configs).

## Controlled experiment
All 5 configs trained on `train_ids` with the fixed reference-repro model settings
from Ticket 1 (unigrams, no sublinear TF, C=2.0), differing only in the
normalization config applied to the tweet text before vectorization.

## Dev evidence
Evaluated on `dev_ids`:

| Config | Dev F1 | Dev P | Dev R |
|---|---|---|---|
| raw | 0.7504 | 0.7939 | 0.7115 |
| baseline (hashtags unwrapped) | 0.7514 | 0.8121 | 0.6992 |
| aggressive | 0.7514 | 0.8121 | 0.6992 |
| **keep_hashtags_raw** | **0.7518** | 0.8110 | 0.7008 |
| keep_case | 0.7514 | 0.8121 | 0.6992 |

`keep_hashtags_raw` wins on dev by 0.0004 F1 and was frozen before touching
heldout, per the assignment's split-usage rule.

## Final held-out evidence
`keep_hashtags_raw` heldout F1 = 0.7595 vs `baseline` heldout F1 = 0.7576.
Only **3 of 1523** heldout predictions differ between the two configs - all 3 were
false positives under `baseline` that become correct (true negative) under
`keep_hashtags_raw`; zero new errors introduced.

**Significance check (McNemar's exact test, paired on the same heldout ids):**
```
baseline-only-right: 0   keep_hashtags_raw-only-right: 3
McNemar exact test p-value: 0.2500
```
With only 3 discrepant pairs, this is not statistically significant - expected,
given how few predictions moved at all. The headline "F1 improved" is true and the
direction is consistently positive (all 3 flips go the same way), but the sample is
far too small to claim a robust generalizable effect from this evidence alone.

## Concrete examples and the actual mechanism (does the movement match the hypothesis?)
The three flipped heldout ids, with the text each config actually sees after
normalization:

**id=767** (true label 0): *"the fall of leaves from a poplar is as fully ordained
as the tumbling of an avalanche - Spurgeon"* - contains **no hashtag at all**.
Normalized text is byte-identical under both configs.

**id=2329** (true label 0): *"Timestack' Photos Collapse Entire Sunsets Into Single
Mesmerizing Images. http://t.co/Cas8xC2DFE"* - also contains **no hashtag**.
Normalized text is byte-identical under both configs.

**id=7908** (true label 0): *"#CCOT #TCOT #radiation Nuclear Emergency Tracking
Center..."* - this one does contain hashtags, and normalizes differently
(`ccot tcot radiation ...` vs `#ccot #tcot #radiation ...`). But feeding both
versions through scikit-learn's actual tokenizer shows they produce **identical
tokens** (`['ccot', 'tcot', 'radiation', ...]`) - sklearn's default token pattern
already strips `#` during tokenization regardless of our preprocessing choice, so
even this example's *own* text isn't the direct cause either.

**This means the hypothesis ("literal hashtag tokens carry distinct signal") is
not actually what's driving the result.** None of the 3 flipped predictions are
explained by the model seeing `#word` as a feature distinct from `word` - because
after tokenization, it never does.

**The real mechanism, found by diffing the fitted vocabularies directly:** the two
configs produce vocabularies of the *same size* (11,156 words) but *different
composition* - 6 words differ each way. Tracing them back to source tweets shows
they come from hashtags glued directly onto a preceding word or symbol with no
space, e.g.:
- `"...tcotåÊ#ccot..."` → unwrapping deletes only the `#`, producing the merged
  token `tcotåêccot`; keeping the literal `#` lets sklearn's tokenizer split on it
  naturally into `tcotåê` and `ccot`.
- `"Rare#Deals_UK"` → merges to `raredeals_uk` under unwrapping vs. splits to
  `rare` + `deals_uk` under `keep_hashtags_raw`.
- `"kwaAaaA#dead"` → merges to `kwaaaaadead` vs. splits to `kwaaaaa` + `dead`.

So `keep_hashtags_raw`'s real advantage in this run is that it **avoids a
whitespace-insertion bug in the unwrap step**, not that it preserves hashtag
semantics. `HASHTAG_RE.sub(r"\1", t)` deletes the `#` character without inserting a
space, so a `word#hashtag` sequence with no space collapses into one nonsense
token under `baseline`, fragmenting/destroying two otherwise-valid words. Those
malformed merged tokens are rare (6 out of 11,156 vocabulary entries) and mostly
come from encoding-corrupted ("mojibake", e.g. `Û\x8f`) or copy-pasted tweets - but
regularized Logistic Regression fits over the whole vocabulary, so removing a
handful of junk tokens slightly shifts coefficients everywhere, which is
apparently enough to flip 3 borderline heldout predictions, including two (id=767,
id=2329) whose own text never touches a hashtag at all.

## Revised conclusion
The measured F1 improvement is real but tiny and not statistically significant
(McNemar p=0.25, n=3). More importantly, **the mechanism does not match the
original hypothesis** - this is not evidence that literal-vs-unwrapped hashtags
carry different task signal; it's evidence of a minor preprocessing bug (missing
whitespace insertion on hashtag unwrap) that happens to be dodged by the
`keep_hashtags_raw` config. A cleaner fix would be to change the unwrap regex to
insert a space (`HASHTAG_RE.sub(r" \1", t)`) rather than to keep hashtags literal -
we'd predict that fix converges to the same result as `keep_hashtags_raw` without
requiring the literal `#` to remain in the vocabulary. This is left as a follow-up
probe rather than resolved here, since the ticket's dev-freeze rule means we
shouldn't retroactively test a 6th config against heldout after seeing this result.

## Limitation
Only 3 heldout predictions changed at all between the two best configs - far too
small a sample to distinguish a genuine normalization effect from noise (confirmed
by the non-significant McNemar test above). The interesting finding here is
methodological rather than about hashtags per se: a small, well-intentioned
preprocessing decision can produce a measurable score change through a completely
different mechanism than the one hypothesized, which is exactly the kind of
"looks understood but isn't" trap the assignment brief warns about. Before trusting
any normalization lever's dev-set F1 gain, it's worth tracing *which* tokens the
change actually touches rather than assuming the surface-level rationale is what's
happening.


## Ticket 2 completion checklist
- Hypothesis stated: yes.
- Intended normalization lever isolated: yes.
- Dev-only model selection: yes, five configs compared before heldout.
- Frozen heldout evaluation: yes, `keep_hashtags_raw` evaluated after dev choice.
- FP/FN movement shown: yes, 3 false positives fixed and no new errors.
- Concrete examples supplied: yes, ids 767, 2329, and 7908.
- Mechanism checked rather than assumed: yes, normalized text, tokenizer output,
  fitted vocabulary diff, and source-tweet trace.
- Limitation supplied: yes, only 3 changed heldout predictions and McNemar p=0.25.
- Final decision: keep the finding as diagnostic evidence of a whitespace bug, not
  as proof that literal hashtag semantics generalize.

## AI usage note
Tool: Claude
AI was used only to inspect tokenization differences and confirm vocabulary diffs. 
All normalization configs, dev/held‑out evaluations, and significance tests were run manually through the project scripts.
