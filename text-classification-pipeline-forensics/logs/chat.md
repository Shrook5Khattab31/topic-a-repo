## Entries

### 2026-07-10 - Fatma - ticket 1 (baseline discrepancy)
Tool: Claude
Prompt/ask: had ticket1_baseline.py running already (naive_default vs
reference_repro variants), pasted the two printed blocks and asked claude to double
check my gap arithmetic against configs/project_contract.json's tolerance, since i
wanted a second pass before writing "does not match" / "matches" in the ticket.
Output used: none of the code changed, just used the check itself.
Verification: naive_default gap -0.0134 vs tol 0.001 -> confirmed does not match.
reference_repro gap +0.0002 vs tol 0.001 -> confirmed matches. re-did the
subtraction myself by hand too (0.7440-0.7574 and 0.7576-0.7574) before trusting it.

### 2026-07-10 - Mohamed - ticket 3 (shortcuts, first look)
Tool: Claude
Prompt/ask: already had keyword_only/length_only/text_only models written and
running, asked claude to sanity check whether f1=0.4358 for length_only is
plausible or if that number looks like a bug (seemed low)
Output used: no code kept, just used the sanity check to decide to keep going with
the same 3-model setup
Verification: reran ticket3_shortcuts.py myself after the check, got the same
0.4358 length_only f1 twice in a row, so it wasnt a fluke, just a genuinely weak
feature.

### 2026-07-11 - Fatma - ticket 1 (validation probes)
Tool: Claude
Prompt/ask: had split_integrity_check() and the 3-run determinism loop already
written and run, pasted the output (0 overlaps everywhere, 0 duplicate ids, 3
identical determinism runs at 0.757601) and asked claude to confirm i was reading
"union of splits == full dataset: True" correctly as meaning no rows are missing
from any split
Output used: just a confirmation, no code or wording taken from claude
Verification: cross-checked union count against len(df)=7613 myself in a python
shell, matched, so the True flag was legit not just printed blindly

### 2026-07-11 - Fatma - ticket 1 (mcnemar sanity check)
Tool: Claude
Prompt/ask: ran mcnemar_test(naive_default, reference_repro) myself and got
32/37/1191/263, p=0.6305 - asked claude to just double check the binomtest call was
using the right n_disc (32+37=69) and not accidentally using the full heldout size
Output used: none, this was purely a "did i call this right" check
Verification: manually recomputed min(32,37)=32 and n=69 and ran binomtest myself
in a separate shell to get the same p=0.6305 independently of the script

### 2026-07-11 - Mohamed - ticket 3 (purity table check)
Tool: Claude
Prompt/ask: had the keyword purity groupby already written, got 21/222 near-pure
keywords, pasted the top 10 table and asked claude to help me read a couple of the
printed example tweets faster since some had weird encoding characters mixed in
Output used: none of the interpretation was taken as-is, i wrote the
legitimate/artifact judgments myself after reading the raw tweets in the table
Verification: went back and reread all 10 sampled keyword's 3 example tweets myself
from the actual script printout, not from claude's paraphrase, before writing the
judgment column in the ticket

### 2026-07-12 - Shrook - ticket 5 (data quality, first pass)
Tool: Claude
Prompt/ask: had find_duplicate_conflicts() and the near-dup normalizer already
written and run (11 exact conflicts, 78 near-dup groups/287 rows), asked claude to
help me inspect a handful of the duplicate group texts since the terminal output
truncates long tweets and i wanted the full text for a few of them
Output used: none, just used it to read full text, no code changes
Verification: opened train.csv directly myself and looked up the full text for
those same ids to confirm the truncated preview matched the full tweet, wasnt
missing context that changes the label-conflict story

### 2026-07-12 - Mohamed - ticket 3 (redundancy check) -- second-opinion pass
Tool: Codex
Prompt/ask: had the keyword_only vs text_only overlap code and the combined hstack
model already written and run (overlap 1010/1125 = 89.8%, combined f1 0.7653,
mcnemar p=1.0000). Since claude had already looked at ticket 1's stats, asked codex
as a second, independent check to confirm the mcnemar count (58/58/1170/237) really
gives p=1.0 and isn't a rounding quirk in scipy's binomtest.
Output used: none, purely a second confirmation of an already-computed number, no
code taken.
Verification: recomputed the exact symmetric binomial test myself by hand
(min(58,58)=58, n=116) separately from both AI checks, got p=1.0 too, so it's
genuinely a tie, not a bug in either the script or either tool's answer.

### 2026-07-13 - Fatma - ticket 2 (normalization sweep)
Tool: Claude
Prompt/ask: had the 5-config dev sweep and heldout freeze already coded and run
(keep_hashtags_raw winning dev by 0.0004, 3 fixed / 0 new on heldout), asked claude
to help me read the compare_error_sets() output faster since i wanted to confirm
all 3 fixed ones were fp->tn and not some mix
Output used: none kept directly, just used it to speed up reading my own printout
Verification: looked at the 3 printed example rows myself (ids 767, 2329, 7908),
confirmed old_pred=1/new_pred=0 with true=0 for all three, so yes all 3 are fp
fixed, matches what i wrote in the ticket table

### 2026-07-14 - Fatma - ticket 2 (mechanism trace)
Tool: Claude
Prompt/ask: already had the tokenizer analyzer probe and vocab diff script written
and run myself (tokens identical, vocab diff = 6 words each side, traced back to
source tweets like "Rare#Deals_UK" -> raredeals_uk vs rare+deals_uk), asked claude
to confirm i was reading the vocabulary diff direction right (which side is
"words only in baseline" vs "only in keep_hashtags_raw") -- this is the "why did a
threshold/normalization change do this" interpretive check the report describes.
Output used: just a confirmation of which set was which, no wording taken
Verification: manually re-derived vocab_b - vocab_k vs vocab_k - vocab_b myself in
a scratch script to be sure i had labeled the two columns correctly before writing
up the "unwrapping merges word#hashtag with no space" conclusion

### 2026-07-13 - Mohamed - ticket 4 (threshold sweep)
Tool: Claude
Prompt/ask: had the threshold sweep 0.30-0.70 already coded and run (peak at
thr=0.45, f1=0.7568), noticed class_weight=balanced gave the exact same
P/R/F1 as thr=0.45 and asked claude if that's expected or a coincidence i should
double check
Output used: none, used the answer just to decide it was worth noting in the
ticket rather than treating it as a bug
Verification: reran both configs myself a second time to confirm the tie wasnt a
one-off (same 0.7756/0.7389/0.7568 both times), then read through the actual
predict_proba outputs for a few rows to see the tie made sense given how balanced
the classes roughly are here

### 2026-07-13 - Mohamed - ticket 4 (svm comparison) -- additional check
Tool: Codex
Prompt/ask: had fit_tfidf_sgd_svm already written and run (dev f1 0.7306, worse
than logreg). Asked codex, separately from the claude check above, whether it's
fair to call this "rejected" in the ticket given it wasn't separately tuned, since
i wanted more than one opinion before writing the limitation section.
Output used: none directly, but it did make me go add the "not exhaustively tuned"
line to the limitation section myself.
Verification: reread sklearn's SGDClassifier docs myself to double check default
max_iter/alpha before deciding how to word the limitation honestly, rather than
taking either tool's framing at face value.

### 2026-07-15 - Mohamed - ticket 4 (validation)
Tool: Claude
Prompt/ask: had ticket4_validation.py already written and run myself (mcnemar
34/30 p=0.708, bootstrap CI [0.7410,0.7908], oracle best thr=0.44 f1=0.7688), asked
claude to re-run the mcnemar count by hand to confirm p=0.708 given n_disc=64
Output used: none, confirmation only
Verification: recomputed binomtest(min(34,30)=30, 64, 0.5) myself separately, got
the same 0.708, matches

### 2026-07-15 - Shrook - ticket 5 (hard negatives)
Tool: Claude
Prompt/ask: had the hard-negative extraction already written and run (295 total
heldout errors, top 8 by confidence printed), asked claude to help me read through
the 8 printed examples a bit faster since a couple had garbled encoding characters
-- same "reading examples for purity/error inspection" category as the ticket 3
keyword work.
Output used: none, just reading assistance
Verification: went back to the raw text field in train.csv myself for each of the
8 ids to confirm what the "true" content of the tweet was despite the encoding
issues, before writing them up as genuinely ambiguous rather than pipeline bugs

### 2026-07-15 - Shrook - ticket 5 (label-fix experiment) -- second-opinion pass
Tool: Gemini
Prompt/ask: had apply_majority_vote_fix() and the before/after comparison already
written and run myself (12 rows changed, dev +0.0031, heldout -0.0060, mcnemar
p=0.0117). Since this was the ticket's key result, asked gemini as an independent
second check (separate from the claude session logged for this ticket) to re-run
the same before/after significance logic and see if it agreed with p=0.0117.
Output used: none of the code came from gemini, just used it as a second
confirmation source before writing "reject" in summary.csv
Verification: recomputed mcnemar myself from the printed 10-vs-1 breakdown
(binomtest(min(10,1)=1, 11, 0.5)) independently of both AI checks, got p=0.0117
again, matches, then manually inspected the duplicate groups in
results/data_quality_audit.csv to confirm the 12 changed rows really were the ones
i expected from the 11 conflict texts

### 2026-07-16 - Fatma - tickets 1 & 2 (wrap-up)
Tool: Claude
Prompt/ask: had already written the findings paragraphs for tickets 1 and 2
myself, asked claude to check my draft wording didnt overstate anything — specifically
whether saying "keep_hashtags_raw fixes hashtag handling" was accurate given what
the vocab-diff trace actually showed
Output used: none of claude's wording was used, but it prompted me to reread my
own vocab-diff output before finalizing and go back and change "fixes hashtag
handling" to something about the whitespace-merge bug instead
Verification: reread my own 07-14 tokenizer/vocab-diff printout again before
finalizing the ticket text, to make sure the final wording matched what the code
actually showed rather than the more "interesting-sounding" original claim

### 2026-07-16 - Mohamed - tickets 3 & 4 (wrap-up)
Tool: Claude
Prompt/ask: had already written up the findings for tickets 3 and 4 myself, asked
claude to check the numbers i quoted (30 fixed/34 new for ticket 4, 89.8% overlap
for ticket 3) against my own results/summary.csv row before finalizing
Output used: none, just a numbers cross-check
Verification: opened results/summary.csv and results/threshold_sweep.csv myself
and matched every number in my draft against the actual rows, line by line, before
submitting the ticket text

### 2026-07-17 - Shrook - final report (structure)
Tool: Claude
Prompt/ask: had already drafted the report outline myself following the 5 tickets'
structure, asked claude to check whether any heldout number in my draft was
actually from a dev-only sweep row mistakenly presented as a frozen result
Output used: none of the outline came from claude, but it did catch that one line
in my draft cited a mid-sweep threshold's dev number as if it were the final
heldout result - i fixed that myself
Verification: went through results/summary.csv row by row myself and matched every
heldout number in the report against it before finalizing

### 2026-07-18 - Fatma - ticket 1 (final tolerance re-check)
Tool: Claude
Prompt/ask: before submitting, asked claude to recheck my arithmetic one more time
on the reference-repro gap (0.7576-0.7574=0.0002) vs tolerance (0.001)
Output used: none, arithmetic sanity check only
Verification: did the subtraction myself by hand again too, matches what the
script printed as MATCHES

### 2026-07-19 - Mohamed - tickets 3 & 4 (final numbers check) -- cross-tool check
Tool: Gemini
Prompt/ask: asked gemini to help re-verify the mcnemar p-values quoted across
tickets 3 and 4 (p=1.0000 and p=0.7080) one more time before submission, as a
different tool from the ones already used on those tickets, just re-running the
binomtest logic by hand from the printed counts.
Output used: none, confirmation only
Verification: recomputed both p-values independently myself using the printed
right/wrong counts, matched both, no discrepancy found between my own math and
either AI tool's answer

### 2026-07-20 - Shrook - final report / log assembly
Tool: Claude
Prompt/ask: asked claude to help format this log file's headers consistently and
check every ticket (1-5) plus the report has at least one entry before we submit
Output used: only the formatting/spacing pass was kept, all prompts, outputs, and
verification text were written by each of us from our own terminal history and
notes, not generated by claude
Verification: manually checked the finished log against project_contract.json and
the five ticket files, confirmed tickets 1-5 and the report each have at least one
entry, and matched entry dates against our own terminal history timestamps