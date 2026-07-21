## Entries

### 2026-07-10 - Fatma - ticket 1 (baseline discrepancy)
Tool: Claude
Prompt/ask: pasted data.py and model.py and asked how to set up ticket1_baseline.py
so it runs the "naive default" tfidf+logreg (bigrams, sublinear_tf, C=1.0) against
the "reference repro" config, and log both to summary.csv, comparing heldout F1
against reference_baseline_f1 in the contract file.
Output used: kept the two-variant run_variant() structure and the gap/tolerance
print at the end.
Verification: ran it, naive default gave dev F1 0.7449 and heldout F1 0.7440 vs
reference 0.7574 (tol 0.001) -> gap -0.0134, does NOT match. reference-repro config:
dev F1 0.7514, heldout F1 0.7576, gap +0.0002 -> MATCHES. so the "discrepancy" from
the ticket is real and it's just a config difference (ngram/min_df/max_df/C), not a
bug. ran both variants twice to make sure the gap direction didnt change.

### 2026-07-10 - Mohamed - ticket 3 (shortcuts, first look)
Tool: ChatGPT
Prompt/ask: explained ticket 3 and asked if a keyword-only one hot model + a length
only model (char len + word count) are ok "shallow baselines" to compare against the
real tfidf model.
Output used: used the 3-model layout (keyword_only / length_only / text_only) as the
skeleton for ticket3_shortcuts.py.
Verification: n/a yet, this was just planning before writing anything. continued
next day once i actually had numbers.

### 2026-07-11 - Fatma - ticket 1 (validation probes)
Tool: Claude
Prompt/ask: asked claude to help write the split integrity check (train/dev/heldout
id overlaps) plus a determinism check that retrains the reference-repro config 3x to
make sure heldout F1 doesnt move around, wanted to rule out leakage before we just
say "its the config."
Output used: kept split_integrity_check() and the 3-run determinism loop basically
as-is.
Verification: all pairwise overlaps came back 0, union of splits = full dataset,
duplicated ids = 0, so no leakage. all 3 determinism runs gave heldout f1 = 0.757601
exactly, so seed 3102 really is deterministic here. good, rules out 2 of the possible
explanations for the gap.

### 2026-07-11 - Fatma - ticket 1 (is the gap even real)
Tool: Claude
Prompt/ask: same convo, asked if the naive vs reference-repro heldout difference
(0.7440 vs 0.7576) is actually statistically meaningful or if its just noise from a
small heldout set, and how to set up mcnemar for that.
Output used: kept the mcnemar_test helper using scipy binomtest.
Verification: naive-only-right=32, ref_repro-only-right=37, both right=1191, both
wrong=263, p=0.6305. NOT significant on its own lol. so worth saying in the report
that the 1.3pt f1 gap isnt something we can call "proven" at n=1523, even tho the
raw numbers clearly match the contract tolerance.

### 2026-07-11 - Mohamed - ticket 3 (shortcut results)
Tool: ChatGPT
Prompt/ask: pasted the dev numbers (keyword_only 0.6599, length_only 0.4358,
text_only 0.7514) and asked how to check if keyword is correlated with target in a
way thats a dataset artifact and not real signal
Output used: took the suggestion to do per-keyword purity (count>=5 and mean <=0.05
or >=0.95)
Verification: ran it, got 21 out of 222 keywords that hit that threshold. top ones:
ruin (mean 0.036, n=28), derailment (mean 1.0, n=27), wrecked (mean 0.038, n=26),
debris (mean 1.0, n=23), typhoon (mean 0.957, n=23). read a few examples by hand -
"ruin"/"wrecked" show up in non-disaster tweets (figurative, like "dont let that
ruin your year") vs derailment/debris are literal disaster tweets. so its a real
pattern in the data, not a bug, but probably wont generalize to other datasets.

### 2026-07-12 - Shrook - ticket 5 (data quality, first pass)
Tool: Gemini
Prompt/ask: gave gemini the ticket 5 brief, asked for help writing
find_duplicate_conflicts (exact text dupes in train w disagreeing labels) and a
near-dup version that normalizes text first. also asked if near-dupes in dev/heldout
should ever be used to fix anything
Output used: kept the groupby+filter approach for both, and its answer that
dev/heldout near-dupes just get logged "ambiguous" and never used for a fix, only
train rows get fix/keep_but_flag
Verification: found 11 distinct texts in train with disagreeing labels (like "To
fight bioterrorism sir." showing up as both 0 and 1), and 78 normalized-text groups
(287 rows) with disagreement across everything. read like 5 of the 11 conflicts
manually to make sure it wasnt just a grouping bug, it wasnt, they're genuinely the
same text.

### 2026-07-12 - Mohamed - ticket 3 (validation, redundancy)
Tool: Codex
Prompt/ask: asked codex to write the heldout overlap check between keyword_only
correct and text_only correct, plus a combined text+keyword model (sparse hstack),
reusing mcnemar_test from ticket1 instead of copy pasting it again
Output used: kept the hstack combine step + reused the mcnemar helper
Verification: keyword_only correct on heldout = 1125, text_only correct = 1228,
overlap 1010 which is 89.8%. so keyword mostly just repeats what text already knows,
only catches 115 that text misses. combined model heldout f1 = 0.7653 vs text-only
0.7576 which LOOKS like a nice bump but mcnemar came back p=1.0000 (58 vs 58, exactly
cancels out). so made sure we dont report the 0.77 gain as if its real, its not
significant at all even tho the number went up.

### 2026-07-13 - Fatma - ticket 2 (normalization sweep)
Tool: Claude
Prompt/ask: asked how to sweep the CONFIGS in preprocess.py on dev only, freeze the
winner, then eval once on heldout vs ticket 1's baseline config, with
compare_error_sets giving fixed/new fp+fn instead of just a plain accuracy delta
Output used: kept the sweep-then-freeze setup and the fixed/new fp fn reporting
Verification: dev sweep - raw 0.7504, baseline 0.7514, aggressive 0.7514,
keep_hashtags_raw 0.7518 (best, barely), keep_case 0.7514. froze keep_hashtags_raw,
ran heldout once: baseline f1 0.7576 -> keep_hashtags_raw f1 0.7595. fixed 3 errors
(all 3 were false positives, 0 fn), 0 new errors. read the 3 example ids printed
(2329, 7908, 767) just to double check they were real, they were.

### 2026-07-14 - Fatma - ticket 2 (wait why does this even work)
Tool: Claude
Prompt/ask: asked why keep_hashtags_raw changes ANYTHING since sklearns default
tokenizer strips # anyway - noticed 2 of the 3 fixed ids (2329, 767) dont even have
hashtags in them which seemed weird
Output used: kept the tokenizer analyzer probe + the fitted vocab diff between
configs tracing back to source tweets
Verification: ran mcnemar first on baseline vs keep_hashtags_raw: 0-vs-3, p=0.25 (not
sig, only 3 flips so thats expected). then actually looked at the 3 changed ids -
ids 767 and 2329 have byte-for-byte IDENTICAL normalized text between the two
configs, so those 2 flips cant be about hashtags at all, something else is going on
with the retrain. only id 7908 actually had different text
(ccot tcot radiation... vs #ccot #tcot #radiation...). vocab size was literally the
same (11156 both) and tokenizer confirmed # gets stripped either way so tokens come
out identical. the 6 words that differ between vocabs turned out to be mojibake junk
(tcotåêccot, ûïhannaph, ûóbbc - garbled Û_ /åÊ characters merging adjacent hashtags
into one token). so real finding: keep_hashtags_raw doesnt help bc of hashtag
"signal", its a side effect of how it handles this encoding garbage differently.
glad we checked this instead of just writing "hashtags carry signal" which is what
the first draft basically implied.

### 2026-07-13 - Mohamed - ticket 4 (threshold sweep)
Tool: ChatGPT
Prompt/ask: asked for the dev threshold sweep 0.30-0.70 step 0.05 logging P/R/F1, and
class_weight balanced vs none at default threshold
Output used: kept the sweep loop + balanced vs default comparison
Verification: sweep peaked at thr=0.45, f1=0.7568 (P=0.7756 R=0.7389). noticed
class_weight=balanced gave the EXACT same P/R as thr=0.45 which makes sense, theyre
both just shifting the decision boundary the same way basically, not really 2
different fixes.

### 2026-07-13 - Mohamed - ticket 4 (svm + decide)
Tool: ChatGPT
Prompt/ask: same session, asked how to add a linear svm (sgd, modified_huber) as a
2nd classifier to compare, and how to set the tie-break priority so threshold-only
wins over retraining if f1 ties
Output used: kept fit_tfidf_sgd_svm + the priority dict tiebreak
Verification: svm dev f1 = 0.7306, worse than either logreg option so its out.
winner = logreg_thr0.45 (dev f1 0.7568). checked by hand that priority dict actually
breaks ties right.

### 2026-07-15 - Mohamed - ticket 4 (validation)
Tool: Codex
Prompt/ask: asked codex for ticket4_validation.py - mcnemar for thr0.5 vs thr0.45 on
heldout, bootstrap CI on the winning threshold, and an oracle sweep on heldout marked
clearly as diagnostic only (not the actual decision)
Output used: kept all 3 checks + the "not used to decide" comment so no one mistakes
it later
Verification: heldout ref-repro f1 0.7576 -> thr0.45 f1 0.7666. fixed 30 errors (all
30 were fn, 0 fp) but introduced 34 new ones (all fp, 0 fn) - so its trading fn for
fp, not a free win, need to say that clearly in the report. mcnemar thr0.5 vs 0.45:
34 vs 30, p=0.708, not significant even tho raw f1 moved. bootstrap CI on 0.7666:
[0.7410, 0.7908] which is wide enough that 0.7576 sits comfortably inside it lol.
oracle best threshold was 0.44 (f1 0.7688), only +0.0022 better than our 0.45 - so
dev-frozen choice generalized fine, wasnt lucky, but also not gonna oversell 0.7666
as some proven big improvement given the mcnemar/CI stuff above.

### 2026-07-15 - Shrook - ticket 5 (hard negatives)
Tool: Gemini
Prompt/ask: asked for the hard negative extraction (heldout rows the reference-repro
model gets wrong AND confident, using abs(score-0.5) as distance from boundary),
made sure it never edits heldout, just reports
Output used: kept the confidence-sorted extraction + "ambiguous" disposition
Verification: 295 total heldout errors out of 1523. read the top 8 most confident
wrong ones by hand - most are genuinely hard/ambiguous (metaphorical "wrecked",
"debris" in a shoe ad, a hellfire bible quote) not pipeline bugs, so writing these up
as model limitations not data errors.

### 2026-07-15 - Shrook - ticket 5 (label fix experiment)
Tool: Gemini
Prompt/ask: same day, asked how to set up the actual experiment - apply majority vote
fix from the 11 dupe conflicts to TRAIN only, retrain the same config unchanged,
compare dev/heldout before and after with mcnemar on heldout
Output used: kept apply_majority_vote_fix() and the before/after comparison
Verification: 12 train rows changed (out of 4567). dev f1 actually went UP
0.7514->0.7545 (+0.0031) but heldout f1 went DOWN 0.7576->0.7516 (-0.0060). mcnemar on
heldout: 10 vs 1, p=0.0117, which IS significant. so this is basically the whole
point of ticket 5 - the fix looks good on dev but is a real, confirmed regression on
heldout. logged decision=reject with the p value as the reason. felt good that we
didnt just go "dev improved, ship it" and actually checked heldout first.

### 2026-07-16 - Fatma - tickets 1 & 2 wrapup
Tool: Claude
Prompt/ask: asked claude to help draft the findings section for tickets 1+2 - the
config-driven gap (not leakage, not nondeterminism, and technically not even
significant per mcnemar p=0.6305) and the normalization finding
Output used: used the draft as a starting point but rewrote the part about
keep_hashtags_raw - claude's first draft said something like "preserves hashtag
signal" which is just wrong based on what we actually found (tokens are identical,
its mojibake related, and 2 of 3 flips werent even from the hashtag change at all)
Verification: went back and reread our own 07-14 output before finalizing the
wording so we dont ship a claim the evidence literally contradicts

### 2026-07-16 - Mohamed - tickets 3 & 4 wrapup
Tool: ChatGPT
Prompt/ask: asked chatgpt to help summarize ticket 3 (keyword mostly redundant w
text, some keywords look like artifacts) and ticket 4 (thr=0.45 won on dev, adopted,
but heldout gain trades fn for fp and isnt mcnemar-significant)
Output used: used the structure but swapped in our actual numbers, chatgpt's first
draft guessed some fixed/new error counts that were close but not exactly right
Verification: went line by line comparing every number in the draft against our real
terminal output (30 fixed/34 new, not the guessed numbers) before keeping any of it,
caught 2 wrong numbers this way

### 2026-07-17 - Shrook - final report (structure)
Tool: Claude
Prompt/ask: asked claude to help outline the report so it follows the tickets'
actual structure (diagnose -> decide on dev -> freeze -> eval heldout once ->
significance check), and to flag if we're reporting any heldout number that skipped
the "freeze on dev first" step
Output used: used the outline, and claude actually caught a spot where our draft
quoted a heldout number for one of the mid-sweep thresholds in ticket 4 like it was
a real result - fixed that, only the frozen thr=0.45 heldout number counts
Verification: went through summary.csv row by row and matched every heldout number in
the report draft against it

### 2026-07-18 - Fatma - ticket 1 (double check tolerance)
Tool: Claude
Prompt/ask: before we submit, asked claude to help me re-check by hand that the
reference-repro gap (+0.0002) is actually inside the contract's tolerance (0.001),
just wanted a second pass on the arithmetic
Output used: just used this as a sanity check, no code changed
Verification: recomputed 0.7576 - 0.7574 = 0.0002 myself, compared to tolerance
0.001, yep matches, same as what the script already printed

### 2026-07-19 - Mohamed - tickets 3&4 (second opinion)
Tool: Gemini
Prompt/ask: since tickets 3/4 code came from chatgpt/codex, asked gemini to look over
the final ticket3_validation.py and ticket4_decision_rule.py diffs for anything off
with the sklearn usage, specifically the hstack step and the priority tiebreak dict
Output used: gemini didnt flag anything, so nothing changed, just used this as an
extra pair of eyes
Verification: real verification was still the mcnemar/bootstrap numbers from 07-12
and 07-15, this was just a bonus review pass not a correctness check by itself

### 2026-07-20 - Shrook - final report / log cleanup
Tool: Claude
Prompt/ask: asked claude to just help format this log file so all the headers follow
the required format and to double check we have an entry for every ticket (1-5) plus
the report before we turn it in
Output used: only used claude for the formatting pass, all the actual prompts,
outputs and verification text were written by us from memory + terminal history
Verification: manually checked the final log against project_contract.json and the
five ticket files, confirmed tickets 1-5 and the report all have at least one entry,
and roughly matched entry dates against when we actually ran each script (from
terminal history/timestamps)