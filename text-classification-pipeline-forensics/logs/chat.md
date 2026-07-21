## Entries

### 2026-07-10 - Fatma - pipeline scaffolding / ticket 1
Tool: Claude
Prompt/ask: Asked for a walkthrough of `pipeline/data.py` and `pipeline/preprocess.py` to
understand where the TF-IDF vectorizer is fit relative to the train/dev/heldout split,
since Ticket 1 flagged suspiciously high dev accuracy. Follow-up: "does fitting
TfidfVectorizer before calling the split function leak dev/heldout vocabulary into
train?" Third message: "show me the exact line to move and how to refit safely using
only `split_indices.json['train']`."
Output used: Confirmed the leak - `preprocess.py` called `vectorizer.fit_transform(full_df)`
before `data.py` sliced by split indices. Kept Claude's suggested reordering: fit the
vectorizer only on rows in `split_indices.json["train"]`, then `.transform()` dev and
heldout separately.
Verification: Re-ran `experiments/ticket1.py` on the corrected order and confirmed the
vocabulary size dropped (fewer dev/heldout-only tokens leaking in), and dev accuracy
dropped from an artificially high number down closer to the heldout number, which
matches the expected "no leakage" pattern described in the ticket.

### 2026-07-11 - Mohamed - ticket 3
Tool: ChatGPT
Prompt/ask: Described Ticket 3 (reported dev F1 not matching `project_contract.json`
tolerance) and asked ChatGPT to explain possible causes of a persistent F1 gap when
the dataset has a 2605/1962 class split. Asked it to list the most common reasons
(imbalance handling, decision threshold, wrong average type) before touching code.
Output used: Used the explanation as a checklist, not code - didn't paste anything
directly into the repo yet.
Verification: N/A yet, this session was purely diagnostic reasoning to plan the fix
before writing code (continued 2026-07-13).

### 2026-07-12 - Fatma - ticket 1
Tool: Claude
Prompt/ask: Followed up on the previous day's fix - asked Claude to help write a
regression-style check in `experiments/ticket1.py` that asserts the fitted vocabulary
only contains tokens seen in the train split, so the leakage bug can't silently
reappear.
Output used: Added the assertion helper Claude drafted (`assert_no_vocab_leak(vectorizer,
train_texts)`) into `experiments/ticket1.py`.
Verification: Ran the check against the corrected pipeline - it passed. Then
temporarily reverted the fix to confirm the assertion actually fails on the old
(buggy) ordering, so I know the test is meaningful and not just trivially passing.

### 2026-07-12 - Shrook - ticket 5
Tool: Gemini
Prompt/ask: Asked Gemini to compare the JSON produced by `pipeline/artifacts.py`
against the schema fields listed in `project_contract.json` and flag any missing or
mis-typed keys. Follow-up: "the contract expects `model_version` and a nested
`metrics.heldout` object - can you show me the minimal diff to `artifacts.py` to
produce that structure without changing the metric values themselves?"
Output used: Kept the restructuring of the export dict into the nested
`metrics.train / metrics.dev / metrics.heldout` shape and the added `model_version`
field. Did not use Gemini's suggestion to rename `f1_macro` to `f1` since the contract
explicitly used the longer name.
Verification: Diffed the new exported JSON key-by-key against `project_contract.json`'s
schema section and confirmed every required key was present with matching types
(string/float/int).

### 2026-07-13 - Fatma - ticket 2
Tool: Claude
Prompt/ask: Moved on to Ticket 2 - asked Claude to check whether `pipeline/data.py`
was actually reading `split_indices.json` or silently re-splitting with
`train_test_split` and a random seed, since the reported split sizes (4567/1523/1523)
didn't line up with what I was seeing in a debug print.
Output used: Confirmed `data.py` had a fallback branch that regenerated indices with
`np.random.seed(42)` whenever the json path argument was left as default `None`, which
is what `experiments/ticket2.py` was doing. Used Claude's one-line fix: making the
`split_path` argument required instead of defaulting to `None`, and raising a
`ValueError` if the loaded split doesn't match the expected sizes in `README_DATA.md`.
Verification: Re-ran `experiments/ticket2.py` with the split path passed explicitly
and printed split sizes - got exactly train 4567 / dev 1523 / heldout 1523 as stated
in `README_DATA.md`, and confirmed no row indices overlapped between the three sets.

### 2026-07-13 - Mohamed - ticket 3
Tool: Codex
Prompt/ask: Asked Codex to modify `pipeline/model.py` so `LogisticRegression` is
instantiated with `class_weight="balanced"`, and to add an option to log both macro
and weighted F1 so we can see which one the contract is actually checking. Second
prompt: "also add a short docstring explaining why class_weight matters for the
2605/1962 imbalance." Third prompt: "can you keep the existing `random_state` and
`max_iter` args unchanged?"
Output used: Kept the `class_weight="balanced"` change and the dual F1 logging. Left
`random_state=13, max_iter=1000` untouched as requested.
Verification: Re-ran `experiments/ticket3.py` on the dev split before/after the
change - F1 on the minority class improved noticeably, and the macro F1 now falls
inside the tolerance band listed in `project_contract.json` for Ticket 3.

### 2026-07-14 - Mohamed - ticket 4
Tool: ChatGPT
Prompt/ask: Ticket 4 flagged that our heldout F1 didn't match the contract's expected
value even after Ticket 3's fix. Asked ChatGPT to compare `project_contract.json`'s
`"metric": "f1_macro"` field against what `experiments/ticket4.py` was actually
computing.
Output used: ChatGPT pointed out `ticket4.py` was calling
`f1_score(y_true, y_pred, average="weighted")` instead of `average="macro"`. Kept the
one-line change to switch the average type; did not accept ChatGPT's unrelated
suggestion to also change the vectorizer's `ngram_range`, since that wasn't part of
this ticket's scope.
Verification: Recomputed heldout F1 with `average="macro"` and it matched the
contract's target value within the stated tolerance (compared numerically, not just
by eye).

### 2026-07-15 - Fatma - ticket 2
Tool: Claude
Prompt/ask: Asked Claude to help write a small assertion in `ticket2.py` confirming
`set(train_idx) & set(dev_idx) & set(heldout_idx) == set()` so a future refactor can't
silently reintroduce the random-split fallback bug.
Output used: Added the disjointness assertion exactly as suggested.
Verification: Ran it - passes on the current pipeline. Also manually spot-checked 10
row indices from each split by hand against `split_indices.json` to be sure the
assertion wasn't just checking an empty/degenerate case.

### 2026-07-15 - Shrook - ticket 5
Tool: Gemini
Prompt/ask: Asked Gemini to help write a small schema-validation function using
Python's `jsonschema` (or a manual dict-key check if that dependency isn't already in
the repo) so `artifacts.py`'s output can be automatically checked against
`project_contract.json` instead of relying on manual diffing.
Output used: Since `jsonschema` wasn't already a project dependency, took Gemini's
fallback suggestion - a plain-Python recursive key/type checker function
`validate_against_contract(exported_dict, contract_dict)` - to avoid adding a new
dependency the team hadn't approved.
Verification: Ran the validator against both the old (broken) and new artifact output
- it correctly failed on the old structure (missing `model_version`) and passed on
the fixed one.

### 2026-07-16 - Mohamed - ticket 4
Tool: Codex
Prompt/ask: Asked Codex to add a comment in `ticket4.py` documenting why
`average="macro"` is required here specifically (per the contract) versus
`"weighted"` used elsewhere in the codebase, so a future reader doesn't "fix" it back.
Output used: Kept the added comment block, lightly reworded it to match our own
docstring style used elsewhere in the repo.
Verification: Re-read the comment against `project_contract.json`'s own description
of the metric to make sure the documented reasoning was accurate, not just plausible.

### 2026-07-17 - Shrook - final report
Tool: Claude
Prompt/ask: Asked Claude to help outline a report structure that ties Tickets 1-5
together as a "forensics narrative" (bug found → root cause → fix → verification),
using the actual numbers from each ticket's `experiments/ticketN.py` run rather than
placeholders. Second prompt: "given these five findings, help me phrase the summary
paragraph without overstating what we actually verified."
Output used: Used Claude's section outline (Findings / Root Cause / Fix / Verification
per ticket) as the report's skeleton. Rewrote the summary paragraph myself after
Claude's draft used a stronger claim ("proves the pipeline is now fully correct") than
we could actually support - toned it down to "resolves the five identified
discrepancies against the contract."
Verification: Cross-checked every number quoted in the report draft (split sizes,
F1 values, class counts) directly against the corresponding `experiments/ticketN.py`
console output and `project_contract.json`, rather than trusting Claude's restated
numbers.

### 2026-07-18 - Fatma - tickets 1 & 2 (final check)
Tool: Claude
Prompt/ask: Before sign-off, asked Claude to help me re-derive by hand whether a
4567/1523/1523 split (70/15/15-ish) with a 2605/1962 class balance is internally
consistent (i.e., the numbers actually sum to the stated total and the class balance
ratio is plausible across all three splits).
Output used: Used Claude's arithmetic check as a sanity pass; did not change any code
based on this session.
Verification: Manually added 4567+1523+1523 and confirmed it matches the total row
count in the raw dataset file, and manually recomputed the class ratio (2605:1962 ≈
57:43) against a printed `value_counts()` from `pipeline/data.py` to confirm it wasn't
just an assumed number.

### 2026-07-19 - Mohamed - tickets 3 & 4 (cross-check)
Tool: Gemini
Prompt/ask: Asked Gemini (as a second opinion, since Ticket 3/4 fixes came from
ChatGPT/Codex) to independently review the final `pipeline/model.py` and
`experiments/ticket4.py` diffs and flag anything that looked inconsistent with
standard scikit-learn usage.
Output used: Gemini didn't flag anything requiring a change; used this purely as an
independent review pass, no code taken from this session.
Verification: Treated the absence of flags as a secondary confirmation, but the
primary verification remained the numeric F1 comparison against
`project_contract.json` done on 2026-07-13/14.

### 2026-07-20 - Shrook - final report / log assembly
Tool: Claude
Prompt/ask: Asked Claude to help format this AI Usage Log file itself - specifically
to make sure every entry followed the required `### [date] - [name] - [ticket]` format
before submission, and to check none of the five tickets were missing an entry.
Output used: Used Claude's formatting pass (spacing/heading consistency) but wrote all
factual content (prompts, outputs, verification steps) ourselves from memory and git
history, not generated by Claude.
Verification: Manually checked the finished log against the five tickets in
`project_contract.json` - confirmed Tickets 1, 2, 3, 4, 5 and the final report each
have at least one entry, and checked git blame/commit timestamps to make sure the
dates in the log roughly match when each corresponding commit actually landed.