"""
Ticket 5 validation: does the proposed training-label correction actually help?

This is the ticket's real controlled experiment: apply the majority-vote label fix
for exact-duplicate conflicts (see experiments/ticket5_data_quality.py) to
train_ids ONLY, retrain the reference-repro model unchanged, and compare dev/heldout
F1 before and after -- with a paired McNemar significance test on heldout.
Heldout labels/examples are never touched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data import load_raw, get_splits
from pipeline.model import build_texts, fit_tfidf_logreg, predict_scores, scores_to_preds, evaluate, REFERENCE_REPRO_TFIDF_KWARGS, REFERENCE_REPRO_C
from experiments.ticket1_validation import mcnemar_test


def apply_majority_vote_fix(train_df):
    conflicts = train_df.groupby("text")["target"].agg(["nunique", "count", "mean"])
    conflicts = conflicts[(conflicts["nunique"] > 1) & (conflicts["count"] >= 2)]
    fixed_df = train_df.copy()
    n_changed = 0
    for text, row in conflicts.iterrows():
        majority_label = 1 if row["mean"] >= 0.5 else 0
        mask = fixed_df["text"] == text
        n_changed += int((fixed_df.loc[mask, "target"] != majority_label).sum())
        fixed_df.loc[mask, "target"] = majority_label
    return fixed_df, n_changed


def main():
    df = load_raw()
    train_df, dev_df, heldout_df = get_splits(df)
    y_dev, y_heldout = dev_df["target"].values, heldout_df["target"].values
    dev_texts = build_texts(dev_df, "baseline")
    heldout_texts = build_texts(heldout_df, "baseline")

    # baseline: no label fix
    train_texts = build_texts(train_df, "baseline")
    vec, clf = fit_tfidf_logreg(train_texts, train_df["target"].values, seed=3102,
                                 C=REFERENCE_REPRO_C, **REFERENCE_REPRO_TFIDF_KWARGS)
    preds_dev_base = scores_to_preds(predict_scores(vec, clf, dev_texts), 0.5)
    preds_held_base = scores_to_preds(predict_scores(vec, clf, heldout_texts), 0.5)
    m_dev_base = evaluate(y_dev, preds_dev_base)
    m_held_base = evaluate(y_heldout, preds_held_base)
    print("=== Baseline (no label fix) ===")
    print(f"dev F1={m_dev_base['f1_target_1']:.4f}  heldout F1={m_held_base['f1_target_1']:.4f}")

    # apply majority-vote fix to train_ids only
    fixed_train_df, n_changed = apply_majority_vote_fix(train_df)
    print(f"\nApplied majority-vote label fix to {n_changed} train rows (out of {len(train_df)})")

    train_texts_fixed = build_texts(fixed_train_df, "baseline")
    vec2, clf2 = fit_tfidf_logreg(train_texts_fixed, fixed_train_df["target"].values, seed=3102,
                                   C=REFERENCE_REPRO_C, **REFERENCE_REPRO_TFIDF_KWARGS)
    preds_dev_fixed = scores_to_preds(predict_scores(vec2, clf2, dev_texts), 0.5)
    preds_held_fixed = scores_to_preds(predict_scores(vec2, clf2, heldout_texts), 0.5)
    m_dev_fixed = evaluate(y_dev, preds_dev_fixed)
    m_held_fixed = evaluate(y_heldout, preds_held_fixed)
    print("=== After label fix ===")
    print(f"dev F1={m_dev_fixed['f1_target_1']:.4f}  heldout F1={m_held_fixed['f1_target_1']:.4f}")
    print(f"dev delta: {m_dev_fixed['f1_target_1']-m_dev_base['f1_target_1']:+.4f}")
    print(f"heldout delta: {m_held_fixed['f1_target_1']-m_held_base['f1_target_1']:+.4f}")

    mcnemar_test(y_heldout, preds_held_base, preds_held_fixed, "no_fix", "label_fix")


if __name__ == "__main__":
    main()
