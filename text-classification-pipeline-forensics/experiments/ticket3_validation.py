"""
Ticket 3 validation: shortcut redundancy and legitimacy checks.

Reproduces every number cited in tickets/ticket-3-shortcuts.md:
  - keyword_only / length_only / text_only heldout F1 (already in ticket3_shortcuts.py)
  - overlap between keyword_only-correct and text_only-correct heldout predictions
  - a combined text+keyword model, and whether its F1 gain is McNemar-significant
  - example tweets for the highest-purity keywords, used to judge legitimacy vs artifact
"""
import sys
from pathlib import Path
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data import load_raw, get_splits
from pipeline.model import build_texts, fit_tfidf_logreg, predict_scores, scores_to_preds, evaluate, REFERENCE_REPRO_TFIDF_KWARGS, REFERENCE_REPRO_C
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from experiments.ticket1_validation import mcnemar_test


def main():
    df = load_raw()
    train_df, dev_df, heldout_df = get_splits(df)
    y_true = heldout_df["target"].values

    train_texts = build_texts(train_df, "baseline")
    heldout_texts = build_texts(heldout_df, "baseline")
    vec_txt, clf_txt = fit_tfidf_logreg(train_texts, train_df["target"].values, seed=3102,
                                         C=REFERENCE_REPRO_C, **REFERENCE_REPRO_TFIDF_KWARGS)
    preds_txt = scores_to_preds(predict_scores(vec_txt, clf_txt, heldout_texts), 0.5)

    kw_vec = CountVectorizer(token_pattern=r"[^\x00-\x1f]+", lowercase=False)
    train_kw = train_df["keyword"].replace("", "MISSING")
    heldout_kw = heldout_df["keyword"].replace("", "MISSING")
    X_train_kw = kw_vec.fit_transform(train_kw)
    clf_kw = LogisticRegression(max_iter=1000, random_state=3102).fit(X_train_kw, train_df["target"].values)
    preds_kw = (clf_kw.predict_proba(kw_vec.transform(heldout_kw))[:, 1] >= 0.5).astype(int)

    print("=== Redundancy between keyword_only and text_only correct predictions ===")
    kw_right = set(heldout_df["id"][preds_kw == y_true])
    txt_right = set(heldout_df["id"][preds_txt == y_true])
    overlap = kw_right & txt_right
    print(f"keyword_only correct: {len(kw_right)}, text_only correct: {len(txt_right)}")
    print(f"overlap: {len(overlap)} ({len(overlap)/len(kw_right)*100:.1f}% of keyword_only-correct also caught by text)")
    print(f"keyword catches but text misses: {len(kw_right - txt_right)}")

    print("\n=== Combined text+keyword model ===")
    X_train_txt = vec_txt.transform(train_texts)
    X_train_combined = sp.hstack([X_train_txt, X_train_kw]).tocsr()
    clf_combined = LogisticRegression(max_iter=1000, C=REFERENCE_REPRO_C, random_state=3102).fit(
        X_train_combined, train_df["target"].values)
    X_heldout_combined = sp.hstack([vec_txt.transform(heldout_texts), kw_vec.transform(heldout_kw)]).tocsr()
    preds_combined = (clf_combined.predict_proba(X_heldout_combined)[:, 1] >= 0.5).astype(int)

    m_txt = evaluate(y_true, preds_txt)
    m_comb = evaluate(y_true, preds_combined)
    print(f"text_only heldout F1:     {m_txt['f1_target_1']:.4f}")
    print(f"text+keyword heldout F1:  {m_comb['f1_target_1']:.4f}")
    mcnemar_test(y_true, preds_txt, preds_combined, "text_only", "text_plus_keyword")

    print("\n=== Keyword purity table + example inspection ===")
    kw_target = train_df.groupby("keyword")["target"].agg(["mean", "count"])
    pure = kw_target[(kw_target["count"] >= 5) & ((kw_target["mean"] <= 0.05) | (kw_target["mean"] >= 0.95))]
    pure_sorted = pure.reindex(pure["count"].sort_values(ascending=False).index)
    print(f"{len(pure)} near-pure keywords (>=5 train rows, >=95% single-class) out of {kw_target.shape[0]} distinct keywords")
    print(pure_sorted.head(10))

    for kw in ["derailment", "aftershock", "wrecked", "body%20bags", "ruin"]:
        examples = train_df[train_df["keyword"] == kw][["text", "target"]].head(3)
        print(f"\n--- keyword={kw!r} ---")
        for _, r in examples.iterrows():
            print(f"  target={r['target']} text={r['text'][:90]!r}")


if __name__ == "__main__":
    main()
