"""
Ticket 3: Feature and shortcut audit.

Measures how much predictive signal comes from `keyword` and tweet length alone
(shallow/metadata features) vs the actual TF-IDF text model, and whether keyword
signal looks like legitimate task information or a dataset artifact (e.g. does
keyword correlate with target in a way that would not generalize, such as via
near-duplicate templated tweets sharing a keyword).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data import load_raw, get_splits
from pipeline.model import build_texts, fit_tfidf_logreg, predict_scores, scores_to_preds, evaluate, REFERENCE_REPRO_TFIDF_KWARGS, REFERENCE_REPRO_C
from pipeline.artifacts import append_summary, append_predictions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parent.parent


def keyword_only_model(train_df, eval_df):
    """Signal from `keyword` field alone (one-hot), ignoring text entirely."""
    vec = CountVectorizer(token_pattern=r"[^\x00-\x1f]+", lowercase=False)
    # treat each row's keyword (possibly empty) as a single "token"
    train_kw = train_df["keyword"].replace("", "MISSING")
    eval_kw = eval_df["keyword"].replace("", "MISSING")
    X_train = vec.fit_transform(train_kw)
    X_eval = vec.transform(eval_kw)
    clf = LogisticRegression(max_iter=1000, random_state=3102)
    clf.fit(X_train, train_df["target"].values)
    scores = clf.predict_proba(X_eval)[:, 1]
    preds = (scores >= 0.5).astype(int)
    return preds, scores


def length_only_model(train_df, eval_df):
    """Signal from character length + word count alone."""
    def feats(d):
        char_len = d["text"].str.len().values.reshape(-1, 1)
        word_len = d["text"].str.split().apply(len).values.reshape(-1, 1)
        return np.hstack([char_len, word_len])
    X_train, X_eval = feats(train_df), feats(eval_df)
    clf = LogisticRegression(max_iter=1000, random_state=3102)
    clf.fit(X_train, train_df["target"].values)
    scores = clf.predict_proba(X_eval)[:, 1]
    preds = (scores >= 0.5).astype(int)
    return preds, scores


def text_only_model(train_df, eval_df):
    train_texts = build_texts(train_df, "baseline")
    eval_texts = build_texts(eval_df, "baseline")
    vec, clf = fit_tfidf_logreg(train_texts, train_df["target"].values, seed=3102,
                                 C=REFERENCE_REPRO_C, **REFERENCE_REPRO_TFIDF_KWARGS)
    scores = predict_scores(vec, clf, eval_texts)
    preds = scores_to_preds(scores, 0.5)
    return preds, scores


def main():
    df = load_raw()
    train_df, dev_df, heldout_df = get_splits(df)
    contract = json.load(open(REPO_ROOT / "configs" / "project_contract.json"))

    print("=== Signal audit on DEV split (choosing nothing, just measuring) ===")
    results = {}
    for name, fn in [("keyword_only", keyword_only_model),
                      ("length_only", length_only_model),
                      ("text_only (TF-IDF, reference config)", text_only_model)]:
        preds, scores = fn(train_df, dev_df)
        m = evaluate(dev_df["target"].values, preds)
        results[name] = m
        print(f"{name:40s} dev_f1={m['f1_target_1']:.4f}  P={m['precision_target_1']:.4f} R={m['recall_target_1']:.4f}")

    # keyword-target correlation strength: how many distinct keywords are near-pure in one class
    kw_target = train_df.groupby("keyword")["target"].agg(["mean", "count"])
    pure_keywords = kw_target[(kw_target["count"] >= 5) & ((kw_target["mean"] <= 0.05) | (kw_target["mean"] >= 0.95))]
    print(f"\nKeywords with >=5 train examples and >=95% single-class purity: {len(pure_keywords)} "
          f"out of {kw_target.shape[0]} distinct keywords")
    print("Top 5 purest high-count keywords:")
    print(pure_keywords.reindex(pure_keywords["count"].sort_values(ascending=False).index).head(5))

    # Final held-out numbers for the text-only reference model (this is the real ticket-3
    # deliverable number: how much of the heldout score is carried by text vs shallow features)
    print("\n=== Heldout evaluation (frozen; for reporting only) ===")
    for name, fn in [("keyword_only", keyword_only_model),
                      ("length_only", length_only_model),
                      ("text_only (TF-IDF, reference config)", text_only_model)]:
        preds, scores = fn(train_df, heldout_df)
        m = evaluate(heldout_df["target"].values, preds)
        print(f"{name:40s} heldout_f1={m['f1_target_1']:.4f}")
        if "text_only" in name:
            rows = []
            for _id, yt, yp, sc in zip(heldout_df["id"], heldout_df["target"], preds, scores):
                rows.append({"id": int(_id), "y_true": int(yt), "y_pred": int(yp), "score": float(sc),
                             "model_name": "tfidf_logreg_text_only", "ticket": 3})
            append_predictions(rows)
            append_summary({
                "ticket": 3, "model_name": "tfidf_logreg_text_only",
                "dev_f1_target_1": round(results["text_only (TF-IDF, reference config)"]["f1_target_1"], 4),
                "heldout_f1_target_1": round(m["f1_target_1"], 4),
                "heldout_accuracy": round(m["accuracy"], 4),
                "fixed_fp": "", "fixed_fn": "", "new_fp": "", "new_fn": "",
                "decision": "n/a - diagnostic ticket, no model swap",
                "decision_reason": f"keyword_only and length_only measured as shallow-signal floors; "
                                    f"{len(pure_keywords)} near-pure keywords found in train (possible artifact)",
            })


if __name__ == "__main__":
    main()
