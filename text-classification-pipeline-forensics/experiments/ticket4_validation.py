"""
Ticket 4 validation.

Reproduces every number cited in tickets/ticket-4-decision-rule.md beyond the
main threshold sweep / summary.csv row:
  - McNemar's exact test comparing thr=0.5 vs the dev-chosen thr=0.45 on heldout
  - bootstrap 95% CI on the winning threshold's heldout F1
  - an oracle threshold sweep directly on heldout (diagnostic only, never used to
    make the actual decision -- checks whether the dev-frozen threshold happened
    to generalize well or got lucky/unlucky)
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data import load_raw, get_splits
from pipeline.model import build_texts, fit_tfidf_logreg, predict_scores, scores_to_preds, REFERENCE_REPRO_TFIDF_KWARGS, REFERENCE_REPRO_C
from experiments.ticket1_validation import mcnemar_test, bootstrap_ci


def main():
    df = load_raw()
    train_df, dev_df, heldout_df = get_splits(df)
    y_true = heldout_df["target"].values

    train_texts = build_texts(train_df, "baseline")
    heldout_texts = build_texts(heldout_df, "baseline")
    vec, clf = fit_tfidf_logreg(train_texts, train_df["target"].values, seed=3102,
                                 C=REFERENCE_REPRO_C, **REFERENCE_REPRO_TFIDF_KWARGS)
    scores = predict_scores(vec, clf, heldout_texts)

    preds_thr05 = scores_to_preds(scores, 0.5)
    preds_thr045 = scores_to_preds(scores, 0.45)

    mcnemar_test(y_true, preds_thr05, preds_thr045, "thr0.5", "thr0.45")
    bootstrap_ci(y_true, preds_thr045)

    print("\n=== Oracle threshold check (diagnostic only -- NOT used to make the decision) ===")
    best_oracle_thr, best_oracle_f1 = 0.5, -1
    for thr in np.arange(0.30, 0.71, 0.01):
        f1 = f1_score(y_true, (scores >= thr).astype(int))
        if f1 > best_oracle_f1:
            best_oracle_f1, best_oracle_thr = f1, thr
    print(f"Oracle best heldout threshold: {best_oracle_thr:.2f}  F1={best_oracle_f1:.4f}")
    print(f"Dev-chosen threshold: 0.45  F1={f1_score(y_true, preds_thr045):.4f}")
    print(f"Gap between oracle and dev-chosen: {best_oracle_f1 - f1_score(y_true, preds_thr045):+.4f}")


if __name__ == "__main__":
    main()
