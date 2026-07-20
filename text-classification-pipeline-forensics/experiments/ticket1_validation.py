"""
Ticket 1 validation / discrepancy probes.

Reproduces every number cited in tickets/ticket-1-baseline.md under "Split
integrity check" and "Discrepancy probes":
  - split leakage / duplicate-id check
  - determinism check (3 independent retrains)
  - bootstrap 95% CI on heldout F1 for the reference-repro config
  - McNemar's exact test comparing naive_default vs reference_repro on heldout
"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data import load_raw, get_splits
from pipeline.model import build_texts, fit_tfidf_logreg, predict_scores, scores_to_preds, evaluate


def split_integrity_check(df, train_df, dev_df, heldout_df):
    train_ids, dev_ids, heldout_ids = set(train_df["id"]), set(dev_df["id"]), set(heldout_df["id"])
    print("=== Split integrity check ===")
    print("train/dev id overlap:    ", len(train_ids & dev_ids))
    print("train/heldout id overlap:", len(train_ids & heldout_ids))
    print("dev/heldout id overlap:  ", len(dev_ids & heldout_ids))
    print("union of splits == full dataset:", len(train_ids | dev_ids | heldout_ids) == len(df))
    print("duplicate ids in train.csv:", int(df["id"].duplicated().sum()))


def determinism_check(train_texts, train_labels, heldout_texts, y_true, n_runs=3):
    print("\n=== Determinism check (retrain reference-repro config from scratch) ===")
    from pipeline.model import REFERENCE_REPRO_TFIDF_KWARGS, REFERENCE_REPRO_C
    results = []
    for run in range(n_runs):
        vec, clf = fit_tfidf_logreg(train_texts, train_labels, seed=3102, C=REFERENCE_REPRO_C, **REFERENCE_REPRO_TFIDF_KWARGS)
        scores = predict_scores(vec, clf, heldout_texts)
        preds = scores_to_preds(scores, 0.5)
        f1 = f1_score(y_true, preds)
        results.append(round(f1, 6))
        print(f"run {run}: heldout_f1={f1:.6f}")
    print("identical across runs:", len(set(results)) == 1)
    return results


def bootstrap_ci(y_true, preds, n_boot=2000, seed=3102):
    print(f"\n=== Bootstrap 95% CI on heldout F1 (n_boot={n_boot}) ===")
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_f1s = np.array([
        f1_score(y_true[idx], preds[idx])
        for idx in (rng.randint(0, n, n) for _ in range(n_boot))
    ])
    lo, hi = np.percentile(boot_f1s, [2.5, 97.5])
    point = f1_score(y_true, preds)
    print(f"Point estimate: {point:.4f}")
    print(f"95% CI: [{lo:.4f}, {hi:.4f}]")
    return lo, hi, point


def mcnemar_test(y_true, preds_a, preds_b, label_a="A", label_b="B"):
    print(f"\n=== McNemar's exact test: {label_a} vs {label_b} ===")
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    both_right = int((correct_a & correct_b).sum())
    both_wrong = int((~correct_a & ~correct_b).sum())
    n_disc = b + c
    p = binomtest(min(b, c), n_disc, 0.5).pvalue if n_disc > 0 else 1.0
    print(f"{label_a}-only-right: {b}   {label_b}-only-right: {c}")
    print(f"both right: {both_right}   both wrong: {both_wrong}")
    print(f"McNemar exact test p-value: {p:.4f}")
    return b, c, p


def main():
    df = load_raw()
    train_df, dev_df, heldout_df = get_splits(df)
    split_integrity_check(df, train_df, dev_df, heldout_df)

    train_texts = build_texts(train_df, "baseline")
    heldout_texts = build_texts(heldout_df, "baseline")
    y_true = heldout_df["target"].values

    determinism_check(train_texts, train_df["target"].values, heldout_texts, y_true)

    from pipeline.model import REFERENCE_REPRO_TFIDF_KWARGS, REFERENCE_REPRO_C
    vec_repro, clf_repro = fit_tfidf_logreg(train_texts, train_df["target"].values, seed=3102,
                                             C=REFERENCE_REPRO_C, **REFERENCE_REPRO_TFIDF_KWARGS)
    preds_repro = scores_to_preds(predict_scores(vec_repro, clf_repro, heldout_texts), 0.5)
    bootstrap_ci(y_true, preds_repro)

    vec_naive, clf_naive = fit_tfidf_logreg(train_texts, train_df["target"].values, seed=3102, C=1.0,
                                             ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
    preds_naive = scores_to_preds(predict_scores(vec_naive, clf_naive, heldout_texts), 0.5)

    mcnemar_test(y_true, preds_naive, preds_repro, "naive_default", "reference_repro")


if __name__ == "__main__":
    main()
