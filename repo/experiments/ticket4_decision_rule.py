"""
Ticket 4: Decision-rule and model ticket.

Sweeps threshold on DEV to show the precision/recall tradeoff (not just best F1),
compares class_weight='balanced' vs None, and trains a second CPU classifier
(linear SVM via SGD) to compare against the reference-repro Logistic Regression.
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data import load_raw, get_splits
from pipeline.model import (build_texts, fit_tfidf_logreg, fit_tfidf_sgd_svm, predict_scores,
                             scores_to_preds, evaluate, REFERENCE_REPRO_TFIDF_KWARGS, REFERENCE_REPRO_C)
from pipeline.artifacts import append_predictions, append_summary, append_threshold_sweep, compare_error_sets
from sklearn.metrics import precision_score, recall_score, f1_score

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    df = load_raw()
    train_df, dev_df, heldout_df = get_splits(df)
    contract = json.load(open(REPO_ROOT / "configs" / "project_contract.json"))

    train_texts = build_texts(train_df, "baseline")
    dev_texts = build_texts(dev_df, "baseline")
    heldout_texts = build_texts(heldout_df, "baseline")

    # --- Threshold sweep on dev, reference-repro LogReg ---
    vec, clf = fit_tfidf_logreg(train_texts, train_df["target"].values, seed=3102,
                                 C=REFERENCE_REPRO_C, **REFERENCE_REPRO_TFIDF_KWARGS)
    dev_scores = predict_scores(vec, clf, dev_texts)

    print("=== Threshold sweep on DEV (logreg, reference-repro config) ===")
    sweep_rows = []
    best_thr, best_f1 = 0.5, -1
    for thr in np.arange(0.30, 0.71, 0.05):
        preds = (dev_scores >= thr).astype(int)
        p = precision_score(dev_df["target"].values, preds, zero_division=0)
        r = recall_score(dev_df["target"].values, preds, zero_division=0)
        f1 = f1_score(dev_df["target"].values, preds)
        sweep_rows.append({"ticket": 4, "threshold": round(float(thr), 2),
                            "precision_target_1": round(p, 4), "recall_target_1": round(r, 4),
                            "f1_target_1": round(f1, 4)})
        print(f"thr={thr:.2f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    append_threshold_sweep(sweep_rows)
    print(f"\nBest dev threshold: {best_thr:.2f} (F1={best_f1:.4f}) -- frozen for heldout")

    # --- class_weight balanced vs None, at default threshold, on dev ---
    print("\n=== class_weight comparison on DEV ===")
    vec_bal, clf_bal = fit_tfidf_logreg(train_texts, train_df["target"].values, seed=3102,
                                         C=REFERENCE_REPRO_C, class_weight="balanced", **REFERENCE_REPRO_TFIDF_KWARGS)
    dev_scores_bal = predict_scores(vec_bal, clf_bal, dev_texts)
    dev_preds_bal = scores_to_preds(dev_scores_bal, 0.5)
    m_bal = evaluate(dev_df["target"].values, dev_preds_bal)
    print(f"class_weight=balanced  dev_f1={m_bal['f1_target_1']:.4f}  P={m_bal['precision_target_1']:.4f} R={m_bal['recall_target_1']:.4f}")
    dev_preds_default = scores_to_preds(dev_scores, 0.5)
    m_default = evaluate(dev_df["target"].values, dev_preds_default)
    print(f"class_weight=None      dev_f1={m_default['f1_target_1']:.4f}  P={m_default['precision_target_1']:.4f} R={m_default['recall_target_1']:.4f}")

    # --- second CPU classifier: linear SVM via SGD ---
    print("\n=== Second classifier: linear SVM (SGD, modified_huber) on DEV ===")
    vec_svm, clf_svm = fit_tfidf_sgd_svm(train_texts, train_df["target"].values, seed=3102, **REFERENCE_REPRO_TFIDF_KWARGS)
    dev_scores_svm = predict_scores(vec_svm, clf_svm, dev_texts)
    dev_preds_svm = scores_to_preds(dev_scores_svm, 0.5)
    m_svm = evaluate(dev_df["target"].values, dev_preds_svm)
    print(f"linear_svm (sgd)       dev_f1={m_svm['f1_target_1']:.4f}  P={m_svm['precision_target_1']:.4f} R={m_svm['recall_target_1']:.4f}")

    # Decide: pick the best dev config among {default thr, best thr, balanced, svm}
    candidates = {
        "logreg_thr0.5": (dev_preds_default, m_default["f1_target_1"]),
        f"logreg_thr{best_thr:.2f}": ((dev_scores >= best_thr).astype(int), best_f1),
        "logreg_balanced_thr0.5": (dev_preds_bal, m_bal["f1_target_1"]),
        "linear_svm_thr0.5": (dev_preds_svm, m_svm["f1_target_1"]),
    }
    winner_name = max(candidates, key=lambda k: candidates[k][1])
    print(f"\nWinner on dev: {winner_name} (dev_f1={candidates[winner_name][1]:.4f}) -- frozen for heldout")

    # --- Evaluate winner on heldout, compare errors to Ticket 1 reference-repro baseline ---
    heldout_scores_ref = predict_scores(vec, clf, heldout_texts)
    heldout_preds_ref = scores_to_preds(heldout_scores_ref, 0.5)
    base_wrong = set(heldout_df["id"][heldout_preds_ref != heldout_df["target"].values])

    if winner_name.startswith("logreg_thr"):
        thr_val = float(winner_name.replace("logreg_thr", ""))
        heldout_preds_final = (heldout_scores_ref >= thr_val).astype(int)
        heldout_scores_final = heldout_scores_ref
        model_name = f"tfidf_logreg_thr{thr_val:.2f}"
    elif winner_name == "logreg_balanced_thr0.5":
        heldout_scores_final = predict_scores(vec_bal, clf_bal, heldout_texts)
        heldout_preds_final = scores_to_preds(heldout_scores_final, 0.5)
        model_name = "tfidf_logreg_balanced"
    else:
        heldout_scores_final = predict_scores(vec_svm, clf_svm, heldout_texts)
        heldout_preds_final = scores_to_preds(heldout_scores_final, 0.5)
        model_name = "tfidf_linear_svm_sgd"

    final_metrics = evaluate(heldout_df["target"].values, heldout_preds_final)
    final_wrong = set(heldout_df["id"][heldout_preds_final != heldout_df["target"].values])
    fixed_ids, new_error_ids = compare_error_sets(base_wrong, final_wrong)

    heldout_lookup = heldout_df.set_index("id")
    base_preds_by_id = dict(zip(heldout_df["id"], heldout_preds_ref))
    final_preds_by_id = dict(zip(heldout_df["id"], heldout_preds_final))

    def classify_errors(ids, preds_by_id):
        fp, fn = 0, 0
        for _id in ids:
            true_label = heldout_lookup.loc[_id, "target"]
            pred_label = preds_by_id.get(_id)
            if pred_label == 1 and true_label == 0:
                fp += 1
            elif pred_label == 0 and true_label == 1:
                fn += 1
        return fp, fn

    fixed_fp, fixed_fn = classify_errors(fixed_ids, base_preds_by_id)
    new_fp, new_fn = classify_errors(new_error_ids, final_preds_by_id)

    print(f"\nHeldout: reference-repro F1={evaluate(heldout_df['target'].values, heldout_preds_ref)['f1_target_1']:.4f}  "
          f"{model_name} F1={final_metrics['f1_target_1']:.4f}")
    print(f"Fixed errors: {len(fixed_ids)} (fp fixed={fixed_fp}, fn fixed={fixed_fn})  "
          f"New errors: {len(new_error_ids)} (new fp={new_fp}, new fn={new_fn})")

    rows = []
    for _id, yt, yp, sc in zip(heldout_df["id"], heldout_df["target"], heldout_preds_final, heldout_scores_final):
        rows.append({"id": int(_id), "y_true": int(yt), "y_pred": int(yp), "score": float(sc),
                     "model_name": model_name, "ticket": 4})
    append_predictions(rows)

    append_summary({
        "ticket": 4, "model_name": model_name,
        "dev_f1_target_1": round(candidates[winner_name][1], 4),
        "heldout_f1_target_1": round(final_metrics["f1_target_1"], 4),
        "heldout_accuracy": round(final_metrics["accuracy"], 4),
        "fixed_fp": fixed_fp, "fixed_fn": fixed_fn, "new_fp": new_fp, "new_fn": new_fn,
        "decision": "adopt" if final_metrics["f1_target_1"] >= m_default["f1_target_1"] else "reject",
        "decision_reason": f"winner={winner_name} chosen on dev among threshold/class_weight/svm candidates",
    })


if __name__ == "__main__":
    main()
