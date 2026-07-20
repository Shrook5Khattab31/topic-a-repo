"""
Ticket 1: Baseline discrepancy diagnosis.

Trains the TF-IDF + Logistic Regression baseline on train_ids, evaluates on
heldout_ids, and compares against configs/project_contract.json within tolerance.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data import load_raw, get_splits
from pipeline.model import (build_texts, fit_tfidf_logreg, predict_scores, scores_to_preds,
                             evaluate, REFERENCE_REPRO_TFIDF_KWARGS, REFERENCE_REPRO_C)
from pipeline.artifacts import append_predictions, append_summary

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_variant(train_df, dev_df, heldout_df, model_name, tfidf_kwargs, C, contract):
    train_texts = build_texts(train_df, "baseline")
    dev_texts = build_texts(dev_df, "baseline")
    heldout_texts = build_texts(heldout_df, "baseline")

    vec, clf = fit_tfidf_logreg(train_texts, train_df["target"].values, seed=3102, C=C, **tfidf_kwargs)

    dev_scores = predict_scores(vec, clf, dev_texts)
    dev_preds = scores_to_preds(dev_scores, 0.5)
    dev_metrics = evaluate(dev_df["target"].values, dev_preds)

    heldout_scores = predict_scores(vec, clf, heldout_texts)
    heldout_preds = scores_to_preds(heldout_scores, 0.5)
    heldout_metrics = evaluate(heldout_df["target"].values, heldout_preds)

    ref_f1 = contract["reference_baseline_f1"]
    tol = contract["tolerance"]
    gap = heldout_metrics["f1_target_1"] - ref_f1
    matches = abs(gap) <= tol

    print(f"\n--- {model_name} ---")
    print(f"Dev F1 (target=1):     {dev_metrics['f1_target_1']:.4f}")
    print(f"Heldout F1 (target=1): {heldout_metrics['f1_target_1']:.4f}")
    print(f"Reference F1:          {ref_f1:.4f}  (tolerance {tol})")
    print(f"Gap: {gap:+.4f}  -> {'MATCHES' if matches else 'DOES NOT MATCH'} reference within tolerance")
    print(f"Heldout accuracy: {heldout_metrics['accuracy']:.4f}")
    print(f"Heldout precision/recall (target=1): {heldout_metrics['precision_target_1']:.4f} / {heldout_metrics['recall_target_1']:.4f}")

    rows = []
    for _id, yt, yp, sc in zip(heldout_df["id"], heldout_df["target"], heldout_preds, heldout_scores):
        rows.append({"id": int(_id), "y_true": int(yt), "y_pred": int(yp), "score": float(sc),
                     "model_name": model_name, "ticket": 1})
    append_predictions(rows)

    append_summary({
        "ticket": 1,
        "model_name": model_name,
        "dev_f1_target_1": round(dev_metrics["f1_target_1"], 4),
        "heldout_f1_target_1": round(heldout_metrics["f1_target_1"], 4),
        "heldout_accuracy": round(heldout_metrics["accuracy"], 4),
        "fixed_fp": "", "fixed_fn": "", "new_fp": "", "new_fn": "",
        "decision": "match" if matches else "gap",
        "decision_reason": f"heldout_f1={heldout_metrics['f1_target_1']:.4f} vs reference={ref_f1:.4f}, gap={gap:+.4f}, tol={tol}",
    })
    return heldout_preds, heldout_scores


def main():
    df = load_raw()
    train_df, dev_df, heldout_df = get_splits(df)
    contract = json.load(open(REPO_ROOT / "configs" / "project_contract.json"))

    # Variant 1: naive/default TF-IDF+LogReg settings (bigrams, sublinear TF, C=1.0)
    run_variant(train_df, dev_df, heldout_df, "tfidf_logreg_naive_default",
                dict(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True), C=1.0, contract=contract)

    # Variant 2: settings that reproduce the reference within tolerance
    run_variant(train_df, dev_df, heldout_df, "tfidf_logreg_reference_repro",
                REFERENCE_REPRO_TFIDF_KWARGS, C=REFERENCE_REPRO_C, contract=contract)


if __name__ == "__main__":
    main()
