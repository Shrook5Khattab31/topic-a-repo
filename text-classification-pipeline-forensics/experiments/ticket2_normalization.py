"""
Ticket 2: Text normalization lever.

Compares normalization configs (URL/mention/hashtag/casing/punct/emoji decisions)
on the DEV split only. Freezes the best dev config, then evaluates once on heldout
and reports which specific false positives/negatives were fixed vs newly introduced
relative to the reference reproduction baseline from Ticket 1.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data import load_raw, get_splits
from pipeline.model import (build_texts, fit_tfidf_logreg, predict_scores, scores_to_preds,
                             evaluate, REFERENCE_REPRO_TFIDF_KWARGS, REFERENCE_REPRO_C)
from pipeline.artifacts import append_predictions, append_summary, compare_error_sets
from pipeline.preprocess import CONFIGS

REPO_ROOT = Path(__file__).resolve().parent.parent


def eval_config(train_df, eval_df, config_name):
    train_texts = build_texts(train_df, config_name)
    eval_texts = build_texts(eval_df, config_name)
    vec, clf = fit_tfidf_logreg(train_texts, train_df["target"].values, seed=3102,
                                 C=REFERENCE_REPRO_C, **REFERENCE_REPRO_TFIDF_KWARGS)
    scores = predict_scores(vec, clf, eval_texts)
    preds = scores_to_preds(scores, 0.5)
    metrics = evaluate(eval_df["target"].values, preds)
    return metrics, preds, scores


def main():
    df = load_raw()
    train_df, dev_df, heldout_df = get_splits(df)
    contract = json.load(open(REPO_ROOT / "configs" / "project_contract.json"))

    print("=== Dev-split comparison across normalization configs ===")
    dev_results = {}
    for name in CONFIGS:
        metrics, preds, scores = eval_config(train_df, dev_df, name)
        dev_results[name] = (metrics, preds, scores)
        print(f"{name:20s} dev_f1={metrics['f1_target_1']:.4f}  "
              f"P={metrics['precision_target_1']:.4f} R={metrics['recall_target_1']:.4f}")

    best_name = max(dev_results, key=lambda n: dev_results[n][0]["f1_target_1"])
    print(f"\nBest dev config: {best_name} (dev_f1={dev_results[best_name][0]['f1_target_1']:.4f})")
    print("Decision frozen. Evaluating on heldout now.")

    # Baseline for error comparison = Ticket 1's reference-repro config ("baseline" normalization)
    base_metrics, base_preds, base_scores = eval_config(train_df, heldout_df, "baseline")
    base_wrong = set(heldout_df["id"][base_preds != heldout_df["target"].values])

    best_metrics, best_preds, best_scores = eval_config(train_df, heldout_df, best_name)
    best_wrong = set(heldout_df["id"][best_preds != heldout_df["target"].values])

    fixed_ids, new_error_ids = compare_error_sets(base_wrong, best_wrong)

    # Split fixed/new errors into FP vs FN using the baseline/best prediction direction
    heldout_lookup = heldout_df.set_index("id")
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

    base_preds_by_id = dict(zip(heldout_df["id"], base_preds))
    best_preds_by_id = dict(zip(heldout_df["id"], best_preds))

    fixed_fp, fixed_fn = classify_errors(fixed_ids, base_preds_by_id)  # errors baseline made, that best config no longer makes
    new_fp, new_fn = classify_errors(new_error_ids, best_preds_by_id)  # new errors introduced by best config

    print(f"\nHeldout: baseline(normalize) F1={base_metrics['f1_target_1']:.4f}  "
          f"{best_name} F1={best_metrics['f1_target_1']:.4f}")
    print(f"Fixed errors: {len(fixed_ids)}  (fp fixed={fixed_fp}, fn fixed={fixed_fn})")
    print(f"New errors:   {len(new_error_ids)}  (new fp={new_fp}, new fn={new_fn})")

    # Show a few concrete examples for the report
    print("\nExample fixed errors (id, text, true, old_pred, new_pred):")
    for _id in list(fixed_ids)[:5]:
        row = heldout_lookup.loc[_id]
        print(f"  id={_id} true={row['target']} old_pred={base_preds_by_id[_id]} "
              f"new_pred={best_preds_by_id[_id]} text={row['text'][:80]!r}")

    print("\nExample new errors (id, text, true, old_pred, new_pred):")
    for _id in list(new_error_ids)[:5]:
        row = heldout_lookup.loc[_id]
        print(f"  id={_id} true={row['target']} old_pred={base_preds_by_id[_id]} "
              f"new_pred={best_preds_by_id[_id]} text={row['text'][:80]!r}")

    rows = []
    for _id, yt, yp, sc in zip(heldout_df["id"], heldout_df["target"], best_preds, best_scores):
        rows.append({"id": int(_id), "y_true": int(yt), "y_pred": int(yp), "score": float(sc),
                     "model_name": f"tfidf_logreg_{best_name}", "ticket": 2})
    append_predictions(rows)

    ref_f1 = contract["reference_baseline_f1"]
    append_summary({
        "ticket": 2,
        "model_name": f"tfidf_logreg_{best_name}",
        "dev_f1_target_1": round(dev_results[best_name][0]["f1_target_1"], 4),
        "heldout_f1_target_1": round(best_metrics["f1_target_1"], 4),
        "heldout_accuracy": round(best_metrics["accuracy"], 4),
        "fixed_fp": fixed_fp, "fixed_fn": fixed_fn, "new_fp": new_fp, "new_fn": new_fn,
        "decision": "adopt" if best_metrics["f1_target_1"] >= base_metrics["f1_target_1"] else "reject",
        "decision_reason": f"best dev config={best_name}; heldout f1 {base_metrics['f1_target_1']:.4f} -> {best_metrics['f1_target_1']:.4f}",
    })


if __name__ == "__main__":
    main()
