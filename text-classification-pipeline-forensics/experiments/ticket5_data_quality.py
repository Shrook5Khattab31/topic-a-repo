"""
Ticket 5: Data-quality and error ticket.

Detects:
  1. Exact text duplicates with conflicting labels (likely mislabels) -- TRAIN split only.
  2. Near-duplicate templated tweets (same normalized text) with label disagreement.
  3. Hard negatives on heldout: high-confidence wrong predictions from Ticket 1's
     reference-repro model, inspected for ambiguity vs genuine model error.

Does NOT edit or remove any heldout labels/examples. Training-label corrections are
proposed only, with original and proposed label both kept in the audit table.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data import load_raw, get_splits
from pipeline.model import build_texts, fit_tfidf_logreg, predict_scores, scores_to_preds, REFERENCE_REPRO_TFIDF_KWARGS, REFERENCE_REPRO_C
from pipeline.preprocess import apply_config
from pipeline.artifacts import append_data_quality_audit


def find_duplicate_conflicts(df, text_col="text"):
    """Exact-text duplicates where target disagrees across rows."""
    grouped = df.groupby(text_col)["target"].agg(["nunique", "count", list, "mean"])
    conflicted = grouped[(grouped["nunique"] > 1) & (grouped["count"] >= 2)]
    return conflicted


def find_near_duplicate_conflicts(df):
    """Normalized-text duplicates (URLs/mentions/case stripped) with label disagreement.
    Catches templated tweets that differ only in a URL or mention."""
    norm = df["text"].apply(lambda t: apply_config(t, "aggressive"))
    tmp = df.assign(_norm=norm)
    grouped = tmp.groupby("_norm")["target"].agg(["nunique", "count"])
    conflicted_norms = set(grouped[(grouped["nunique"] > 1) & (grouped["count"] >= 2)].index)
    return tmp[tmp["_norm"].isin(conflicted_norms)].sort_values("_norm")


def main():
    df = load_raw()
    train_df, dev_df, heldout_df = get_splits(df)

    print("=== Exact-text duplicate conflicts (TRAIN split) ===")
    conflicts = find_duplicate_conflicts(train_df)
    print(f"{len(conflicts)} distinct texts with disagreeing labels in train")
    print(conflicts.head(10))

    audit_rows = []
    for text, row in conflicts.iterrows():
        matching_ids = train_df[train_df["text"] == text]["id"].tolist()
        majority_label = 1 if row["mean"] >= 0.5 else 0
        for _id in matching_ids:
            orig_label = train_df.loc[train_df["id"] == _id, "target"].values[0]
            disposition = "keep_but_flag" if orig_label == majority_label else "fix"
            audit_rows.append({
                "id": int(_id),
                "issue_type": "exact_duplicate_label_conflict",
                "evidence": f"text repeated {int(row['count'])}x in train with labels {row['list']}; "
                            f"majority_label={majority_label}, this_row_label={orig_label} (proposed_label={majority_label})",
                "disposition": disposition,
                "confidence": "medium",
            })

    print("\n=== Near-duplicate (normalized-text) conflicts across ALL splits, reported only ===")
    near_dupes = find_near_duplicate_conflicts(df)
    print(f"{near_dupes['_norm'].nunique()} distinct normalized-text groups with label disagreement, "
          f"{len(near_dupes)} rows total")
    for _, row in near_dupes.head(6).iterrows():
        split = "train" if row["id"] in set(train_df["id"]) else ("dev" if row["id"] in set(dev_df["id"]) else "heldout")
        audit_rows.append({
            "id": int(row["id"]),
            "issue_type": "near_duplicate_label_conflict",
            "evidence": f"normalized_text={row['_norm'][:80]!r} label={row['target']} split={split}",
            "disposition": "ambiguous" if split in {"dev", "heldout"} else "keep_but_flag",
            "confidence": "low",
        })

    # --- Hard negatives: confident wrong predictions on heldout (report only, never edit heldout) ---
    print("\n=== Hard negatives / high-confidence heldout errors (reported only, NOT edited) ===")
    train_texts = build_texts(train_df, "baseline")
    heldout_texts = build_texts(heldout_df, "baseline")
    vec, clf = fit_tfidf_logreg(train_texts, train_df["target"].values, seed=3102,
                                 C=REFERENCE_REPRO_C, **REFERENCE_REPRO_TFIDF_KWARGS)
    scores = predict_scores(vec, clf, heldout_texts)
    preds = scores_to_preds(scores, 0.5)

    wrong_mask = preds != heldout_df["target"].values
    confidence = abs(scores - 0.5)  # distance from decision boundary as a confidence proxy
    hard_neg_df = heldout_df[wrong_mask].copy()
    hard_neg_df["score"] = scores[wrong_mask]
    hard_neg_df["confidence"] = confidence[wrong_mask]
    hard_neg_df = hard_neg_df.sort_values("confidence", ascending=False)

    print(f"{len(hard_neg_df)} total heldout errors; top 8 most confident (model was very sure and still wrong):")
    for _, row in hard_neg_df.head(8).iterrows():
        print(f"  id={row['id']} true={row['target']} score={row['score']:.3f} text={row['text'][:90]!r}")
        audit_rows.append({
            "id": int(row["id"]),
            "issue_type": "hard_negative_confident_error",
            "evidence": f"model_score={row['score']:.3f}, true_label={row['target']}, text={row['text'][:100]!r}",
            "disposition": "ambiguous",  # heldout: report only, never propose a fix or removal
            "confidence": "high" if row["confidence"] > 0.3 else "medium",
        })

    append_data_quality_audit(audit_rows)
    print(f"\nWrote {len(audit_rows)} rows to results/data_quality_audit.csv")


if __name__ == "__main__":
    main()
