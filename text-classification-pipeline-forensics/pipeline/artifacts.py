"""
Export helpers that match the required artifact schemas from the starter README:

heldout_predictions.csv: id,y_true,y_pred,score,model_name,ticket
results/summary.csv:     ticket,model_name,dev_f1_target_1,heldout_f1_target_1,
                         heldout_accuracy,fixed_fp,fixed_fn,new_fp,new_fn,decision,decision_reason
results/threshold_sweep.csv: ticket,threshold,precision_target_1,recall_target_1,f1_target_1
results/data_quality_audit.csv: id,issue_type,evidence,disposition,confidence
"""
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "predictions"
RESULTS_DIR = REPO_ROOT / "results"
PRED_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

PREDICTION_FIELDS = ["id", "y_true", "y_pred", "score", "model_name", "ticket"]
SUMMARY_FIELDS = ["ticket", "model_name", "dev_f1_target_1", "heldout_f1_target_1",
                  "heldout_accuracy", "fixed_fp", "fixed_fn", "new_fp", "new_fn",
                  "decision", "decision_reason"]
THRESHOLD_FIELDS = ["ticket", "threshold", "precision_target_1", "recall_target_1", "f1_target_1"]
AUDIT_FIELDS = ["id", "issue_type", "evidence", "disposition", "confidence"]


def append_predictions(rows, filename="heldout_predictions.csv"):
    """rows: list of dicts with keys id,y_true,y_pred,score,model_name,ticket"""
    path = PRED_DIR / filename
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PREDICTION_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def append_summary(row, filename="summary.csv"):
    """row: dict with keys ticket,model_name,dev_f1_target_1,heldout_f1_target_1,
    heldout_accuracy,fixed_fp,fixed_fn,new_fp,new_fn,decision,decision_reason"""
    path = RESULTS_DIR / filename
    fields = SUMMARY_FIELDS
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


def append_threshold_sweep(rows, filename="threshold_sweep.csv"):
    path = RESULTS_DIR / filename
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=THRESHOLD_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def append_data_quality_audit(rows, filename="data_quality_audit.csv"):
    path = RESULTS_DIR / filename
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def compare_error_sets(baseline_wrong_ids: set, new_wrong_ids: set):
    """Returns (fixed_ids, new_error_ids) between two prediction runs on the same eval set."""
    fixed = baseline_wrong_ids - new_wrong_ids
    new_errors = new_wrong_ids - baseline_wrong_ids
    return fixed, new_errors
