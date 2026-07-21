"""
Data loading utilities for Topic A: Text Classification Pipeline Forensics.

Loads the full labeled train.csv, applies the FIXED split from split_indices.json,
and exposes train / dev / heldout dataframes. Does not regenerate the split.
"""
import json
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def load_raw(csv_path: Path = None) -> pd.DataFrame:
    csv_path = csv_path or (DATA_DIR / "train.csv")
    df = pd.read_csv(csv_path)
    # Kaggle csv sometimes has NaN keyword/location; normalize to empty string
    df["keyword"] = df["keyword"].fillna("")
    df["location"] = df["location"].fillna("")
    return df


def load_split_ids(split_path: Path = None) -> dict:
    split_path = split_path or (DATA_DIR / "split_indices.json")
    with open(split_path) as f:
        return json.load(f)



def validate_split_ids(df: pd.DataFrame, split_ids: dict) -> None:
    """Validate the fixed split before any model uses it.

    The assignment depends on train/dev/heldout being disjoint and covering the
    provided Kaggle train.csv exactly once. Failing fast here prevents silent
    leakage or accidental split drift.
    """
    required = ["train_ids", "dev_ids", "heldout_ids"]
    missing_keys = [k for k in required if k not in split_ids]
    if missing_keys:
        raise ValueError(f"split_indices.json missing keys: {missing_keys}")

    split_sets = {k: set(split_ids[k]) for k in required}
    if any(len(split_sets[k]) != len(split_ids[k]) for k in required):
        raise ValueError("duplicate ids found inside one split list")

    overlaps = {
        "train/dev": split_sets["train_ids"] & split_sets["dev_ids"],
        "train/heldout": split_sets["train_ids"] & split_sets["heldout_ids"],
        "dev/heldout": split_sets["dev_ids"] & split_sets["heldout_ids"],
    }
    bad = {name: ids for name, ids in overlaps.items() if ids}
    if bad:
        examples = {name: sorted(ids)[:5] for name, ids in bad.items()}
        raise ValueError(f"split overlap detected: {examples}")

    csv_ids = set(df["id"])
    union_ids = split_sets["train_ids"] | split_sets["dev_ids"] | split_sets["heldout_ids"]
    if union_ids != csv_ids:
        missing = sorted(union_ids - csv_ids)[:5]
        extra = sorted(csv_ids - union_ids)[:5]
        raise ValueError(f"split ids do not match train.csv exactly; missing={missing}, extra={extra}")

def get_splits(df: pd.DataFrame = None, split_ids: dict = None):
    """Returns (train_df, dev_df, heldout_df) using the fixed id lists.
    Raises if any id is missing so silent split drift is caught early."""
    df = df if df is not None else load_raw()
    split_ids = split_ids if split_ids is not None else load_split_ids()

    validate_split_ids(df, split_ids)

    id_to_row = df.set_index("id")

    def subset(ids):
        missing = set(ids) - set(id_to_row.index)
        if missing:
            raise ValueError(f"{len(missing)} split ids not found in train.csv, e.g. {list(missing)[:5]}")
        return id_to_row.loc[ids].reset_index()

    train_df = subset(split_ids["train_ids"])
    dev_df = subset(split_ids["dev_ids"])
    heldout_df = subset(split_ids["heldout_ids"])
    return train_df, dev_df, heldout_df


if __name__ == "__main__":
    df = load_raw()
    train_df, dev_df, heldout_df = get_splits(df)
    print("train:", train_df.shape, train_df["target"].value_counts().to_dict())
    print("dev:", dev_df.shape, dev_df["target"].value_counts().to_dict())
    print("heldout:", heldout_df.shape, heldout_df["target"].value_counts().to_dict())
