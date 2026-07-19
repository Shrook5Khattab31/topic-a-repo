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


def get_splits(df: pd.DataFrame = None, split_ids: dict = None):
    """Returns (train_df, dev_df, heldout_df) using the fixed id lists.
    Raises if any id is missing so silent split drift is caught early."""
    df = df if df is not None else load_raw()
    split_ids = split_ids if split_ids is not None else load_split_ids()

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
