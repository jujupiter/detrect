"""Split trashnet 70/15/15 (stratified by class) into data/splits_trashnet.csv.

The split is committed to git, so every machine trains and tests on the same images: this script keeps it
unless --force is given.
usage: python split.py [--force]
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import TRASHNET_SPLITS

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help=f"overwrite the existing {TRASHNET_SPLITS}")
args = parser.parse_args()
if Path(TRASHNET_SPLITS).exists() and not args.force:
    print(f"Keeping the existing {TRASHNET_SPLITS} (use --force to make a new split)")
    raise SystemExit

root = Path("data/trashnet")
df = pd.DataFrame(
    # sorted: the order glob() lists files in differs between systems, and with it the split
    [(str(p), p.parent.name) for p in sorted(root.glob("*/*.jpg"))],
    columns=["path", "label"],
)
print(df.label.value_counts())

train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=42)
val, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=42)
pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]).to_csv(TRASHNET_SPLITS, index=False)
