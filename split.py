"""Split trashnet 70/15/15 (stratified by class) into data/splits_trashnet.csv."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from common import TRASHNET_SPLITS

root = Path("data/trashnet")
df = pd.DataFrame(
    [(str(p), p.parent.name) for p in root.glob("*/*.jpg")],
    columns=["path", "label"],
)
print(df.label.value_counts())

train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=42)
val, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=42)
pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]).to_csv(TRASHNET_SPLITS, index=False)
