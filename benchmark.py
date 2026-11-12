"""Evaluate one or more checkpoints on every benchmark: the test splits of trashnet, RealWaste and TACO,
plus the Wikimedia Commons photos in data/commons_studio/ and data/commons_outdoor/ (never used for training).

usage: python benchmark.py models/resnet50_base.pth models/resnet50_mixed.pth
Writes results/benchmark_<checkpoint>.csv with one prediction per image, then prints the accuracy per class
and a confusion matrix per benchmark: rows are the true class, columns the predicted class.

A results CSV can be given instead of a checkpoint, to print its report again without running the model:
python benchmark.py results/benchmark_resnet50_mixed.csv
"""
import sys
from pathlib import Path

import pandas as pd

from common import CLASSES, load_model, load_splits, predict_all

COMMONS = {"data/commons_studio": "commons studio", "data/commons_outdoor": "commons outdoor"}


def benchmarks():
    tests = load_splits().query("split == 'test'").assign(benchmark=lambda d: d.source + " test")
    commons = [pd.read_csv(f"{folder}/sources.csv").assign(benchmark=name) for folder, name in COMMONS.items()]
    df = pd.concat([d[["path", "label", "benchmark"]] for d in [tests] + commons], ignore_index=True)
    present = df.path.map(lambda p: Path(p).exists())
    if not present.all():  # e.g. Commons photos that could not be downloaded (see download_commons.py)
        print("Skipping missing images:", df[~present].groupby("benchmark").size().to_dict())
    return df[present].reset_index(drop=True)


def confusion(res):
    """One confusion matrix per benchmark: how many images of each true class (rows) got each prediction (columns)."""
    for name, d in res.groupby("benchmark", sort=False):
        m = pd.crosstab(pd.Categorical(d.label, CLASSES), pd.Categorical(d.pred, CLASSES), dropna=False)
        m.index.name, m.columns.name = "true", "predicted"
        print(f"\n{name} (n = {len(d)}, accuracy {d.ok.mean():.3f})")
        print(m)


def main():
    Path("results").mkdir(exist_ok=True)
    summary, counts = {}, None
    pd.set_option("display.width", 200)
    for arg in sys.argv[1:]:
        if arg.endswith(".csv"):
            res = pd.read_csv(arg)
        else:
            res = predict_all(load_model(arg), benchmarks())
            res.to_csv(f"results/benchmark_{Path(arg).stem}.csv", index=False)
        summary[Path(arg).name] = res.groupby("benchmark", sort=False).ok.mean()
        counts = res.benchmark.value_counts(sort=False)
        print(f"\n{arg}: accuracy per class")
        print(res.pivot_table(index="label", columns="benchmark", values="ok", aggfunc="mean", sort=False).reindex(CLASSES).round(2).to_string())
        confusion(res)
    print("\nAccuracy per benchmark (n = " + ", ".join(f"{b} {n}" for b, n in counts.items()) + ")")
    print(pd.DataFrame(summary).round(3).to_string())


if __name__ == "__main__":
    main()
