"""Evaluate one or more checkpoints on every benchmark: the test splits of trashnet, RealWaste and TACO,
plus the Wikimedia Commons photos in data/commons_studio/ and data/commons_outdoor/ (never used for training).

usage: python benchmark.py models/resnet50_base.pth models/resnet50_mixed.pth
Writes results/benchmark_<checkpoint>.csv with one prediction per image.
"""
import sys
from pathlib import Path

import pandas as pd

from common import load_model, load_splits, predict_all

COMMONS = {"data/commons_studio": "commons studio", "data/commons_outdoor": "commons outdoor"}


def benchmarks():
    tests = load_splits().query("split == 'test'").assign(benchmark=lambda d: d.source + " test")
    commons = [pd.read_csv(f"{folder}/sources.csv").assign(benchmark=name) for folder, name in COMMONS.items()]
    df = pd.concat([d[["path", "label", "benchmark"]] for d in [tests] + commons], ignore_index=True)
    present = df.path.map(lambda p: Path(p).exists())
    if not present.all():  # e.g. Commons photos that could not be downloaded (see download_commons.py)
        print("Skipping missing images:", df[~present].groupby("benchmark").size().to_dict())
    return df[present].reset_index(drop=True)


def main():
    df = benchmarks()
    Path("results").mkdir(exist_ok=True)
    summary = {}
    pd.set_option("display.width", 200)
    for ckpt in sys.argv[1:]:
        res = predict_all(load_model(ckpt), df)
        res.to_csv(f"results/benchmark_{Path(ckpt).stem}.csv", index=False)
        summary[Path(ckpt).name] = res.groupby("benchmark", sort=False).ok.mean()
        print(f"\n{ckpt}: accuracy per class")
        print(res.pivot_table(index="label", columns="benchmark", values="ok", aggfunc="mean").round(2))
    print("\nAccuracy per benchmark (n = " + ", ".join(f"{b} {n}" for b, n in df.benchmark.value_counts(sort=False).items()) + ")")
    print(pd.DataFrame(summary).round(3))


if __name__ == "__main__":
    main()
