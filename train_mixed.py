"""Fine-tune the trashnet model on trashnet + RealWaste + TACO, with every batch mixing the three sources.

Run prepare_extra_data.py and train.py first. Starts from models/<arch>_base.pth and saves models/<arch>_mixed.pth.
usage: python train_mixed.py [--arch resnet50] [--epochs 20] [--patience 4]
"""
import argparse
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from common import (ARCHS, DEFAULT_ARCH, DEVICE, NUM_WORKERS, TRAIN_TF, TrashDataset, base_checkpoint, load_model, load_splits,
                    mixed_checkpoint, predict_all, save_model)

# images per source in every batch of 32
BATCH_MIX = {"trashnet": 12, "realwaste": 10, "taco": 10}
LR = 5e-5  # lower than train.py: we start from an already trained model


class MixedBatchSampler(Sampler):
    """Each batch takes a fixed number of images from each source. One epoch = one pass over trashnet."""

    def __init__(self, sources, mix, seed=0):
        self.pools = {s: [i for i, src in enumerate(sources) if src == s] for s in mix}
        self.mix = mix
        self.rng = random.Random(seed)
        self.queues = {s: [] for s in mix}
        self.n_batches = len(self.pools["trashnet"]) // mix["trashnet"]

    def take(self, source, k):
        out = []
        while len(out) < k:
            if not self.queues[source]:  # reshuffle a source once it is used up
                self.queues[source] = self.rng.sample(self.pools[source], len(self.pools[source]))
            out.append(self.queues[source].pop())
        return out

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = [i for s, k in self.mix.items() for i in self.take(s, k)]
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


def evaluate(model, df):
    """Accuracy per source."""
    return predict_all(model, df).groupby("source").ok.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default=DEFAULT_ARCH, choices=ARCHS)
    parser.add_argument("--epochs", type=int, default=20, help="maximum number of epochs")
    parser.add_argument("--patience", type=int, default=4, help="stop after this many epochs without improvement")
    args = parser.parse_args()
    out = mixed_checkpoint(args.arch)

    df = load_splits()
    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    print(train.groupby(["source", "label"]).size().unstack(0), "\n")

    sampler = MixedBatchSampler(train.source.tolist(), BATCH_MIX)
    train_loader = DataLoader(TrashDataset(train, TRAIN_TF), batch_sampler=sampler, num_workers=NUM_WORKERS)

    model = load_model(base_checkpoint(args.arch))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    acc = evaluate(model, val)
    print("Before fine-tuning | val acc " + " ".join(f"{s} {a:.3f}" for s, a in acc.items()), flush=True)
    # select on the mean over sources, so trashnet (the easiest) does not decide alone
    best, best_epoch = acc.mean(), 0
    save_model(model, args.arch, out)

    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total_loss = time.time(), 0.0
        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % 25 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {total_loss / step:.4f} ({time.time() - start:.0f}s)", flush=True)
        acc = evaluate(model, val)
        print(f"Epoch {epoch:02d} | train loss {total_loss / len(train_loader):.4f} | val acc "
              + " ".join(f"{s} {a:.3f}" for s, a in acc.items()) + f" | mean {acc.mean():.3f} | {time.time() - start:.0f}s", flush=True)
        if acc.mean() > best:
            best, best_epoch = acc.mean(), epoch
            save_model(model, args.arch, out)
            print(f"           -> checkpoint saved (mean val acc {best:.4f})", flush=True)
        elif epoch - best_epoch >= args.patience:
            print(f"No improvement for {args.patience} epochs: stopping.")
            break
    print(f"Best: epoch {best_epoch}, mean val acc {best:.4f}, saved to {out}")


if __name__ == "__main__":
    main()
