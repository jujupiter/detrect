"""Train a model on trashnet alone, from ImageNet weights. Saves models/<arch>_base.pth and prints test accuracy.

usage: python train.py [--arch resnet50]
"""
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (ARCHS, DEFAULT_ARCH, DEVICE, EVAL_TF, NUM_WORKERS, TRAIN_TF, TRASHNET_SPLITS, TrashDataset,
                    base_checkpoint, build_model, load_checkpoint, read_split_file, save_model)

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-4


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_acc += (outputs.argmax(1) == labels).float().mean().item()
    return total_loss / len(loader), total_acc / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default=DEFAULT_ARCH, choices=ARCHS)
    args = parser.parse_args()
    checkpoint = base_checkpoint(args.arch)

    df = read_split_file(TRASHNET_SPLITS)
    train_loader = DataLoader(TrashDataset(df[df.split == "train"], TRAIN_TF), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(TrashDataset(df[df.split == "val"],   EVAL_TF),  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(TrashDataset(df[df.split == "test"],  EVAL_TF),  batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = build_model(args.arch)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc     = run_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, args.arch, checkpoint)
            print(f"           -> saved to {checkpoint} (val acc {val_acc:.4f})")

    model.load_state_dict(load_checkpoint(checkpoint)[1])
    _, test_acc = run_epoch(model, test_loader, criterion)
    print(f"\nTest accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
