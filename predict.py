"""Classify images with a trained checkpoint and print the top 3 classes.

usage: python predict.py [--model models/best_model.pth] IMAGE [IMAGE ...]
"""
import argparse

import torch
from PIL import Image

from common import CLASSES, DEVICE, EVAL_TF, MIXED_CHECKPOINT, load_model

TOP_K = 3


def predict(model, path):
    image = EVAL_TF(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = model(image).softmax(1)[0]
    top = probs.topk(TOP_K)
    return [(CLASSES[i], p.item()) for p, i in zip(top.values, top.indices)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MIXED_CHECKPOINT)
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    model = load_model(args.model)
    for path in args.images:
        print(path)
        for label, prob in predict(model, path):
            print(f"  {label:<10} {prob:6.1%}")


if __name__ == "__main__":
    main()
