"""Classify the benchmark photos with no training at all: a vision-language model (CLIP, SigLIP) picks the class whose
description best matches each photo. A reference point for #5: how far general knowledge alone goes.

The descriptions below follow our labelling conventions (README) and were written before any result was seen:
tuning them on the test photos would turn the test sets into training data.

usage: python zero_shot.py [--model google/siglip-base-patch16-224]
       python zero_shot.py --model openai/clip-vit-base-patch32
Writes results/benchmark_zeroshot_<model>.csv in benchmark.py's format: `python benchmark.py <that csv>` prints the
report. The run's energy goes to results/runs.csv.
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

from benchmark import benchmarks
from common import CLASSES, DEVICE, measured_run

DESCRIPTIONS = {
    "cardboard": "a photo of cardboard, such as a cardboard box, an egg carton or a pizza box",
    "glass": "a photo of glass, such as a glass bottle, a jar or a piece of broken glass",
    "metal": "a photo of metal, such as a drink can, a food tin, a bottle cap or aluminium foil",
    "paper": "a photo of paper, such as a paper bag, a paper cup, a newspaper or a paper wrapper",
    "plastic": "a photo of plastic, such as a plastic bottle, a plastic bag, a plastic cup, a straw or a face mask",
    "trash": "a photo of non-recyclable trash, such as a cigarette butt, a snack wrapper, polystyrene foam or a wet wipe",
}
BATCH = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/siglip-base-patch16-224", help="a CLIP or SigLIP model on Hugging Face")
    args = parser.parse_args()
    out = Path("results") / f"benchmark_zeroshot_{args.model.replace('/', '_')}.csv"

    from transformers import AutoModel, AutoProcessor
    df = benchmarks()
    with measured_run("zero_shot.py", args.model):
        model = AutoModel.from_pretrained(args.model, use_safetensors=True).to(DEVICE).eval()
        processor = AutoProcessor.from_pretrained(args.model)
        texts = [DESCRIPTIONS[c] for c in CLASSES]
        preds, confs = [], []
        with torch.no_grad():
            text_in = processor(text=texts, padding="max_length", return_tensors="pt").to(DEVICE)
            text = model.get_text_features(**text_in)
            text = text / text.norm(dim=-1, keepdim=True)
            for start in range(0, len(df), BATCH):
                images = [Image.open(p).convert("RGB") for p in df.path[start:start + BATCH]]
                image = model.get_image_features(**processor(images=images, return_tensors="pt").to(DEVICE))
                image = image / image.norm(dim=-1, keepdim=True)
                probs = (100 * image @ text.T).softmax(1)  # cosine similarity; the scale only affects the confidence
                conf, idx = probs.max(1)
                preds += [CLASSES[i] for i in idx.cpu().tolist()]
                confs += conf.cpu().tolist()
    res = df.assign(pred=preds, conf=confs, ok=lambda d: d.pred == d.label)
    res.to_csv(out, index=False)
    print(f"Written {out}")
    print(res.groupby("benchmark", sort=False).ok.mean().round(3).to_string())


if __name__ == "__main__":
    main()
