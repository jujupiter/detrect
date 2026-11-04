"""Download RealWaste and TACO, map them to the 6 trashnet classes, and write data/splits_extra.csv.

RealWaste (CC BY 4.0): https://archive.ics.uci.edu/dataset/908/realwaste
TACO (annotations CC BY 4.0, images under free licences): http://tacodataset.org
TACO is a detection dataset: each annotated object is cropped (with some surrounding context) into its own image.
"""
import argparse
import io
import json
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from common import EXTRA_SPLITS

REALWASTE_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
TACO_ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"

REALWASTE_CLASSES = {"Cardboard": "cardboard", "Glass": "glass", "Metal": "metal",
                     "Paper": "paper", "Plastic": "plastic", "Miscellaneous Trash": "trash"}

# TACO category name -> trashnet class, by material. Ambiguous categories (paper cups, drink cartons, ...) are left out.
TACO_CLASSES = {
    "metal": ["Aluminium foil", "Aerosol", "Drink can", "Food Can", "Metal bottle cap", "Metal lid", "Pop tab", "Scrap metal"],
    "glass": ["Glass bottle", "Broken glass", "Glass cup", "Glass jar"],
    "plastic": ["Clear plastic bottle", "Other plastic bottle", "Plastic bottle cap", "Disposable plastic cup", "Other plastic cup",
                "Plastic lid", "Disposable food container", "Other plastic container", "Spread tub", "Tupperware",
                "Plastic utensils", "Plastic straw", "Single-use carrier bag"],
    "paper": ["Normal paper", "Magazine paper", "Wrapping paper", "Paper bag"],
    "cardboard": ["Corrugated carton", "Egg carton", "Meal carton", "Other carton", "Pizza box", "Toilet tube"],
    "trash": ["Cigarette", "Styrofoam piece", "Foam cup", "Foam food container", "Crisp packet", "Other plastic wrapper",
              "Plastic film", "Tissues", "Plastic glooves", "Rope & strings"],
}
TACO_MAP = {name: label for label, names in TACO_CLASSES.items() for name in names}
TACO_MIN_SIDE = 96       # px at full resolution: smaller objects are too blurry to classify
TACO_CONTEXT = 0.25      # margin added around each box, as a fraction of its size
TACO_MAX_PER_CLASS = 600
CROP_MAX_SIDE = 384
SEED = 42


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "detrect/0.1"})
    return urllib.request.urlopen(req, timeout=120).read()


def prepare_realwaste(out, zip_path=None):
    if zip_path is None:
        zip_path = Path("realwaste.zip")
        print("Downloading RealWaste (~690 MB)...")
        zip_path.write_bytes(fetch(REALWASTE_URL))
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            parts = Path(name).parts
            if len(parts) < 2 or parts[-2] not in REALWASTE_CLASSES or not name.endswith(".jpg"):
                continue
            label = REALWASTE_CLASSES[parts[-2]]
            dest = out / label / Path(name).name.replace(" ", "_")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(name))
            rows.append((str(dest), label, "realwaste", ""))
    return pd.DataFrame(rows, columns=["path", "label", "source", "group"])


def taco_crops(image, anns):
    """Square crop around each box, with context, resized so the longest side is CROP_MAX_SIDE."""
    W, H = image.size
    for ann in anns:
        x, y, w, h = ann["bbox"]
        side = max(w, h) * (1 + 2 * TACO_CONTEXT)
        cx, cy = x + w / 2, y + h / 2
        box = (max(0, cx - side / 2), max(0, cy - side / 2), min(W, cx + side / 2), min(H, cy + side / 2))
        crop = image.crop(tuple(int(v) for v in box))
        crop.thumbnail((CROP_MAX_SIDE, CROP_MAX_SIDE))
        yield ann, crop


def prepare_taco(out):
    data = json.loads(fetch(TACO_ANNOTATIONS_URL))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    anns = [a for a in data["annotations"]
            if cats[a["category_id"]] in TACO_MAP and min(a["bbox"][2], a["bbox"][3]) >= TACO_MIN_SIDE]
    # cap each class so cigarettes and plastic film do not dominate
    rng = random.Random(SEED)
    by_label = {}
    for a in anns:
        by_label.setdefault(TACO_MAP[cats[a["category_id"]]], []).append(a)
    kept = [a for group in by_label.values() for a in rng.sample(group, min(len(group), TACO_MAX_PER_CLASS))]
    per_image = {}
    for a in kept:
        per_image.setdefault(a["image_id"], []).append(a)
    images = {im["id"]: im for im in data["images"]}

    def work(image_id):
        im = images[image_id]
        dests = [(ann, out / TACO_MAP[cats[ann["category_id"]]] / f"taco_{ann['id']}.jpg") for ann in per_image[image_id]]
        rows = [(str(d), TACO_MAP[cats[a["category_id"]]], "taco", f"taco_img_{image_id}") for a, d in dests]
        if all(d.exists() for _, d in dests):  # already done by a previous run
            return rows
        try:
            # boxes are drawn on the photo as displayed, so apply its EXIF rotation first
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(fetch(im["flickr_url"])))).convert("RGB")
        except Exception as e:
            print("skip", im["flickr_url"], e)
            return []
        if image.size != (im["width"], im["height"]):
            print("skip", im["flickr_url"], "size", image.size, "does not match annotations", (im["width"], im["height"]))
            return []
        for (ann, crop), (_, dest) in zip(taco_crops(image, per_image[image_id]), dests):
            dest.parent.mkdir(parents=True, exist_ok=True)
            crop.save(dest, quality=90)
        return rows

    print(f"Downloading {len(per_image)} TACO images for {len(kept)} objects...")
    with ThreadPoolExecutor(8) as pool:
        rows = [r for rs in pool.map(work, per_image) for r in rs]
    return pd.DataFrame(rows, columns=["path", "label", "source", "group"])


def split(df):
    """70/15/15, stratified by class. TACO crops from the same photo stay in the same split."""
    df = df.copy()
    df["group"] = df.group.where(df.group != "", df.path)
    groups = df.groupby("group").label.first()
    train, rest = train_test_split(groups.index, test_size=0.30, stratify=groups, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=groups[rest], random_state=SEED)
    df["split"] = df.group.map({**{g: "train" for g in train}, **{g: "val" for g in val}, **{g: "test" for g in test}})
    return df.drop(columns="group")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--realwaste-zip", type=Path, help="use an already downloaded realwaste.zip")
    args = parser.parse_args()
    realwaste = split(prepare_realwaste(Path("data/realwaste"), args.realwaste_zip))
    taco = split(prepare_taco(Path("data/taco")))
    df = pd.concat([realwaste, taco])
    df.to_csv(EXTRA_SPLITS, index=False)
    print(df.groupby(["source", "label"]).size().unstack(0))
    print(df.groupby(["source", "split"]).size().unstack(0))


if __name__ == "__main__":
    main()
