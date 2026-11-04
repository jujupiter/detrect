# detrect

Garbage image classification project, starting from the trashnet dataset.

## Goal

Build a trash detection and classification system, developed in phases:

1. **Single-item classification** *(current)* — one piece of trash per image, 6 coarse categories (trashnet)
2. **Richer categories** — finer-grained labels beyond the 6 trashnet classes
3. **Multi-object detection** — multiple pieces of trash per image with bounding boxes

## What each file does

| File | Role | Reads | Writes |
|---|---|---|---|
| `common.py` | Shared code: classes, transforms, dataset, model, batch prediction, file paths | | |
| `download_data.sh` | Download trashnet | | `data/trashnet/` |
| `split.py` | Split trashnet 70/15/15 | `data/trashnet/` | `data/splits_trashnet.csv` |
| `download_commons.py` | Download the 40 Wikimedia Commons test photos, with fallbacks (see below) | `data/commons_*/sources.csv` | `data/commons_*/` |
| `prepare_extra_data.py` | Download RealWaste and TACO, map them to the 6 classes, split them | | `data/realwaste/`, `data/taco/`, `data/splits_extra.csv` |
| `train.py` | Train ResNet-18 on trashnet only, from ImageNet weights | `data/splits_trashnet.csv` | `models/best_model.pth` |
| `train_mixed.py` | Fine-tune that model on trashnet + RealWaste + TACO, every batch mixing the three | both splits files, `models/best_model.pth` | `models/best_model_mixed.pth` |
| `benchmark.py` | Score checkpoints on every test set | splits files, `data/commons_*` | `results/benchmark_<model>.csv` |
| `predict.py` | Classify your own images (top 3) | a checkpoint | |

## Data

| Folder | Content | Licence | In git |
|---|---|---|---|
| `data/trashnet/` | [trashnet](https://github.com/garythung/trashnet): ~2,500 items on a white background | see repo | no, `download_data.sh` |
| `data/realwaste/` | [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste): 3,587 landfill items on concrete | CC BY 4.0 | no, `prepare_extra_data.py` |
| `data/taco/` | Objects cropped from [TACO](http://tacodataset.org) litter photos | CC BY 4.0 | no, `prepare_extra_data.py` |
| `data/commons_studio/` | 20 product-style photos from Wikimedia Commons | per photo, in `sources.csv` | `sources.csv` only, `download_commons.py` |
| `data/commons_outdoor/` | 20 outdoor photos from Wikimedia Commons | per photo, in `sources.csv` | `sources.csv` only, `download_commons.py` |

Every source has its own train/val/test split. TACO crops from the same photo stay in the same split.
The Commons sets are only used for testing, never for training.

The Commons photos are not stored in git. Their `sources.csv` records each photo's title, licence, author
(to credit them), the thumbnail used and checksums. `download_commons.py` tries the recorded thumbnail URL,
then the Commons API (which follows renames and finds the recorded version if the file was re-uploaded),
then the Internet Archive. Photos it cannot get are reported and skipped by `benchmark.py`. A one-off
`python download_commons.py --archive` saves a copy of every thumbnail to the Internet Archive, so that
the last fallback has something to find.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch must be installed separately — use the official installer for your platform:
https://pytorch.org/get-started/locally/

On Windows, activate with `.venv\Scripts\activate` instead.

## Usage

```bash
# Data
bash download_data.sh
python split.py
python prepare_extra_data.py          # ~700 MB RealWaste zip + ~1,000 TACO photos
python download_commons.py            # 40 test photos, 8 MB

# Training
python train.py                       # trashnet only -> models/best_model.pth
python train_mixed.py                 # + RealWaste and TACO -> models/best_model_mixed.pth

# Evaluation
python benchmark.py models/best_model.pth models/best_model_mixed.pth
python predict.py path/to/photo.jpg   # uses models/best_model_mixed.pth unless --model is given
```

On a CPU-only machine the data loaders run in the main process (`NUM_WORKERS` in `common.py`):
worker processes there compete with training for the same cores, and on macOS they have hung the process.

## System requirements

- Python 3.11+
- `curl` and `unzip` (for the download script)
