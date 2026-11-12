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
| `common.py` | Shared code: classes, transforms, dataset, models (`ARCHS`), checkpoints, batch prediction, file paths | | |
| `download_data.sh` | Download trashnet | | `data/trashnet/` |
| `split.py` | Split trashnet 70/15/15 | `data/trashnet/` | `data/splits_trashnet.csv` |
| `download_commons.py` | Download the 40 Wikimedia Commons test photos, with fallbacks (see below) | `data/commons_*/sources.csv` | `data/commons_*/` |
| `prepare_extra_data.py` | Download RealWaste and TACO, map them to the 6 classes, split them | | `data/realwaste/`, `data/taco/`, `data/splits_extra.csv` |
| `train.py` | Train a model (`--arch`, ResNet-50 by default) on trashnet only, from ImageNet weights | `data/splits_trashnet.csv` | `models/<arch>_base.pth` |
| `train_mixed.py` | Fine-tune that model on trashnet + RealWaste + TACO, every batch mixing the three | both splits files, `models/<arch>_base.pth` | `models/<arch>_mixed.pth` |
| `benchmark.py` | Score checkpoints on every test set, with a confusion matrix per test set | splits files, `data/commons_*` | `results/benchmark_<model>.csv` |
| `predict.py` | Classify your own images (top 3) | a checkpoint | |
| `collect_commons.py` | Collect candidate photos from Commons for a large test set, by J-code | `data/commons_queries.csv`, `data/jlist.csv` | `data/commons_candidates/` |
| `review_commons.py` | Review those candidates in the browser (class, J-code, studio or outdoor), then export them | `data/commons_candidates/` | `labels.csv`, `data/commons_large/` |

## Data

| Folder | Content | Licence | In git |
|---|---|---|---|
| `data/trashnet/` | [trashnet](https://github.com/garythung/trashnet): ~2,500 items on a white background | see repo | no, `download_data.sh` |
| `data/realwaste/` | [RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste): 3,587 landfill items on concrete | CC BY 4.0 | no, `prepare_extra_data.py` |
| `data/taco/` | Objects cropped from [TACO](http://tacodataset.org) litter photos | annotations CC BY 4.0, images per photo (see `CREDITS.md`) | no, `prepare_extra_data.py` |
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

### J-codes

Phase 2 uses the EU [Joint List of Litter Categories](https://doi.org/10.2760/127473) (JRC, 2021): 183 litter
types, each with a J-code (J175 = metal drinks cans). `data/jlist.csv` is the full list, from the JRC spreadsheet.
`data/taco_jcodes.csv` and `data/extra_jcodes.csv` map the classes of TACO and of four other datasets to it, with the
quality of each match (exact, merge, broader, none).

### Large Commons test set

The 20-photo Commons sets are too small to compare models (one photo = 5 points). A larger one is built in two steps:

```bash
python collect_commons.py      # ~2,000 candidates from Commons categories listed per J-code, freely licensed only
python review_commons.py       # check each photo in the browser, ~3 s per photo; stop and resume any time
python review_commons.py --export   # accepted photos -> data/commons_large/sources.csv
```

## Licence and credits

The code is under the MIT licence (`LICENSE`). The datasets keep their own licences: `CREDITS.md` lists them, with
the credit to give wherever the data or a model trained on it is published.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins PyTorch 2.2.2 and NumPy 1.x on Intel Macs only, because that is the last PyTorch
built for them. Anywhere else it accepts any recent version and keeps one that is already installed. For
a specific CUDA build, use the official installer: https://pytorch.org/get-started/locally/

On Windows, activate with `.venv\Scripts\activate` instead.

### On Kaggle

Create a notebook, set Accelerator to a GPU and turn Internet on. PyTorch and the other packages are
already installed. Then run:

```bash
%cd /kaggle/working
!git -C detrect pull || git clone -b init https://github.com/jujupiter/detrect.git
%cd detrect
!bash download_data.sh && python prepare_extra_data.py && python download_commons.py
!python train.py && python train_mixed.py
!python benchmark.py models/resnet50_base.pth models/resnet50_mixed.pth
```

Committed runs ("Save & Run All") publish everything in `/kaggle/working` as output, datasets included.
Keep the notebook private, or delete `data/` at the end of the run before making it public.

## Usage

```bash
# Data
bash download_data.sh
python split.py
python prepare_extra_data.py          # ~700 MB RealWaste zip + ~1,000 TACO photos
python download_commons.py            # 40 test photos, 8 MB

# Training
python train.py                       # trashnet only -> models/resnet50_base.pth
python train_mixed.py                 # + RealWaste and TACO -> models/resnet50_mixed.pth
python train.py --arch resnet18       # any model from ARCHS in common.py, same for train_mixed.py

# Evaluation
python benchmark.py models/resnet50_base.pth models/resnet50_mixed.pth
python benchmark.py results/benchmark_resnet50_mixed.csv   # report again, with confusion matrices, no model run
python predict.py path/to/photo.jpg   # uses models/resnet50_mixed.pth unless --model is given
```

The data scripts only download what is missing, so they are safe to run again. Both splits files are
committed, so every machine trains and tests on the same images: `split.py` and `prepare_extra_data.py`
keep them unless given `--force` / `--resplit`. Images listed in a split that could not be downloaded
(e.g. a TACO photo that was taken off Flickr) are skipped by training and benchmarking, with a message.

Models (`--arch`, all with ImageNet weights from torchvision): `resnet18`, `resnet50` (default),
`efficientnet_b0`, `convnext_tiny`. Checkpoints store their architecture, so `benchmark.py` and `predict.py`
work out the model to build by themselves. The first ResNet-18 checkpoints (`models/best_model*.pth`, saved
before this) are plain weights and load as ResNet-18.

On a CPU-only machine the data loaders run in the main process (`NUM_WORKERS` in `common.py`):
worker processes there compete with training for the same cores, and on macOS they have hung the process.

## System requirements

- Python 3.11+
- `curl` and `unzip` (for the download script)
