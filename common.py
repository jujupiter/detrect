"""Shared by every script: classes, image transforms, dataset, model and batch prediction."""
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
# Loader worker processes only pay off when a GPU does the maths. On a CPU they compete with
# training for the same cores and RAM, and on macOS they have hung the process at shutdown.
NUM_WORKERS = 0 if DEVICE.type == "cpu" else 4

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

TRASHNET_SPLITS = "data/splits_trashnet.csv"
EXTRA_SPLITS = "data/splits_extra.csv"
# Pretrained ImageNet backbones that build_model() can start from. Each gets a new 6-class head.
ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),                 # 11.7M params, 1.8 GFLOPs
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),                 # 25.6M params, 4.1 GFLOPs
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),  # 5.3M params, 0.4 GFLOPs
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),  # 28.6M params, 4.5 GFLOPs
}
DEFAULT_ARCH = "resnet50"


def base_checkpoint(arch):
    """train.py: trashnet only."""
    return f"models/{arch}_base.pth"


def mixed_checkpoint(arch):
    """train_mixed.py: trashnet + RealWaste + TACO."""
    return f"models/{arch}_mixed.pth"


TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class TrashDataset(Dataset):
    """Images listed in a dataframe with `path` and `label` columns."""

    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row.path).convert("RGB")
        return self.transform(image), CLASS_TO_IDX[row.label]


def build_model(arch, pretrained=True):
    """`arch` from ARCHS, with ImageNet weights (unless pretrained=False) and a new 6-class head."""
    constructor, weights = ARCHS[arch]
    model = constructor(weights=weights if pretrained else None)
    if hasattr(model, "fc"):  # ResNets
        model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    else:  # EfficientNet, ConvNeXt: the last layer of `classifier` is the head
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    return model.to(DEVICE)


def save_model(model, arch, path):
    """Save the weights together with the architecture name, so load_model() can rebuild it."""
    Path(path).parent.mkdir(exist_ok=True)
    torch.save({"arch": arch, "state_dict": model.state_dict()}, path)


def load_checkpoint(path):
    """(arch, state_dict). Older checkpoints are a bare ResNet-18 state_dict."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    if "state_dict" in ckpt:
        return ckpt["arch"], ckpt["state_dict"]
    return "resnet18", ckpt


def load_model(path):
    """Rebuild the checkpoint's architecture, without ImageNet weights: the checkpoint overwrites every layer anyway."""
    arch, state_dict = load_checkpoint(path)
    model = build_model(arch, pretrained=False)
    model.load_state_dict(state_dict)
    return model.eval()


def read_split_file(path):
    """A splits CSV, without the images that are not on disk (e.g. TACO photos that failed to download)."""
    df = pd.read_csv(path)
    present = df.path.map(lambda p: Path(p).exists())
    if not present.all():
        print(f"{path}: skipping {(~present).sum()} images that are not on disk")
    return df[present].reset_index(drop=True)


def load_splits():
    """trashnet + RealWaste + TACO, with columns path, label, source, split."""
    trashnet = read_split_file(TRASHNET_SPLITS).assign(source="trashnet")
    extra = read_split_file(EXTRA_SPLITS)
    return pd.concat([trashnet, extra], ignore_index=True)


def predict_all(model, df):
    """Add pred, conf and ok columns to a dataframe of images."""
    model.eval()
    loader = DataLoader(TrashDataset(df, EVAL_TF), batch_size=64, num_workers=NUM_WORKERS)
    preds, confs = [], []
    with torch.no_grad():
        for images, _ in loader:
            conf, idx = model(images.to(DEVICE)).softmax(1).max(1)
            preds += [CLASSES[i] for i in idx.cpu().tolist()]
            confs += conf.cpu().tolist()
    return df.assign(pred=preds, conf=confs, ok=lambda d: d.pred == d.label)
