"""Shared by every script: classes, image transforms, dataset, model and batch prediction."""
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
BASE_CHECKPOINT = "models/best_model.pth"         # train.py: trashnet only
MIXED_CHECKPOINT = "models/best_model_mixed.pth"  # train_mixed.py: trashnet + RealWaste + TACO

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


def build_model():
    """ResNet-18 with ImageNet weights and a new 6-class head."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(DEVICE)


def load_model(path):
    """Same architecture as build_model(), but no ImageNet weights: the checkpoint overwrites every layer anyway."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


def load_splits():
    """trashnet + RealWaste + TACO, with columns path, label, source, split."""
    trashnet = pd.read_csv(TRASHNET_SPLITS).assign(source="trashnet")
    extra = pd.read_csv(EXTRA_SPLITS)
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
