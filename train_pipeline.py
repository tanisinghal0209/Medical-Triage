from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

from chest_xray_model import Trainer, ModelConfig


PROJECT_ROOT = Path(__file__).resolve().parent
NIHS_ROOT = Path(os.environ.get("NIHS_ROOT", str(PROJECT_ROOT / "NIHS")))
IMAGE_DIR = NIHS_ROOT / "images"
DATA_ENTRY_CLEAN = NIHS_ROOT / "Data_Entry_2017.cleaned.csv"
DATA_ENTRY_RAW = NIHS_ROOT / "Data_Entry_2017.csv"
TRAIN_VAL_LIST = NIHS_ROOT / "train_val_list.cleaned.txt"
TRAIN_VAL_RAW = NIHS_ROOT / "train_val_list.txt"
TEST_LIST = NIHS_ROOT / "test_list.cleaned.txt"
TEST_RAW = NIHS_ROOT / "test_list.txt"


def resolve_path(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


def load_name_list(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


class NIHDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        csv_file: Path,
        transform=None,
        allowed_names: Optional[Set[str]] = None,
    ):
        self.image_dir = Path(image_dir)
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.classes = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia",
        ]

        self.df["Finding Labels"] = self.df["Finding Labels"].fillna("No Finding").astype(str)
        self.df = self.df.dropna(subset=["Finding Labels", "Image Index"]).reset_index(drop=True)

        self.image_map = {p.name: p for p in self.image_dir.glob("*.png") if p.is_file()}
        initial_count = len(self.df)
        self.df = self.df[self.df["Image Index"].isin(self.image_map.keys())].reset_index(drop=True)

        if allowed_names is not None:
            allowed_names = set(allowed_names)
            self.df = self.df[self.df["Image Index"].isin(allowed_names)].reset_index(drop=True)

        print(
            f"Sanitized dataset: kept {len(self.df)} rows "
            f"(dropped {initial_count - len(self.df)} missing/unlisted)."
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = self.image_map.get(img_name)
        if img_path is None:
            raise FileNotFoundError(f"Missing image: {img_name}")

        image = Image.open(img_path).convert("RGB")
        label_str = row["Finding Labels"]
        labels = torch.zeros(len(self.classes), dtype=torch.float32)
        for i, cls in enumerate(self.classes):
            if cls in label_str:
                labels[i] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, labels, img_name


def start_training():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    data_entry = resolve_path(DATA_ENTRY_CLEAN, DATA_ENTRY_RAW)
    train_val_list = resolve_path(TRAIN_VAL_LIST, TRAIN_VAL_RAW)
    test_list = resolve_path(TEST_LIST, TEST_RAW)

    cfg = ModelConfig()
    cfg.NUM_CLASSES = 14
    cfg.EPOCHS = 20
    cfg.LR = 1e-4
    cfg.BATCH_SIZE = 8
    cfg.FREEZE_BACKBONE = True
    cfg.OUTPUT_ROOT = str(PROJECT_ROOT / "chest_xray_pipeline")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_val_names = load_name_list(train_val_list)
    test_names = load_name_list(test_list)

    if not train_val_names:
        raise FileNotFoundError(f"Train/val list not found or empty: {train_val_list}")

    train_val_dataset = NIHDataset(
        IMAGE_DIR,
        data_entry,
        transform=transform,
        allowed_names=train_val_names,
    )
    test_dataset = NIHDataset(
        IMAGE_DIR,
        data_entry,
        transform=transform,
        allowed_names=test_names if test_names else None,
    )

    train_size = int(0.9 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    generator = torch.Generator().manual_seed(cfg.SEED)
    train_ds, val_ds = random_split(train_val_dataset, [train_size, val_size], generator=generator)

    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        ),
    }

    trainer = Trainer(loaders, cfg=cfg)

    checkpoint_path = Path(cfg.OUTPUT_ROOT) / "checkpoints" / "last_model.pth"
    start_epoch = 1
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "history" in checkpoint:
            trainer.history = checkpoint["history"]
        if "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resuming at epoch {start_epoch}")

    trainer.start_epoch = start_epoch
    print("\nSTARTING TRAINING")
    print(f"NIHS root: {NIHS_ROOT}")
    print(f"Device   : {trainer.device}")
    print(f"Batch    : {cfg.BATCH_SIZE}")
    print(f"Train/Val: {len(train_val_dataset)} images total")
    print(f"Test     : {len(test_dataset)} images total")
    trainer.fit()


if __name__ == "__main__":
    start_training()
