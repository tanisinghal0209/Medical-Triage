"""
==============================================================================
 CHEST X-RAY PNEUMONIA CLASSIFIER — Training Script
==============================================================================
 Binary classification: NORMAL vs PNEUMONIA
 Supports: MobileNetV2 (fast, recommended) or ResNet50
 Optimized for Apple M4 with MPS acceleration
==============================================================================
"""

import os, sys, time, copy, json, logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, f1_score, precision_score, recall_score,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s — %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ==============================================================================
# CONFIG
# ==============================================================================

class Config:
    # Paths
    DATA_DIR      = "/Users/tanishasinghal/Downloads/chest_xray"
    OUTPUT_DIR    = "./chest_xray_pipeline"

    # Model: "mobilenet" (fast, recommended) or "resnet50"
    BACKBONE      = "mobilenet"
    NUM_CLASSES   = 2
    CLASS_NAMES   = ["NORMAL", "PNEUMONIA"]

    # Training
    EPOCHS        = 15
    BATCH_SIZE    = 32
    LR            = 1e-4
    WEIGHT_DECAY  = 1e-4
    IMAGE_SIZE    = 224
    SEED          = 42
    NUM_WORKERS   = 0      # 0 for macOS compatibility

    # Early stopping
    PATIENCE      = 5

cfg = Config()
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
log.info(f"Device: {DEVICE}")


# ==============================================================================
# DATA LOADING
# ==============================================================================

def get_dataloaders(cfg: Config) -> Dict[str, DataLoader]:
    """Create train/val/test dataloaders with augmentation."""

    train_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    loaders = {}
    for split in ["train", "val", "test"]:
        path = Path(cfg.DATA_DIR) / split
        if not path.exists():
            log.warning(f"Split '{split}' not found at {path}")
            continue
        tfm = train_transform if split == "train" else eval_transform
        ds = datasets.ImageFolder(str(path), transform=tfm)
        loaders[split] = DataLoader(
            ds, batch_size=cfg.BATCH_SIZE, shuffle=(split == "train"),
            num_workers=cfg.NUM_WORKERS, pin_memory=False,
        )
        log.info(f"  {split}: {len(ds)} images | {len(loaders[split])} batches")

    return loaders


# ==============================================================================
# MODEL
# ==============================================================================

def build_model(cfg: Config) -> nn.Module:
    """Build pretrained model with custom classifier head."""

    if cfg.BACKBONE == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, cfg.NUM_CLASSES),
        )
        log.info(f"MobileNetV2 — {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    elif cfg.BACKBONE == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, cfg.NUM_CLASSES),
        )
        log.info(f"ResNet50 — {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    else:
        raise ValueError(f"Unknown backbone: {cfg.BACKBONE}")

    return model.to(DEVICE)


# ==============================================================================
# TRAINING
# ==============================================================================

def train_model(cfg: Config) -> Tuple[nn.Module, Dict]:
    """Full training pipeline with evaluation."""

    log.info("=" * 60)
    log.info(f"  PNEUMONIA CLASSIFIER — {cfg.BACKBONE.upper()}")
    log.info(f"  Device: {DEVICE} | Epochs: {cfg.EPOCHS} | BS: {cfg.BATCH_SIZE}")
    log.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = get_dataloaders(cfg)
    train_loader = loaders["train"]
    val_loader = loaders.get("val", loaders.get("test"))

    # Class weights for imbalanced data (3875 pneumonia vs 1341 normal)
    train_ds = train_loader.dataset
    class_counts = np.bincount([s[1] for s in train_ds.samples])
    weights = 1.0 / class_counts.astype(np.float32)
    weights = weights / weights.sum() * len(weights)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    log.info(f"  Class weights: {dict(zip(cfg.CLASS_NAMES, weights.tolist()))}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_weights = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += images.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += images.size(0)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        log.info(
            f"  Epoch {epoch:2d}/{cfg.EPOCHS} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"LR={lr_now:.1e} | {elapsed:.1f}s"
        )

        # Checkpoint best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            log.info(f"  ★ New best val_acc = {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.PATIENCE:
                log.info(f"  Early stopping at epoch {epoch} (patience={cfg.PATIENCE})")
                break

    # Restore best weights
    if best_weights:
        model.load_state_dict(best_weights)
    model.eval()
    log.info(f"\n  Best validation accuracy: {best_val_acc:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    save_dir = Path(cfg.OUTPUT_DIR) / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "best_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "backbone": cfg.BACKBONE,
        "num_classes": cfg.NUM_CLASSES,
        "class_names": cfg.CLASS_NAMES,
        "best_val_acc": best_val_acc,
        "epoch": epoch,
    }, ckpt_path)
    log.info(f"  Model saved → {ckpt_path}")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    test_metrics = {}
    if "test" in loaders:
        test_metrics = evaluate_model(model, loaders["test"], cfg)

    # ── Plot training curves ──────────────────────────────────────────────────
    plot_training_curves(history, cfg)

    return model, {"history": history, "best_val_acc": best_val_acc, "test": test_metrics}


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_model(model: nn.Module, loader: DataLoader, cfg: Config) -> Dict:
    """Evaluate on test set with full metrics."""
    log.info("\n  Evaluating on test set…")
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # P(PNEUMONIA)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1": float(f1_score(all_labels, all_preds, average="weighted")),
        "precision": float(precision_score(all_labels, all_preds, average="weighted")),
        "recall": float(recall_score(all_labels, all_preds, average="weighted")),
        "auroc": float(roc_auc_score(all_labels, all_probs)),
    }

    log.info(f"\n  ┌─────────── TEST RESULTS ───────────┐")
    log.info(f"  │  Accuracy  : {metrics['accuracy']:.4f}              │")
    log.info(f"  │  F1 Score  : {metrics['f1']:.4f}              │")
    log.info(f"  │  Precision : {metrics['precision']:.4f}              │")
    log.info(f"  │  Recall    : {metrics['recall']:.4f}              │")
    log.info(f"  │  AUC-ROC   : {metrics['auroc']:.4f}              │")
    log.info(f"  └─────────────────────────────────────┘")

    # Classification report
    report = classification_report(all_labels, all_preds,
                                    target_names=cfg.CLASS_NAMES, zero_division=0)
    log.info(f"\n{report}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    log.info(f"  Confusion Matrix:")
    log.info(f"                 Pred NORMAL  Pred PNEUMONIA")
    log.info(f"  True NORMAL    {cm[0][0]:>8}    {cm[0][1]:>8}")
    log.info(f"  True PNEUMONIA {cm[1][0]:>8}    {cm[1][1]:>8}")

    # Save confusion matrix plot
    plot_confusion_matrix(cm, cfg)

    # Save metrics
    out = Path(cfg.OUTPUT_DIR) / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ==============================================================================
# PLOTS
# ==============================================================================

def plot_training_curves(history: Dict, cfg: Config):
    """Plot loss and accuracy curves."""
    out = Path(cfg.OUTPUT_DIR) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val", linewidth=2)
    ax1.set_title("Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "r-", label="Val", linewidth=2)
    ax2.set_title("Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Pneumonia Classifier — {cfg.BACKBONE.upper()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = str(out / "training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Training curves → {path}")


def plot_confusion_matrix(cm: np.ndarray, cfg: Config):
    """Plot confusion matrix."""
    out = Path(cfg.OUTPUT_DIR) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(cfg.CLASS_NAMES, fontsize=11)
    ax.set_yticklabels(cfg.CLASS_NAMES, fontsize=11)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {cfg.BACKBONE.upper()}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = str(out / "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Confusion matrix → {path}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Pneumonia Classifier")
    parser.add_argument("--backbone", choices=["mobilenet", "resnet50"], default="mobilenet",
                        help="Model backbone (default: mobilenet)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-dir", type=str, default=cfg.DATA_DIR)
    args = parser.parse_args()

    cfg.BACKBONE = args.backbone
    cfg.EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.LR = args.lr
    cfg.DATA_DIR = args.data_dir

    model, results = train_model(cfg)

    print("\n" + "=" * 60)
    print(f"  ✅ Training complete!")
    print(f"  Best val accuracy: {results['best_val_acc']:.4f}")
    if results["test"]:
        print(f"  Test accuracy:     {results['test']['accuracy']:.4f}")
        print(f"  Test AUC-ROC:      {results['test']['auroc']:.4f}")
    print(f"  Model saved:       ./chest_xray_pipeline/checkpoints/best_model.pth")
    print(f"  Plots saved:       ./chest_xray_pipeline/plots/")
    print("=" * 60)
