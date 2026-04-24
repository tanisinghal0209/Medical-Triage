"""
================================================================================
  STEP 2 — SPATIAL COMPUTER VISION ENGINEER
  Multi-Modal Deepfake Detection Project
  Module: Chest X-Ray Multi-Label Classifier (Transfer Learning)
  Backbone: ResNet50 (pretrained on ImageNet)
  Compatible: Google Colab | PyTorch GPU
================================================================================

DEPENDS ON:  chest_xray_pipeline.py  (Step 1)
  - Expects loaders dict from run_pipeline()
  - OR can be run standalone with mock=True

OUTPUTS:
  <OUTPUT_ROOT>/
      checkpoints/
          best_model.pth          ← best val-AUROC checkpoint
          last_model.pth          ← final epoch weights
      plots/
          training_curves.png
          confusion_matrix.png
          roc_curves.png
      results/
          classification_report.json
          predictions.csv

QUICK START (Colab):
  from chest_xray_model import ModelConfig, Trainer, predict_image
  trainer = Trainer(loaders)        # loaders from Step 1
  trainer.fit()
  results = predict_image("path/to/xray.jpg")
================================================================================
"""

# ==============================================================================
# SECTION 0 — IMPORTS
# ==============================================================================

import os
import json
import time
import copy
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as T

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,
    multilabel_confusion_matrix,
    average_precision_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1 — MODEL CONFIGURATION
# ==============================================================================

class ModelConfig:
    """
    All hyperparameters in one place. Edit here only.
    """
    # ── PATHS (mirror Step 1 OUTPUT_ROOT) ────────────────────────────────────
    OUTPUT_ROOT: str = "./chest_xray_pipeline"

    # ── MODEL ────────────────────────────────────────────────────────────────
    BACKBONE: str          = "mobilenet_v2" # "mobilenet_v2" or "resnet50"
    PRETRAINED: bool       = True
    FREEZE_BACKBONE: bool  = False        # False = fine-tune all layers
    DROPOUT_RATE: float    = 0.3          # Lower dropout for MobileNet
    NUM_CLASSES: int       = 14

    DISEASE_CLASSES: List[str] = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia",
    ]

    # ── TRAINING ─────────────────────────────────────────────────────────────
    EPOCHS: int            = 30
    LR: float              = 1e-4
    WEIGHT_DECAY: float    = 1e-5

    # ── LOSS FUNCTION ────────────────────────────────────────────────────────
    # Class-frequency-based positive weights to address imbalance.
    # Will be computed from train loader if USE_POS_WEIGHT=True.
    USE_POS_WEIGHT: bool   = True
    POS_WEIGHT_CAP: float  = 10.0    # Clamp max weight to avoid instability

    # ── SCHEDULER ────────────────────────────────────────────────────────────
    # ReduceLROnPlateau — halves LR if val AUROC stagnates
    SCHEDULER_PATIENCE: int    = 3
    SCHEDULER_FACTOR: float    = 0.5
    MIN_LR: float              = 1e-7

    # ── EARLY STOPPING ───────────────────────────────────────────────────────
    EARLY_STOP_PATIENCE: int   = 7     # epochs without val improvement
    EARLY_STOP_DELTA: float    = 1e-4  # minimum improvement threshold

    # ── DECISION THRESHOLD ───────────────────────────────────────────────────
    # Probability cutoff for converting sigmoid output → binary prediction
    THRESHOLD: float = 0.5

    # ── REPRODUCIBILITY ──────────────────────────────────────────────────────
    SEED: int = 42


mcfg = ModelConfig()

# ── Output directories ────────────────────────────────────────────────────────
def _makedirs(mcfg: ModelConfig) -> Dict[str, Path]:
    base = Path(mcfg.OUTPUT_ROOT)
    dirs = {
        "checkpoints": base / "checkpoints",
        "plots":       base / "plots",
        "results":     base / "results",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

DIRS = _makedirs(mcfg)

# ── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    log.info("🚀 Active device: APPLE SILICON GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    log.info("🚀 Active device: NVIDIA GPU (CUDA)")
else:
    DEVICE = torch.device("cpu")
    log.warning("⚠️ Active device: CPU (Training will be slow)")

torch.manual_seed(mcfg.SEED)
np.random.seed(mcfg.SEED)


# ==============================================================================
# SECTION 2 — MODEL ARCHITECTURE
# ==============================================================================

class ChestXRayModel(nn.Module):
    """
    Modular model supporting MobileNetV2 and ResNet50 backbones.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # ── Load pretrained backbone ──────────────────────────────────────────
        if cfg.BACKBONE == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if cfg.PRETRAINED else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features # 2048
            backbone.fc = nn.Identity()
            log.info("Backbone: ResNet50 loaded")
        else: # Default to MobileNetV2
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if cfg.PRETRAINED else None
            backbone = models.mobilenet_v2(weights=weights)
            in_features = backbone.last_channel # 1280
            backbone.classifier = nn.Identity()
            log.info("Backbone: MobileNetV2 loaded")

        # ── Optionally freeze all backbone layers ────────────────────────────
        if cfg.FREEZE_BACKBONE:
            for param in backbone.parameters():
                param.requires_grad = False
            log.info("Backbone frozen.")

        self.backbone = backbone

        # ── Custom classification head ────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=cfg.DROPOUT_RATE),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=cfg.DROPOUT_RATE / 2),
            nn.Linear(512, cfg.NUM_CLASSES),
        )

        # ── Initialise head weights ─────────────────
        self._init_weights()
        log.info(
            f"Model built: {self._count_params():,} total params | "
            f"{self._count_params(trainable=True):,} trainable"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet outputs [B, 2048, 1, 1] or [B, 2048] after Identity
        # MobileNet outputs [B, 1280, 7, 7] or [B, 1280] after Identity
        features = self.backbone(x)
        # Ensure we have [B, C, H, W] for AdaptiveAvgPool2d if needed, 
        # but Identity might return flattened features depending on the layer swapped.
        if len(features.shape) == 2:
            # If already flattened, skip AdaptiveAvgPool2d/Flatten
            logits = self.classifier[4:](features) # Linear(in, 512)...
        else:
            logits = self.classifier(features)
        return logits

    def _init_weights(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _count_params(self, trainable: bool = False) -> int:
        params = self.parameters() if not trainable \
            else filter(lambda p: p.requires_grad, self.parameters())
        return sum(p.numel() for p in params)


# ==============================================================================
# SECTION 3 — LOSS FUNCTION WITH CLASS BALANCING
# ==============================================================================

def compute_pos_weights(
    train_loader: DataLoader,
    num_classes: int,
    cap: float = 10.0,
) -> torch.Tensor:
    """
    Optimized version: Tries to get labels from dataset directly to avoid 
    loading all images during the weight computation phase.
    """
    log.info("Computing positive weights from training labels…")
    
    # Try to access labels directly from the dataset to save 20+ minutes
    try:
        if hasattr(train_loader.dataset, 'dataset') and hasattr(train_loader.dataset.dataset, 'df'):
            # It's a Subset of NIHDataset
            df = train_loader.dataset.dataset.df.iloc[train_loader.dataset.indices]
            # Convert multi-label strings to matrix
            classes = [
                "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                "Mass", "Nodule", "Pneumonia", "Pneumothorax",
                "Consolidation", "Edema", "Emphysema", "Fibrosis",
                "Pleural_Thickening", "Hernia"
            ]
            n_pos = torch.zeros(num_classes)
            for i, cls in enumerate(classes):
                n_pos[i] = df['Finding Labels'].str.contains(cls).sum()
            n_total = len(df)
        else:
            raise AttributeError("Fallback to slow method")
    except:
        # Fallback to slow method if dataset structure is different
        n_pos = torch.zeros(num_classes)
        n_total = 0
        for _, labels, _ in train_loader:
            n_pos   += labels.sum(dim=0)
            n_total += labels.size(0)

    pos_weight = (n_total - n_pos) / (n_pos + 1e-6)
    pos_weight = pos_weight.clamp(min=1.0, max=cap)
    log.info(f"pos_weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
    return pos_weight


# ==============================================================================
# SECTION 4 — METRICS
# ==============================================================================

class MetricAccumulator:
    """
    Accumulates logits + labels across batches, then computes
    Accuracy, F1 (macro), AUROC (macro) at epoch end.
    """

    def __init__(self, disease_classes: List[str], threshold: float = 0.5):
        self.classes   = disease_classes
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._logits: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self._logits.append(logits.detach().cpu().numpy())
        self._labels.append(labels.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        logits = np.concatenate(self._logits, axis=0)   # [N, C]
        labels = np.concatenate(self._labels, axis=0)   # [N, C]
        
        # SAFETY CHECK: Remove any NaN values that might have crept into labels/logits
        if np.isnan(labels).any():
            mask = ~np.isnan(labels).any(axis=1)
            labels = labels[mask]
            logits = logits[mask]
            
        probs  = 1.0 / (1.0 + np.exp(-logits))          # sigmoid

        preds  = (probs >= self.threshold).astype(int)

        # ── Exact-match accuracy ──────────────────────────────────────────────
        exact_match = (preds == labels).all(axis=1).mean()

        # ── Per-label accuracy ────────────────────────────────────────────────
        per_label_acc = (preds == labels).mean(axis=0)

        # ── Macro F1 ──────────────────────────────────────────────────────────
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_per   = f1_score(labels, preds, average=None, zero_division=0)

        # ── AUROC (skip classes with no positive samples in batch) ────────────
        valid_cols = labels.sum(axis=0) > 0
        if valid_cols.sum() > 0:
            auroc = roc_auc_score(
                labels[:, valid_cols],
                probs[:, valid_cols],
                average="macro",
            )
            auroc_per = roc_auc_score(
                labels[:, valid_cols],
                probs[:, valid_cols],
                average=None,
            )
            auroc_dict = {
                self.classes[i]: float(auroc_per[j])
                for j, i in enumerate(np.where(valid_cols)[0])
            }
        else:
            auroc = 0.0
            auroc_dict = {}

        return {
            "exact_match_acc": float(exact_match),
            "per_label_acc":   per_label_acc.tolist(),
            "f1_macro":        float(f1_macro),
            "f1_per_class":    {c: float(v) for c, v in zip(self.classes, f1_per)},
            "auroc_macro":     float(auroc),
            "auroc_per_class": auroc_dict,
            "probs":           probs,
            "preds":           preds,
            "labels":          labels,
        }


# ==============================================================================
# SECTION 5 — EARLY STOPPING
# ==============================================================================

class EarlyStopping:
    """
    Monitors a metric (higher = better by default) and stops training
    when no improvement is observed for `patience` consecutive epochs.
    Also saves the best model weights.
    """

    def __init__(
        self,
        patience: int = 7,
        delta: float = 1e-4,
        save_path: Optional[Path] = None,
        mode: str = "max",      # 'max' for AUROC, 'min' for loss
    ):
        self.patience   = patience
        self.delta      = delta
        self.save_path  = save_path
        self.mode       = mode
        self.counter    = 0
        self.best_score = None
        self.best_weights = None
        self.stop        = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == "max" and score > self.best_score + self.delta:
            improved = True
        elif self.mode == "min" and score < self.best_score - self.delta:
            improved = True

        if improved:
            self.best_score   = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter      = 0
            if self.save_path:
                torch.save(
                    {
                        "model_state_dict": self.best_weights,
                        "best_score": self.best_score,
                    },
                    self.save_path,
                )
                log.info(f"  ✓ Best model saved (score={score:.4f}) → {self.save_path}")
        else:
            self.counter += 1
            log.info(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
                log.info("  Early stopping triggered.")

        return self.stop


# ==============================================================================
# SECTION 6 — TRAINING & VALIDATION LOOPS
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    metrics: MetricAccumulator,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, Dict]:
    """
    Runs one full training epoch with optional AMP (mixed precision).

    Returns: (avg_loss, metrics_dict)
    """
    model.train()
    metrics.reset()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (imgs, labels, _) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        metrics.update(logits, labels)

        if (batch_idx + 1) % max(1, n_batches // 5) == 0:
            log.info(
                f"  Batch [{batch_idx+1:4d}/{n_batches}] "
                f"loss={loss.item():.4f}"
            )

    avg_loss = total_loss / n_batches
    return avg_loss, metrics.compute()


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: MetricAccumulator,
) -> Tuple[float, Dict]:
    """
    Runs one full validation epoch (no gradients).

    Returns: (avg_loss, metrics_dict)
    """
    model.eval()
    metrics.reset()
    total_loss = 0.0

    for imgs, labels, _ in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        metrics.update(logits, labels)

    avg_loss = total_loss / len(loader)
    return avg_loss, metrics.compute()


# ==============================================================================
# SECTION 7 — TRAINER CLASS (ORCHESTRATOR)
# ==============================================================================

class Trainer:
    """
    High-level orchestrator that wires together:
        model → loss → optimizer → scheduler → early stopping → metrics

    Usage:
        trainer = Trainer(loaders)
        history = trainer.fit()
        trainer.evaluate_test()
    """

    def __init__(
        self,
        loaders: Dict[str, DataLoader],
        cfg: ModelConfig = mcfg,
    ):
        self.loaders = loaders
        self.cfg     = cfg
        self.device  = DEVICE

        # ── Build model ───────────────────────────────────────────────────────
        self.model = ChestXRayModel(cfg).to(self.device)

        # ── Loss: BCEWithLogitsLoss + class balancing ─────────────────────────
        if cfg.USE_POS_WEIGHT and "train" in loaders:
            pos_weight = compute_pos_weights(
                loaders["train"], cfg.NUM_CLASSES, cfg.POS_WEIGHT_CAP
            ).to(self.device)
        else:
            pos_weight = None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # ── Optimizer: AdamW ──────────────────────────────────────────────────
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY,
        )

        # ── LR Scheduler: ReduceLROnPlateau ───────────────────────────────────
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            patience=cfg.SCHEDULER_PATIENCE,
            factor=cfg.SCHEDULER_FACTOR,
            min_lr=cfg.MIN_LR,
        )

        # ── Early stopping ────────────────────────────────────────────────────
        self.early_stopping = EarlyStopping(
            patience=cfg.EARLY_STOP_PATIENCE,
            delta=cfg.EARLY_STOP_DELTA,
            save_path=DIRS["checkpoints"] / "best_model.pth",
            mode="max",
        )

        # ── AMP scaler (only on CUDA) ─────────────────────────────────────────
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if self.device.type == "cuda" else None
        )

        # ── Metrics accumulators ──────────────────────────────────────────────
        self.train_metrics = MetricAccumulator(cfg.DISEASE_CLASSES, cfg.THRESHOLD)
        self.val_metrics   = MetricAccumulator(cfg.DISEASE_CLASSES, cfg.THRESHOLD)
        self.test_metrics  = MetricAccumulator(cfg.DISEASE_CLASSES, cfg.THRESHOLD)

        # ── History log ──────────────────────────────────────────────────────
        self.history: Dict[str, List] = {
            "train_loss": [], "val_loss": [],
            "train_f1": [], "val_f1": [],
            "train_auroc": [], "val_auroc": [],
            "lr": [],
        }

    # ── fit() ─────────────────────────────────────────────────────────────────
    def fit(self) -> Dict[str, List]:
        """
        Runs the full training loop for cfg.EPOCHS epochs or until
        early stopping fires.

        Returns: history dict with per-epoch metrics.
        """
        log.info("=" * 60)
        log.info(f"  TRAINING START — {self.cfg.EPOCHS} max epochs")
        log.info(f"  Device: {self.device} | AMP: {self.scaler is not None}")
        log.info("=" * 60)

        for epoch in range(1, self.cfg.EPOCHS + 1):
            t0 = time.time()
            log.info(f"\n── EPOCH {epoch}/{self.cfg.EPOCHS} "
                     f"(LR={self.optimizer.param_groups[0]['lr']:.2e}) ──")

            # ── Train ─────────────────────────────────────────────────────────
            train_loss, train_m = train_one_epoch(
                self.model, self.loaders["train"],
                self.criterion, self.optimizer,
                self.device, self.train_metrics, self.scaler,
            )

            # ── Validate ──────────────────────────────────────────────────────
            val_loss, val_m = validate_one_epoch(
                self.model, self.loaders["val"],
                self.criterion, self.device, self.val_metrics,
            )

            # ── Scheduler step on val AUROC ───────────────────────────────────
            self.scheduler.step(val_m["auroc_macro"])

            # ── Log epoch summary ─────────────────────────────────────────────
            elapsed = time.time() - t0
            log.info(
                f"  Train → loss={train_loss:.4f} | "
                f"F1={train_m['f1_macro']:.4f} | "
                f"AUROC={train_m['auroc_macro']:.4f}"
            )
            log.info(
                f"  Val   → loss={val_loss:.4f} | "
                f"F1={val_m['f1_macro']:.4f} | "
                f"AUROC={val_m['auroc_macro']:.4f} | "
                f"time={elapsed:.1f}s"
            )

            # ── Append to history ─────────────────────────────────────────────
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_f1"].append(train_m["f1_macro"])
            self.history["val_f1"].append(val_m["f1_macro"])
            self.history["train_auroc"].append(train_m["auroc_macro"])
            self.history["val_auroc"].append(val_m["auroc_macro"])
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # ── SAVE INTERMEDIATE CHECKPOINT (New Resume Feature) ─────────────
            checkpoint_path = DIRS["checkpoints"] / "last_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            }, checkpoint_path)
            log.info(f"  💾 Progress saved to {checkpoint_path}")

            # ── Early stopping check ──────────────────────────────────────────
            should_stop = self.early_stopping(val_m["auroc_macro"], self.model)
            if should_stop:
                log.info(f"  Training stopped at epoch {epoch}.")
                break

        # ── Save last checkpoint ──────────────────────────────────────────────
        torch.save(
            {"model_state_dict": self.model.state_dict(),
             "history": self.history},
            DIRS["checkpoints"] / "last_model.pth",
        )
        log.info(f"Last checkpoint saved → {DIRS['checkpoints']}/last_model.pth")

        # ── Restore best weights ──────────────────────────────────────────────
        if self.early_stopping.best_weights is not None:
            self.model.load_state_dict(self.early_stopping.best_weights)
            log.info("Best weights restored for final evaluation.")

        # ── Plot training curves ──────────────────────────────────────────────
        plot_training_curves(self.history, DIRS["plots"])

        return self.history

    # ── evaluate_test() ───────────────────────────────────────────────────────
    def evaluate_test(self) -> Dict:
        """
        Evaluates the best model on the held-out test set.
        Saves classification report, confusion matrix, and ROC curves.
        """
        log.info("\n" + "=" * 60)
        log.info("  TEST SET EVALUATION")
        log.info("=" * 60)

        test_loss, test_m = validate_one_epoch(
            self.model, self.loaders["test"],
            self.criterion, self.device, self.test_metrics,
        )

        log.info(f"  Test Loss  : {test_loss:.4f}")
        log.info(f"  Test F1    : {test_m['f1_macro']:.4f}")
        log.info(f"  Test AUROC : {test_m['auroc_macro']:.4f}")
        log.info(f"  Exact Match Acc: {test_m['exact_match_acc']:.4f}")

        # ── Per-class summary ─────────────────────────────────────────────────
        log.info("\n  Per-class AUROC:")
        for cls, auc in test_m["auroc_per_class"].items():
            log.info(f"    {cls:<25s}: {auc:.4f}")

        # ── Save JSON report ──────────────────────────────────────────────────
        report = {
            "test_loss": test_loss,
            "f1_macro":  test_m["f1_macro"],
            "auroc_macro": test_m["auroc_macro"],
            "exact_match_acc": test_m["exact_match_acc"],
            "f1_per_class": test_m["f1_per_class"],
            "auroc_per_class": test_m["auroc_per_class"],
        }
        report_path = DIRS["results"] / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"\nClassification report saved → {report_path}")

        # ── Save predictions CSV ──────────────────────────────────────────────
        pred_df = pd.DataFrame(
            test_m["probs"],
            columns=[f"prob_{c}" for c in self.cfg.DISEASE_CLASSES],
        )
        gt_df = pd.DataFrame(
            test_m["labels"],
            columns=[f"gt_{c}" for c in self.cfg.DISEASE_CLASSES],
        )
        pd.concat([pred_df, gt_df], axis=1).to_csv(
            DIRS["results"] / "predictions.csv", index=False
        )

        # ── Visualisations ────────────────────────────────────────────────────
        plot_confusion_matrix(
            test_m["labels"], test_m["preds"],
            self.cfg.DISEASE_CLASSES, DIRS["plots"],
        )
        plot_roc_curves(
            test_m["labels"], test_m["probs"],
            self.cfg.DISEASE_CLASSES, DIRS["plots"],
        )

        return report


# ==============================================================================
# SECTION 8 — PREDICTION FUNCTION
# ==============================================================================

def predict_image(
    image_path: str,
    model: Optional[nn.Module] = None,
    checkpoint_path: Optional[str] = None,
    cfg: ModelConfig = mcfg,
    device: torch.device = DEVICE,
) -> Dict[str, float]:
    """
    Runs inference on a single chest X-ray image.

    Args:
        image_path      : Path to a .jpg / .png X-ray image.
        model           : Trained ChestXRayResNet50 (if already loaded).
        checkpoint_path : Path to .pth checkpoint (used if model=None).
        cfg             : ModelConfig instance.
        device          : torch.device.

    Returns:
        dict  {disease_name: probability_float}  — sorted descending.

    Example:
        results = predict_image("/content/sample_xray.jpg")
        # → {'Effusion': 0.82, 'Atelectasis': 0.61, ..., 'Hernia': 0.03}
    """
    # ── Load model if not provided ────────────────────────────────────────────
    if model is None:
        ckpt_path = checkpoint_path or str(DIRS["checkpoints"] / "best_model.pth")
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Run trainer.fit() first or pass a trained model."
            )
        model = ChestXRayResNet50(cfg).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        log.info(f"Model loaded from checkpoint: {ckpt_path}")

    model.eval()

    # ── Preprocessing (identical to val/test transform) ───────────────────────
    transform = T.Compose([
        T.Resize((cfg.NUM_CLASSES * 16, cfg.NUM_CLASSES * 16)),  # 224x224
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    # Fix: hardcode 224 for clarity
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Cannot open image at {image_path}: {e}")

    tensor = transform(img).unsqueeze(0).to(device)   # [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(tensor)                          # [1, 14]
        probs  = torch.sigmoid(logits).squeeze(0)      # [14]

    results = {
        cls: round(float(prob), 4)
        for cls, prob in zip(cfg.DISEASE_CLASSES, probs.cpu())
    }

    # ── Sort by probability descending ────────────────────────────────────────
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    log.info("Prediction results:")
    for disease, prob in results.items():
        bar = "█" * int(prob * 20)
        flag = "⚠" if prob >= cfg.THRESHOLD else " "
        log.info(f"  {flag} {disease:<25s}: {prob:.4f}  {bar}")

    return results


# ==============================================================================
# SECTION 9 — VISUALISATION: TRAINING CURVES
# ==============================================================================

def plot_training_curves(history: Dict[str, List], output_dir: Path) -> None:
    """
    Plots 3-panel training curves:
        Panel 1: Train vs Val Loss
        Panel 2: Train vs Val F1 (macro)
        Panel 3: Train vs Val AUROC (macro) + LR on secondary axis
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Training History", fontsize=15, fontweight="bold")

    # Colour palette
    C_TRAIN = "#1565C0"
    C_VAL   = "#E53935"
    C_LR    = "#43A047"

    # ── Loss ─────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color=C_TRAIN, linewidth=2, label="Train")
    ax.plot(epochs, history["val_loss"],   color=C_VAL,   linewidth=2, label="Val", linestyle="--")
    ax.set_title("BCE Loss", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3)

    # ── F1 ───────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history["train_f1"], color=C_TRAIN, linewidth=2, label="Train")
    ax.plot(epochs, history["val_f1"],   color=C_VAL,   linewidth=2, label="Val", linestyle="--")
    ax.set_title("Macro F1-Score", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)

    # ── AUROC + LR ────────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(epochs, history["train_auroc"], color=C_TRAIN, linewidth=2, label="Train AUROC")
    ax.plot(epochs, history["val_auroc"],   color=C_VAL,   linewidth=2, label="Val AUROC", linestyle="--")
    ax.set_title("Macro AUROC + LR", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1); ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(epochs, history["lr"], color=C_LR, linewidth=1.5,
             linestyle=":", alpha=0.7, label="LR")
    ax2.set_ylabel("Learning Rate", color=C_LR)
    ax2.tick_params(axis="y", labelcolor=C_LR)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    out_path = output_dir / "training_curves.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"Training curves saved → {out_path}")


# ==============================================================================
# SECTION 10 — VISUALISATION: CONFUSION MATRIX
# ==============================================================================

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    disease_classes: List[str],
    output_dir: Path,
) -> None:
    """
    Plots per-class 2×2 confusion matrices (TP/FP/TN/FN) in a grid.
    Uses multilabel_confusion_matrix from sklearn.
    """
    mcm = multilabel_confusion_matrix(labels, preds)  # [C, 2, 2]
    n_classes = len(disease_classes)

    n_cols = 4
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
    fig.suptitle(
        "Per-Class Confusion Matrices (Test Set)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    cmap = LinearSegmentedColormap.from_list("blue_red", ["#FFFFFF", "#1565C0"])

    for idx, (cls_cm, cls_name) in enumerate(zip(mcm, disease_classes)):
        ax = axes[idx // n_cols][idx % n_cols]
        # cls_cm:  [[TN, FP], [FN, TP]]
        sns.heatmap(
            cls_cm, annot=True, fmt="d", cmap=cmap,
            xticklabels=["Pred NEG", "Pred POS"],
            yticklabels=["True NEG", "True POS"],
            ax=ax, cbar=False,
            linewidths=0.5, linecolor="gray",
        )
        tn, fp, fn, tp = cls_cm.ravel()
        sens = tp / (tp + fn + 1e-6)
        spec = tn / (tn + fp + 1e-6)
        ax.set_title(
            f"{cls_name}\nSens={sens:.2f} Spec={spec:.2f}",
            fontsize=8, fontweight="bold",
        )

    # Hide unused axes
    for i in range(n_classes, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].axis("off")

    plt.tight_layout()
    out_path = output_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"Confusion matrix saved → {out_path}")


# ==============================================================================
# SECTION 11 — VISUALISATION: ROC CURVES
# ==============================================================================

def plot_roc_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    disease_classes: List[str],
    output_dir: Path,
) -> None:
    """
    Plots individual ROC curves for each disease class.
    Annotates each curve with its AUC score.
    """
    from sklearn.metrics import roc_curve

    n_classes = len(disease_classes)
    n_cols = 4
    n_rows = (n_classes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
    fig.suptitle(
        "ROC Curves — Per Disease Class (Test Set)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    axes = axes.flatten()

    for idx, cls_name in enumerate(disease_classes):
        ax = axes[idx]
        y_true = labels[:, idx]
        y_score = probs[:, idx]

        if y_true.sum() == 0:
            ax.text(0.5, 0.5, "No positives\nin test set",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cls_name, fontsize=9)
            ax.axis("off")
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        ax.plot(fpr, tpr, color="#1565C0", linewidth=2, label=f"AUC={auc:.3f}")
        ax.plot([0, 1], [0, 1], color="#BDBDBD", linewidth=1, linestyle="--")
        ax.fill_between(fpr, tpr, alpha=0.08, color="#1565C0")
        ax.set_title(f"{cls_name}", fontsize=9, fontweight="bold")
        ax.set_xlabel("FPR", fontsize=7)
        ax.set_ylabel("TPR", fontsize=7)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    for i in range(n_classes, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    out_path = output_dir / "roc_curves.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"ROC curves saved → {out_path}")


# ==============================================================================
# SECTION 12 — MODEL LOADING UTILITY
# ==============================================================================

def load_model(
    checkpoint_path: str,
    cfg: ModelConfig = mcfg,
    device: torch.device = DEVICE,
) -> nn.Module:
    """
    Convenience function to load a saved model from a .pth checkpoint.

    Usage:
        model = load_model("/content/chest_xray_pipeline/checkpoints/best_model.pth")
        results = predict_image("xray.jpg", model=model)
    """
    model = ChestXRayResNet50(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(f"Model loaded from {checkpoint_path}")
    return model


# ==============================================================================
# SECTION 13 — STANDALONE SMOKE TEST (no real data needed)
# ==============================================================================

def _run_smoke_test() -> None:
    """
    Tests the entire pipeline end-to-end with random tensors.
    Verifies: model forward, loss computation, one train step, metrics, predict.
    """
    import tempfile
    log.info("=" * 60)
    log.info("  SMOKE TEST — no real data required")
    log.info("=" * 60)

    B, C, H, W = 4, 3, 224, 224
    N_CLS = mcfg.NUM_CLASSES

    # Fake batches
    fake_imgs   = torch.randn(B, C, H, W)
    fake_labels = (torch.rand(B, N_CLS) > 0.7).float()

    # ── Forward pass ──────────────────────────────────────────────────────────
    model = ChestXRayResNet50(mcfg).to(DEVICE)
    logits = model(fake_imgs.to(DEVICE))
    assert logits.shape == (B, N_CLS), f"Unexpected output shape: {logits.shape}"
    log.info(f"  Forward pass OK — output shape: {logits.shape}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(logits, fake_labels.to(DEVICE))
    log.info(f"  Loss OK — value: {loss.item():.4f}")

    # ── Backward ──────────────────────────────────────────────────────────────
    loss.backward()
    log.info("  Backward pass OK")

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = MetricAccumulator(mcfg.DISEASE_CLASSES, mcfg.THRESHOLD)
    acc.update(logits.detach(), fake_labels.to(DEVICE))
    m = acc.compute()
    log.info(f"  Metrics OK — F1={m['f1_macro']:.4f} | AUROC={m['auroc_macro']:.4f}")

    # ── Predict from synthetic image ──────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(f.name)
        tmp_path = f.name

    preds = predict_image(tmp_path, model=model)
    assert len(preds) == N_CLS, "predict_image returned wrong number of classes"
    log.info(f"  predict_image OK — {len(preds)} disease probabilities returned")

    log.info("  ALL SMOKE TESTS PASSED ✓")


# ==============================================================================
# SECTION 14 — MAIN ENTRY POINT
# ==============================================================================

"""
================================================================================
COPY THESE CELLS INTO YOUR COLAB NOTEBOOK
================================================================================

─── CELL A: Mount Drive + run Step 1 pipeline ───────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

# Run Step 1 first to get loaders
from chest_xray_pipeline import run_pipeline
loaders = run_pipeline(mock=False)   # or mock=True for testing


─── CELL B: Train the model ─────────────────────────────────────────────────
from chest_xray_model import Trainer, ModelConfig

# (Optional) Override config
cfg = ModelConfig()
cfg.EPOCHS = 20
cfg.LR = 3e-5

trainer = Trainer(loaders, cfg=cfg)
history = trainer.fit()


─── CELL C: Evaluate on test set ────────────────────────────────────────────
report = trainer.evaluate_test()
print(f"Test AUROC: {report['auroc_macro']:.4f}")
print(f"Test F1:    {report['f1_macro']:.4f}")


─── CELL D: Single-image inference ──────────────────────────────────────────
from chest_xray_model import predict_image

results = predict_image("/content/my_xray.jpg")
# Output: {'Effusion': 0.82, 'Atelectasis': 0.61, ..., 'Hernia': 0.03}

for disease, prob in results.items():
    print(f"{disease:<25s}: {prob:.1%}")


─── CELL E: Load saved model ────────────────────────────────────────────────
from chest_xray_model import load_model, predict_image

model  = load_model("/content/chest_xray_pipeline/checkpoints/best_model.pth")
result = predict_image("/content/xray.jpg", model=model)
================================================================================
"""

if __name__ == "__main__":
    _run_smoke_test()
