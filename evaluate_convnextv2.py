"""
evaluate_convnextv2.py
----------------------
Standalone evaluation script for the trained ConvNeXtV2 NIH model.
Evaluates both:
  1. convnextv2-nih-results/best_convnext_nih  (saved by trainer.save_model at end of training)
  2. convnextv2-nih-results/checkpoint-2660    (final epoch checkpoint with optimizer state)

Run:
    python evaluate_convnextv2.py
"""

import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    classification_report, multilabel_confusion_matrix
)
from sklearn.model_selection import train_test_split
import json

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
DATASET_ROOT   = Path("/Users/tanishasinghal/Downloads/archive (2)")
CSV_PATH       = DATASET_ROOT / "Data_Entry_2017.csv"
OUTPUT_DIR     = Path("./convnextv2-nih-results")
TOTAL_IMAGES   = 20000   # must match training to get the same test split
BATCH_SIZE     = 16
DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu"

CHECKPOINTS_TO_EVAL = {
    "best_convnext_nih  (trainer.save_model)": str(OUTPUT_DIR / "best_convnext_nih"),
    "checkpoint-2660    (final epoch)":        str(OUTPUT_DIR / "checkpoint-2660"),
}

ALL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
label2id = {l: i for i, l in enumerate(ALL_LABELS)}

# ── DATASET ────────────────────────────────────────────────────────────────────
class NIHEvalDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        # Build fast image lookup across 12 sub-folders
        self.image_map: dict[str, Path] = {}
        for i in range(1, 13):
            folder = DATASET_ROOT / f"images_{i:03d}" / "images"
            if folder.exists():
                for p in folder.glob("*.png"):
                    self.image_map[p.name] = p

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_map.get(row["Image Index"])
        if img_path is None:
            return self.__getitem__((idx + 1) % len(self.df))

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        labels = torch.zeros(len(ALL_LABELS))
        for lbl in ALL_LABELS:
            if lbl in row["Finding Labels"]:
                labels[label2id[lbl]] = 1.0

        return pixel_values, labels


def get_test_df():
    """Reproduce the exact test split used during training."""
    df = pd.read_csv(CSV_PATH)
    df = df.sample(n=TOTAL_IMAGES, random_state=42).reset_index(drop=True)
    _, test_df = train_test_split(df, test_size=0.15, random_state=42)
    return test_df


def evaluate_checkpoint(name: str, ckpt_path: str, test_df: pd.DataFrame):
    print(f"\n{'='*70}")
    print(f"  Evaluating: {name}")
    print(f"  Path: {ckpt_path}")
    print(f"{'='*70}")

    if not Path(ckpt_path).exists():
        print(f"  ❌ Checkpoint not found — skipping.")
        return None

    # Load processor & model
    processor = AutoImageProcessor.from_pretrained(ckpt_path)
    model = AutoModelForImageClassification.from_pretrained(ckpt_path)
    model.eval()
    model.to(DEVICE)

    # Build the same val transform used at training time
    size = processor.size.get("height", 224)
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    dataset = NIHEvalDataset(test_df, transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (pixel_values, labels) in enumerate(loader):
            pixel_values = pixel_values.to(DEVICE)
            outputs = model(pixel_values=pixel_values)
            logits  = outputs.logits.cpu().numpy()
            probs   = 1 / (1 + np.exp(-logits))   # sigmoid
            all_probs.append(probs)
            all_labels.append(labels.numpy())

            if (batch_idx + 1) % 20 == 0:
                done = (batch_idx + 1) * BATCH_SIZE
                print(f"  [{done}/{len(dataset)}] samples processed…", flush=True)

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    preds      = (all_probs > 0.5).astype(float)

    # ── Metrics ────────────────────────────────────────────────────────────────
    try:
        auroc = roc_auc_score(all_labels, all_probs, average="macro")
    except Exception:
        auroc = float("nan")

    f1_macro  = f1_score(all_labels, preds, average="macro",  zero_division=0)
    f1_micro  = f1_score(all_labels, preds, average="micro",  zero_division=0)
    f1_sample = f1_score(all_labels, preds, average="samples",zero_division=0)
    acc       = accuracy_score(all_labels, preds)

    print(f"\n  ✅ Results:")
    print(f"     AUROC (macro)    : {auroc:.4f}")
    print(f"     F1   (macro)     : {f1_macro:.4f}")
    print(f"     F1   (micro)     : {f1_micro:.4f}")
    print(f"     F1   (samples)   : {f1_sample:.4f}")
    print(f"     Exact-match acc  : {acc:.4f}")

    # Per-class AUROC & F1
    print(f"\n  📊 Per-class breakdown:")
    print(f"  {'Label':<22} {'AUROC':>8} {'F1':>8} {'Support':>9}")
    print(f"  {'-'*50}")
    for i, lbl in enumerate(ALL_LABELS):
        support = int(all_labels[:, i].sum())
        try:
            c_auroc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except Exception:
            c_auroc = float("nan")
        c_f1 = f1_score(all_labels[:, i], preds[:, i], zero_division=0)
        print(f"  {lbl:<22} {c_auroc:>8.4f} {c_f1:>8.4f} {support:>9d}")

    results = {
        "checkpoint": name,
        "path":       ckpt_path,
        "auroc_macro": round(auroc,    4),
        "f1_macro":    round(f1_macro, 4),
        "f1_micro":    round(f1_micro, 4),
        "f1_samples":  round(f1_sample,4),
        "exact_match_accuracy": round(acc, 4),
    }
    return results


def main():
    print("🔍 Loading test split (reproducing training split)…")
    test_df = get_test_df()
    print(f"   Test set size: {len(test_df)} images")
    print(f"   Device: {DEVICE}\n")

    all_results = []
    for name, path in CHECKPOINTS_TO_EVAL.items():
        result = evaluate_checkpoint(name, path, test_df)
        if result:
            all_results.append(result)

    # Save summary JSON
    out_path = OUTPUT_DIR / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*70}")
    print("  📋 SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        print(f"\n  {r['checkpoint'].strip()}")
        print(f"    AUROC : {r['auroc_macro']}")
        print(f"    F1    : {r['f1_macro']}  (macro)")
        print(f"    Acc   : {r['exact_match_accuracy']}  (exact match)")

    print(f"\n  Results saved → {out_path}\n")


if __name__ == "__main__":
    main()
