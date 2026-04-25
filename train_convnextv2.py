import os
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    TrainingArguments, 
    Trainer
)
import numpy as np

# ── CONFIGURATION ──────────────────────────────────────────────────────────
DATASET_ROOT = Path("/Users/tanishasinghal/Downloads/archive (2)")
CSV_PATH = DATASET_ROOT / "Data_Entry_2017.csv"
TOTAL_IMAGES_TO_USE = 20000
MODEL_CHECKPOINT = "facebook/convnextv2-tiny-1k-224"
OUTPUT_DIR = "./convnextv2-nih-results"

# NIH-14 Labels
ALL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
label2id = {label: i for i, label in enumerate(ALL_LABELS)}
id2label = {i: label for i, label in enumerate(ALL_LABELS)}

# ── DATASET CLASS ──────────────────────────────────────────────────────────
class NIHConvNextDataset(Dataset):
    def __init__(self, df, image_processor, transform=None):
        self.df = df
        self.image_processor = image_processor
        self.transform = transform
        
        # Build image map for fast lookup across 12 folders
        self.image_map = {}
        for i in range(1, 13):
            folder = DATASET_ROOT / f"images_{i:03d}" / "images"
            if folder.exists():
                for img_path in folder.glob("*.png"):
                    self.image_map[img_path.name] = img_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        img_path = self.image_map.get(img_name)
        
        if not img_path:
            # Fallback/Error handling
            return self.__getitem__((idx + 1) % len(self.df))

        image = Image.open(img_path).convert("RGB")
        
        # Multi-label encoding
        labels = torch.zeros(len(ALL_LABELS))
        finding_str = row["Finding Labels"]
        for label in ALL_LABELS:
            if label in finding_str:
                labels[label2id[label]] = 1.0
        
        if self.transform:
            pixel_values = self.transform(image)
        else:
            # Use transformers default processor if no manual transform
            pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][0]

        return {"pixel_values": pixel_values, "labels": labels}

# ── PREPARE DATA ───────────────────────────────────────────────────────────
print(f"🔍 Loading labels from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Shuffle and take 20k
df = df.sample(n=TOTAL_IMAGES_TO_USE, random_state=42).reset_index(drop=True)
train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

print(f"✅ Data split: {len(train_df)} training, {len(test_df)} testing images.")

# ── MODEL & PROCESSOR ──────────────────────────────────────────────────────
print(f"📥 Loading processor and model: {MODEL_CHECKPOINT}...")
processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=len(ALL_LABELS),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    problem_type="multi_label_classification"
)

# ── TRANSFORMS (Modern ConvNeXt approach) ──────────────────────────────────
size = processor.size.get('height', 224)
train_transform = T.Compose([
    T.RandomResizedCrop(size),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=processor.image_mean, std=processor.image_std)
])

val_transform = T.Compose([
    T.Resize((size, size)),
    T.ToTensor(),
    T.Normalize(mean=processor.image_mean, std=processor.image_std)
])

train_ds = NIHConvNextDataset(train_df, processor, transform=train_transform)
test_ds = NIHConvNextDataset(test_df, processor, transform=val_transform)

import evaluate
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply sigmoid to get probabilities
    probs = 1 / (1 + np.exp(-logits))
    # Threshold at 0.5 for binary predictions
    predictions = (probs > 0.5).astype(float)
    
    # Calculate AUROC (Macro average)
    try:
        auroc = roc_auc_score(labels, probs, average="macro")
    except:
        auroc = 0.0 # Handle case where some classes might have no positive samples in eval
        
    f1 = f1_score(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    
    return {
        "auroc": auroc,
        "f1": f1,
        "accuracy": acc
    }

# ── TRAINING ARGUMENTS (Mac M4 Optimized) ──────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    remove_unused_columns=False,
    eval_strategy="epoch",            # Evaluate at the end of each epoch
    save_strategy="epoch",
    learning_rate=5e-5,
    
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,
    
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=10,                 # Log loss every 10 steps
    load_best_model_at_end=True,
    metric_for_best_model="auroc",     # Track AUROC for the best model
    save_total_limit=1,
    
    # HARDWARE: Accelerator will auto-detect MPS on Mac
    dataloader_num_workers=0,
    report_to="none"
)

# ── START TRAINING ─────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=processor,
    compute_metrics=compute_metrics,
)

print("🚀 Starting training on MPS (Metal)... Hold tight!")
trainer.train()

# Save final model
trainer.save_model(f"{OUTPUT_DIR}/best_convnext_nih")
print(f"🎉 Training complete! Model saved to {OUTPUT_DIR}/best_convnext_nih")
