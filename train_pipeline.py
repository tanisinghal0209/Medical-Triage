import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from chest_xray_model import Trainer, ModelConfig, ChestXRayModel

# --- CUSTOM NIH DATASET CLASS ---
class NIHDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.classes = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax",
            "Consolidation", "Edema", "Emphysema", "Fibrosis",
            "Pleural_Thickening", "Hernia"
        ]
        
        # 1. CLEAN DATA: Remove any rows with missing labels and force to string
        self.df['Finding Labels'] = self.df['Finding Labels'].fillna("No Finding").astype(str)
        self.df = self.df.dropna(subset=['Finding Labels', 'Image Index'])
        
        # 2. MAP IMAGES: Search across all images_00x folders
        print("🔍 Mapping image locations (this may take a minute)...")
        self.image_map = {}
        for i in range(1, 13):
            folder = os.path.join(root_dir, f"images_{i:03d}", "images")
            if os.path.exists(folder):
                for img in os.listdir(folder):
                    self.image_map[img] = os.path.join(folder, img)

        # 3. FILTER DATA: Only keep rows where the image actually exists on disk
        initial_count = len(self.df)
        self.df = self.df[self.df['Image Index'].isin(self.image_map.keys())].reset_index(drop=True)
        print(f"✅ Data Sanitized: Kept {len(self.df)} valid images (Dropped {initial_count - len(self.df)} missing/corrupt).")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = self.image_map.get(img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Parse labels (Multi-label)
        label_str = self.df.iloc[idx, 1]
        labels = torch.zeros(len(self.classes))
        for i, cls in enumerate(self.classes):
            if cls in label_str:
                labels[i] = 1
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels, img_name

# --- MAIN EXECUTION ---
def start_training():
    NIH_PATH = "/Users/tanishasinghal/Downloads/archive (2)"
    CSV_PATH = os.path.join(NIH_PATH, "Data_Entry_2017.csv")
    
    cfg = ModelConfig()
    cfg.NUM_CLASSES = 14
    cfg.BATCH_SIZE = 16 # Adjust based on your memory
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Load Dataset
    full_dataset = NIHDataset(NIH_PATH, CSV_PATH, transform=transform)
    
    # 2. Split (90% Train, 10% Val)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True),
        "val": DataLoader(val_ds, batch_size=cfg.BATCH_SIZE)
    }

    # 3. Initialize Trainer
    trainer = Trainer(loaders, cfg=cfg)
    
    # --- AUTO-RESUME LOGIC ---
    checkpoint_path = "./chest_xray_pipeline/checkpoints/last_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"🔄 Found checkpoint: {checkpoint_path}. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('mps'))
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.history = checkpoint['history']
        start_epoch = checkpoint['epoch']
        print(f"✅ Successfully resumed from Epoch {start_epoch}")
    
    # Live Training!
    print("\n🚀 STARTING LIVE TRAINING DASHBOARD")
    print("-----------------------------------")
    trainer.fit()

if __name__ == "__main__":
    start_training()
