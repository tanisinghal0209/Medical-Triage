# 🏥 Medical AI Triage: Multi-Modal ConvNeXtV2 Pipeline

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-MPS_Accelerated-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced clinical decision support system that integrates state-of-the-art **ConvNeXtV2** image classification with Natural Language Processing (NLP) to provide automated triage scores and explainable heatmaps for chest X-ray analysis.

---

## 🚀 Key Features

### 🖼️ 1. Intelligent Image Classification
- **Backbone**: **ConvNeXtV2-Tiny** (Hugging Face Transformers), providing superior feature extraction over traditional ResNet/MobileNet models.
- **Dataset**: Trained on a subset of the NIH Chest X-ray 14 dataset (multi-label classification).
- **Optimization**: Fully accelerated for **Apple Silicon (M1/M2/M3/M4)** using Metal Performance Shaders (MPS).
- **Interpretability**: Custom **Grad-CAM** implementation for visual explainability, mapped to the final stages of the ConvNeXt encoder.

### 📝 2. Clinical NLP & Fusion
- **Clinical NER**: Automatically extracts symptoms, diseases, and medications from noisy clinical notes using spaCy/Transformer-based entities.
- **Triage Fusion**: A multi-modal engine that combines image probabilities (ConvNeXt) with clinical text features to calculate a prioritized **Triage Urgency Score**.
- **Summarization**: Generates concise clinical summaries from complex patient history.

### 🛠️ 3. Modern ML Engineering
- **Checkpointing**: Leverages the Hugging Face `Trainer` API for robust save/resume capabilities.
- **Evaluation**: Standalone evaluation suite providing AUROC, F1-score (Micro/Macro), and per-class performance metrics.
- **UI Dashboard**: Real-time Flask-based interface for image uploads, live prediction, and heatmap generation.

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tanisinghal0209/Medical-Triage.git
   cd Medical-Triage
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch torchvision pandas numpy scikit-learn flask transformers pillow tqdm matplotlib evaluate
   ```

3. **Data Configuration**:
   Ensure your NIH dataset is located at: `/Users/tanishasinghal/Downloads/archive (2)`.
   *You can adjust dataset paths in `train_convnextv2.py`.*

---

## 🖥️ Usage Guide

### 📈 Training & Evaluation
To start the ConvNeXtV2 training process:
```bash
python train_convnextv2.py
```

To evaluate the best checkpoints and see detailed metrics:
```bash
python evaluate_convnextv2.py
```

### 🌐 Running the Web Application
To launch the interactive triage dashboard:
```bash
python app.py
```
- Access the UI at: **`http://localhost:5050`**
- The app automatically detects the best **ConvNeXtV2** checkpoint in `convnextv2-nih-results/`.

---

## 📊 Technical Accomplishments

| Feature | Implementation | Benefit |
|---|---|---|
| **GPU Engine** | PyTorch MPS (Apple Silicon) | Native acceleration on Mac hardware |
| **Model** | ConvNeXtV2-Tiny | State-of-the-art accuracy vs MobileNet/ResNet |
| **Pipeline** | Multi-Modal Fusion | Combined Image + Text intelligence for triage |
| **Explainability** | Grad-CAM | Visual heatmaps for clinical validation |

---

## 📂 Project Structure
*   `train_convnextv2.py`: Training logic using HF Trainer.
*   `evaluate_convnextv2.py`: Detailed evaluation suite.
*   `app.py`: Flask backend and image service.
*   `triage_fusion.py`: Multi-modal score calculation.
*   `gradcam_xray.py`: ConvNeXt-optimized heatmap generation.
*   `legacy_models/`: Archive of previous ResNet/MobileNet experiments (local only).

---

## 📜 License
This project is licensed under the MIT License.

## 🤝 Acknowledgments
- **Dataset**: NIH Clinical Center for the ChestX-ray14 dataset.
- **Model Architecture**: Facebook AI Research for the ConvNeXtV2 backbone.
