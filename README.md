# 🏥 Medical AI Triage: Multi-Modal Chest X-Ray Pipeline

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch MPS](https://img.shields.io/badge/PyTorch-MPS_Accelerated-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced clinical decision support system that integrates deep learning image classification with Natural Language Processing (NLP) to provide automated triage scores and explainable heatmaps for chest X-ray analysis.

---

## 🚀 Key Features

### 🖼️ 1. Intelligent Image Classification
- **Backbone**: High-speed **MobileNetV2** optimized for edge deployment.
- **Dataset**: Trained on the NIH Chest X-ray 14 dataset (112,000+ images).
- **Optimization**: Fully accelerated for **Apple Silicon (M4/M3/M2/M1)** using Metal Performance Shaders (MPS).
- **Interpretability**: Integrated **Grad-CAM** for visual explainability, showing heatmaps of where the model is looking for disease.

### 📝 2. Clinical NLP & Fusion
- **Clinical NER**: Automatically extracts symptoms, diseases, and medications from clinical notes.
- **Triage Fusion**: A multi-modal engine that combines image probabilities with clinical text features to calculate a prioritized **Triage Urgency Score**.
- **Summarization**: Generates concise clinical summaries from complex patient notes.

### 🛠️ 3. Robust Pipeline Engineering
- **Smart Resume**: Automatic checkpointing at every epoch; training resumes instantly after any interruption.
- **Data Sanitization**: Advanced filtering for corrupted images or missing labels in large-scale datasets.
- **Live Dashboard**: Real-time logging of loss, F1-score, and AUROC metrics.

---

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tanisinghal0209/Medical-Triage.git
   cd Medical-Triage
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch torchvision pandas numpy scikit-learn flask transformers pillow tqdm matplotlib seaborn
   ```

3. **Data Configuration**:
   Ensure your NIH dataset is located at: `/Users/tanishasinghal/Downloads/archive (2)`.
   *You can change this path in `train_pipeline.py`.*

---

## 🖥️ Usage Guide

### 📈 Training the Model
To start or resume the multi-label training process:
```bash
python train_pipeline.py
```
- **Pro-tip**: You can adjust `PERCENT_USED` in the script to train on a smaller subset (e.g., 0.1 for 10%) for rapid testing.

### 🌐 Running the Web Application
To launch the interactive triage dashboard:
```bash
python app.py
```
- Access the UI at: **`http://localhost:5050`**
- The app automatically loads the `best_model.pth` from your training runs.

---

## 📊 Technical Accomplishments

| Feature | Implementation | Benefit |
|---|---|---|
| **GPU Engine** | PyTorch MPS (Apple Silicon) | 8-10x faster training on Mac |
| **Model** | MobileNetV2 | Reduced memory footprint vs ResNet50 |
| **Resiliency** | Auto-Resume Checkpoints | Protection against system crashes |
| **Explainability** | Grad-CAM | Clinical trust through visual heatmaps |

---

## 📜 License
This project is licensed under the MIT License.

## 🤝 Acknowledgments
- **Dataset**: NIH Clinical Center for the ChestX-ray14 dataset.
- **Backbone**: Torchvision for pretrained model architectures.
