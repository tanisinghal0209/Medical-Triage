"""
================================================================================
  STEP 6 — MULTI-MODAL FUSION: TRIAGE URGENCY PREDICTION
  Chest X-Ray Multi-Modal Pipeline
  Module: Fuse image + NLP outputs → triage urgency score (0–1)

  Inputs:  Image model probabilities + NER entities + (optional) summary
  Output:  predict_triage(image_output, text_output) → float score [0, 1]
================================================================================

ARCHITECTURE:
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │  Image Model │   │  NER Entities │   │  Text Summary│
  │  14 disease  │   │  4 categories │   │  (optional)  │
  │  probabilities│  │  counts+conf  │   │  embedding   │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                   │                   │
    ┌────▼────┐        ┌────▼────┐         ┌────▼────┐
    │Normalize│        │Feature  │         │Normalize│
    │ [0,1]   │        │Engineer │         │ [0,1]   │
    └────┬────┘        └────┬────┘         └────┬────┘
         │                   │                   │
         └───────────┬───────┴───────────────────┘
                     │
              ┌──────▼──────┐
              │ Concatenate │
              │ Feature Vec │
              └──────┬──────┘
                     │
           ┌─────────▼─────────┐
           │ Logistic Regression│   ← interpretable
           │   OR Simple MLP   │   ← more expressive
           └─────────┬─────────┘
                     │
              ┌──────▼──────┐
              │ Triage Score │
              │   [0, 1]    │
              └─────────────┘

DEPENDS ON:
  - chest_xray_model.py  (Step 2) — disease probabilities
  - clinical_ner.py      (Step 4) — entity extraction
  - clinical_summarizer.py (Step 5) — text summaries (optional)

QUICK START:
  from triage_fusion import predict_triage, TriageFusionModel
  score = predict_triage(image_probs, ner_entities)
================================================================================
"""

# ==============================================================================
# SECTION 0 — IMPORTS
# ==============================================================================

import json
import copy
import logging
import warnings
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, mean_absolute_error,
    classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, auc,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1 — CONFIGURATION
# ==============================================================================

# Disease classes from the image model (Step 2)
DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

# Clinical severity weights — used to map disease probabilities to urgency.
# Higher weight = more clinically urgent finding.
# Based on standard emergency medicine triage protocols.
DISEASE_SEVERITY: Dict[str, float] = {
    "Pneumothorax":       1.0,   # Tension PTX is life-threatening
    "Pneumonia":          0.8,   # Can be severe, esp. in elderly/immunocompromised
    "Edema":              0.85,  # Pulmonary edema → acute heart failure
    "Consolidation":      0.75,  # Active infection marker
    "Effusion":           0.7,   # Large effusions compromise breathing
    "Cardiomegaly":       0.65,  # Heart failure indicator
    "Mass":               0.6,   # Possible malignancy
    "Infiltration":       0.6,   # Infection/inflammation
    "Atelectasis":        0.5,   # Lung collapse, variable severity
    "Emphysema":          0.5,   # Chronic, but exacerbations are urgent
    "Nodule":             0.4,   # Usually incidental, needs follow-up
    "Fibrosis":           0.45,  # Chronic progressive
    "Pleural_Thickening": 0.3,   # Usually chronic
    "Hernia":             0.25,  # Rarely urgent unless strangulated
}

# Symptom severity mapping for NER-detected symptoms
SYMPTOM_SEVERITY: Dict[str, float] = {
    "shortness of breath": 0.9, "dyspnea": 0.9, "hemoptysis": 0.95,
    "chest pain": 0.85, "syncope": 0.9, "seizure": 0.95,
    "altered mental status": 0.95, "confusion": 0.8,
    "fever": 0.6, "chills": 0.5, "night sweats": 0.5,
    "cough": 0.4, "productive cough": 0.5, "dry cough": 0.35,
    "wheezing": 0.55, "stridor": 0.85,
    "palpitations": 0.6, "edema": 0.55, "dizziness": 0.5,
    "nausea": 0.3, "vomiting": 0.4, "fatigue": 0.3,
    "weight loss": 0.45, "pain": 0.5, "headache": 0.35,
}

# NER entity categories
NER_CATEGORIES = ["symptoms", "diseases", "medications", "risk_factors"]

# Feature dimensions
N_IMAGE_FEATURES = len(DISEASE_CLASSES)           # 14
N_IMAGE_DERIVED = 5                                # max, mean, severity, n_positive, entropy
N_NER_FEATURES = 4 + 4 + 4                        # counts + avg_conf + max_conf per category
N_NER_DERIVED = 4                                  # symptom_severity, risk_burden, poly_med, disease_load
N_SUMMARY_FEATURES = 8                            # optional text summary embedding features
TOTAL_FEATURE_DIM = N_IMAGE_FEATURES + N_IMAGE_DERIVED + N_NER_FEATURES + N_NER_DERIVED  # 35 (base)

# Clinical urgency keywords used for summary embedding
URGENCY_KEYWORDS = {
    "critical", "emergent", "emergency", "urgent", "acute", "severe", "life-threatening",
    "stat", "immediate", "deteriorating", "unstable", "decompensated", "shock",
    "respiratory failure", "cardiac arrest", "intubation", "code blue",
}
CLINICAL_NEGATION_CUES = {
    "no", "not", "none", "without", "denies", "denied", "negative",
    "absent", "unremarkable", "normal", "resolved", "improved",
}
MEDICAL_TERM_MARKERS = {
    "diagnosis", "prognosis", "treatment", "therapy", "examination",
    "imaging", "laboratory", "assessment", "impression", "finding",
    "pathology", "biopsy", "procedure", "surgery", "medication",
}


# ==============================================================================
# SECTION 2 — FEATURE ENGINEERING
# ==============================================================================

class FeatureEngineer:
    """
    Converts raw multi-modal inputs into a normalized numerical feature vector.

    Feature groups:
      1. Image features  (14): Raw disease probabilities [0, 1]
      2. Image derived   (5):  max_prob, mean_prob, severity_score, n_positive, entropy
      3. NER counts      (4):  Number of entities per category
      4. NER confidence  (8):  Avg + max confidence per category
      5. NER derived     (4):  symptom_severity, risk_burden, polypharmacy, disease_load

    Total: 35 features → concatenated into a single vector
    """

    def __init__(self, disease_threshold: float = 0.5):
        """
        Args:
            disease_threshold: Probability cutoff for considering a disease "positive".
        """
        self.threshold = disease_threshold
        self.scaler = StandardScaler()
        self._is_fitted = False

    # ── Image features ────────────────────────────────────────────────────────

    def extract_image_features(
        self, image_probs: Dict[str, float]
    ) -> np.ndarray:
        """
        Convert disease probability dict → numerical feature vector.

        Args:
            image_probs: {disease_name: probability} from predict_image().
                         Can also be a list of 14 floats in DISEASE_CLASSES order.

        Returns:
            Feature vector of shape [N_IMAGE_FEATURES + N_IMAGE_DERIVED]
        """
        # ── Parse input format ────────────────────────────────────────────────
        if isinstance(image_probs, dict):
            probs = np.array([
                float(image_probs.get(cls, 0.0))
                for cls in DISEASE_CLASSES
            ], dtype=np.float32)
        elif isinstance(image_probs, (list, np.ndarray)):
            probs = np.array(image_probs, dtype=np.float32)[:N_IMAGE_FEATURES]
            # Pad if shorter than expected
            if len(probs) < N_IMAGE_FEATURES:
                probs = np.pad(probs, (0, N_IMAGE_FEATURES - len(probs)))
        else:
            probs = np.zeros(N_IMAGE_FEATURES, dtype=np.float32)

        # ── Derived features ──────────────────────────────────────────────────
        max_prob = float(probs.max()) if len(probs) > 0 else 0.0
        mean_prob = float(probs.mean()) if len(probs) > 0 else 0.0
        n_positive = int((probs >= self.threshold).sum())

        # Severity-weighted score: Σ(prob_i × severity_i) / Σ(severity_i)
        severity_weights = np.array([
            DISEASE_SEVERITY.get(cls, 0.5) for cls in DISEASE_CLASSES
        ], dtype=np.float32)
        severity_score = float(
            np.dot(probs, severity_weights) / (severity_weights.sum() + 1e-8)
        )

        # Prediction entropy (uncertainty measure)
        # Higher entropy → model is uncertain → may need human review
        p_clipped = np.clip(probs, 1e-8, 1 - 1e-8)
        entropy = float(-np.mean(
            p_clipped * np.log(p_clipped) + (1 - p_clipped) * np.log(1 - p_clipped)
        ))

        derived = np.array([
            max_prob, mean_prob, severity_score, n_positive / N_IMAGE_FEATURES, entropy
        ], dtype=np.float32)

        return np.concatenate([probs, derived])

    # ── NER features ──────────────────────────────────────────────────────────

    def extract_ner_features(
        self, ner_entities: Dict[str, List[Dict]]
    ) -> np.ndarray:
        """
        Convert NER entity dict → numerical feature vector.

        Args:
            ner_entities: Output from extract_entities():
                {
                    "symptoms":     [{"text": ..., "confidence": ...}, ...],
                    "diseases":     [...],
                    "medications":  [...],
                    "risk_factors": [...]
                }

        Returns:
            Feature vector of shape [N_NER_FEATURES + N_NER_DERIVED]
        """
        counts = []     # 4 values: entity count per category
        avg_confs = []  # 4 values: average confidence per category
        max_confs = []  # 4 values: max confidence per category

        for category in NER_CATEGORIES:
            entities = ner_entities.get(category, [])
            n = len(entities)
            counts.append(n)

            if n > 0:
                confidences = [
                    ent.get("confidence", 0.5) for ent in entities
                ]
                avg_confs.append(float(np.mean(confidences)))
                max_confs.append(float(np.max(confidences)))
            else:
                avg_confs.append(0.0)
                max_confs.append(0.0)

        # Normalize counts (cap at 10 to prevent outlier domination)
        counts_norm = [min(c, 10) / 10.0 for c in counts]

        # ── Derived NER features ──────────────────────────────────────────────

        # 1. Symptom severity score
        symptom_texts = [
            ent.get("text", "").lower()
            for ent in ner_entities.get("symptoms", [])
        ]
        symptom_severity = 0.0
        if symptom_texts:
            severities = [
                SYMPTOM_SEVERITY.get(s, 0.4) for s in symptom_texts
            ]
            symptom_severity = float(np.max(severities))  # Worst symptom drives urgency

        # 2. Risk factor burden (more risk factors → higher urgency)
        risk_count = len(ner_entities.get("risk_factors", []))
        risk_burden = min(risk_count, 5) / 5.0

        # 3. Polypharmacy indicator (many meds → complex patient)
        med_count = len(ner_entities.get("medications", []))
        polypharmacy = min(med_count, 8) / 8.0

        # 4. Disease load (number of co-existing conditions)
        disease_count = len(ner_entities.get("diseases", []))
        disease_load = min(disease_count, 6) / 6.0

        derived = np.array([
            symptom_severity, risk_burden, polypharmacy, disease_load
        ], dtype=np.float32)

        base = np.array(
            counts_norm + avg_confs + max_confs,
            dtype=np.float32,
        )

        return np.concatenate([base, derived])

    # ── Summary embedding features (optional) ─────────────────────────────────

    def extract_summary_embedding(
        self, summary_text: str
    ) -> np.ndarray:
        """
        Convert clinical summary text → lightweight feature vector.

        Uses keyword-based features instead of heavy transformer embeddings
        to keep the model simple and interpretable.

        Features (8 total):
          1. Normalized text length (chars / 1000, capped)
          2. Sentence count (normalized)
          3. Urgency keyword density
          4. Clinical severity keyword score
          5. Negation cue density
          6. Average word length (normalized)
          7. Medical term density
          8. Lexical diversity (unique words / total words)

        Args:
            summary_text: Clinical summary string.

        Returns:
            Feature vector of shape [N_SUMMARY_FEATURES] = [8]
        """
        if not summary_text or not summary_text.strip():
            return np.zeros(N_SUMMARY_FEATURES, dtype=np.float32)

        text = summary_text.strip().lower()
        words = re.findall(r'\b\w+\b', text)
        n_words = max(len(words), 1)
        word_set = set(words)

        # 1. Normalized text length (cap at 2000 chars)
        text_len = min(len(text), 2000) / 2000.0

        # 2. Sentence count (normalized, cap at 10)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        n_sentences = min(len(sentences), 10) / 10.0

        # 3. Urgency keyword density
        urgency_count = sum(1 for w in words if w in URGENCY_KEYWORDS)
        # Also check multi-word urgency phrases
        for phrase in URGENCY_KEYWORDS:
            if ' ' in phrase and phrase in text:
                urgency_count += 1
        urgency_density = min(urgency_count / n_words, 1.0)

        # 4. Clinical severity score (weighted match against severity keywords)
        severity_hits = 0
        for symptom, sev in SYMPTOM_SEVERITY.items():
            if symptom in text:
                severity_hits += sev
        severity_score = min(severity_hits / 5.0, 1.0)  # Normalize

        # 5. Negation cue density
        negation_count = sum(1 for w in words if w in CLINICAL_NEGATION_CUES)
        negation_density = min(negation_count / n_words, 1.0)

        # 6. Average word length (normalized: typical avg is 4-6)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        avg_word_len_norm = min(avg_word_len / 10.0, 1.0)

        # 7. Medical term density
        medical_count = sum(1 for w in words if w in MEDICAL_TERM_MARKERS)
        # Also check against disease/symptom terms
        medical_count += sum(1 for w in word_set if w in {
            s.lower() for s in SYMPTOM_SEVERITY
        })
        medical_density = min(medical_count / n_words, 1.0)

        # 8. Lexical diversity (type-token ratio)
        lexical_diversity = len(word_set) / n_words if n_words > 0 else 0

        return np.array([
            text_len, n_sentences, urgency_density, severity_score,
            negation_density, avg_word_len_norm, medical_density,
            lexical_diversity,
        ], dtype=np.float32)

    # ── Combined feature vector ───────────────────────────────────────────────

    def build_feature_vector(
        self,
        image_probs: Dict[str, float],
        ner_entities: Dict[str, List[Dict]],
        summary_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Build the complete fused feature vector from all modalities.

        Args:
            image_probs  : Disease probabilities from image model.
            ner_entities : Entity dict from NER pipeline.
            summary_text : Optional clinical summary text.

        Returns:
            Numpy array of shape [TOTAL_FEATURE_DIM] or
            [TOTAL_FEATURE_DIM + N_SUMMARY_FEATURES] if summary provided.
        """
        img_feat = self.extract_image_features(image_probs)
        ner_feat = self.extract_ner_features(ner_entities)
        parts = [img_feat, ner_feat]

        if summary_text is not None:
            summary_feat = self.extract_summary_embedding(summary_text)
            parts.append(summary_feat)

        return np.concatenate(parts)

    def build_feature_matrix(
        self,
        samples: List[Dict],
    ) -> np.ndarray:
        """
        Build feature matrix for multiple samples.

        Args:
            samples: List of dicts with "image_probs", "ner_entities",
                     and optionally "summary_text" keys.

        Returns:
            Numpy array of shape [N_samples, feature_dim]
        """
        vectors = []
        for sample in samples:
            vec = self.build_feature_vector(
                sample.get("image_probs", {}),
                sample.get("ner_entities", {}),
                sample.get("summary_text", None),
            )
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    # ── Normalization ─────────────────────────────────────────────────────────

    def fit_scaler(self, X: np.ndarray) -> None:
        """Fit the StandardScaler on training data."""
        self.scaler.fit(X)
        # Replace zero variance with 1.0 to avoid NaN from division by zero
        self.scaler.scale_ = np.where(
            self.scaler.scale_ == 0, 1.0, self.scaler.scale_
        )
        self._is_fitted = True

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted normalization."""
        if not self._is_fitted:
            log.warning("Scaler not fitted — returning raw features.")
            return X
        result = self.scaler.transform(X)
        # Safety: replace any remaining NaN/inf with 0
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def get_feature_names(self, include_summary: bool = False) -> List[str]:
        """Return human-readable names for each feature dimension."""
        names = []
        # Image raw probs
        names.extend([f"img_{cls.lower()}" for cls in DISEASE_CLASSES])
        # Image derived
        names.extend(["img_max_prob", "img_mean_prob", "img_severity",
                       "img_n_positive_ratio", "img_entropy"])
        # NER counts
        names.extend([f"ner_{cat}_count" for cat in NER_CATEGORIES])
        # NER avg confidence
        names.extend([f"ner_{cat}_avg_conf" for cat in NER_CATEGORIES])
        # NER max confidence
        names.extend([f"ner_{cat}_max_conf" for cat in NER_CATEGORIES])
        # NER derived
        names.extend(["ner_symptom_severity", "ner_risk_burden",
                       "ner_polypharmacy", "ner_disease_load"])
        # Summary features (optional)
        if include_summary:
            names.extend([
                "summary_text_length", "summary_n_sentences",
                "summary_urgency_density", "summary_severity_score",
                "summary_negation_density", "summary_avg_word_len",
                "summary_medical_density", "summary_lexical_diversity",
            ])
        return names


# ==============================================================================
# SECTION 3 — TRIAGE MODELS
# ==============================================================================

# ── 3A: Logistic Regression (interpretable baseline) ─────────────────────────

class LogisticTriageModel:
    """
    Logistic Regression for triage prediction.
    Highly interpretable — feature coefficients show what drives urgency.
    """

    def __init__(self, C: float = 1.0):
        self.model = LogisticRegression(
            C=C, max_iter=1000, solver="lbfgs",
            class_weight="balanced",
        )
        self._is_trained = False
        self._trained_dim = None
        self.feature_engineer = FeatureEngineer()

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train on feature matrix X and binary labels y."""
        self.feature_engineer.fit_scaler(X)
        X_norm = self.feature_engineer.normalize(X)

        self.model.fit(X_norm, y)
        self._is_trained = True
        self._trained_dim = X.shape[1]

        # Training metrics
        y_pred = self.model.predict(X_norm)
        y_prob = self.model.predict_proba(X_norm)[:, 1]

        metrics = self._compute_metrics(y, y_pred, y_prob)
        log.info(f"Logistic Regression trained | Acc={metrics['accuracy']:.3f} | "
                 f"AUC={metrics['auroc']:.3f} | F1={metrics['f1']:.3f}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return triage scores [0, 1] for feature matrix X."""
        X_norm = self.feature_engineer.normalize(X)
        return self.model.predict_proba(X_norm)[:, 1]

    def predict_single(
        self, image_probs: Dict, ner_entities: Dict,
        summary_text: Optional[str] = None,
    ) -> float:
        """Predict triage score for a single patient."""
        # Only include summary if model was trained with summary features
        use_summary = summary_text
        if self._trained_dim is not None and self._trained_dim == TOTAL_FEATURE_DIM:
            use_summary = None
        vec = self.feature_engineer.build_feature_vector(
            image_probs, ner_entities, use_summary
        )
        X = vec.reshape(1, -1)
        return float(self.predict(X)[0])

    def get_feature_importance(self) -> List[Tuple[str, float]]:
        """Return feature names + coefficients sorted by importance."""
        if not self._is_trained:
            return []
        n_coefs = len(self.model.coef_[0])
        include_summary = n_coefs > TOTAL_FEATURE_DIM
        names = self.feature_engineer.get_feature_names(include_summary=include_summary)
        coefs = self.model.coef_[0]
        # Handle dimension mismatch gracefully
        n = min(len(names), len(coefs))
        importance = sorted(
            zip(names[:n], coefs[:n]),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return importance

    @staticmethod
    def _compute_metrics(y_true, y_pred, y_prob) -> Dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "auroc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "mae": mean_absolute_error(y_true, y_prob),
        }


# ── 3B: Simple MLP (more expressive) ─────────────────────────────────────────

class MLPTriageNet(nn.Module):
    """
    2-layer MLP for triage score prediction.

    Architecture:
        Input [31] → Linear(31, 64) → ReLU → Dropout(0.3)
                   → Linear(64, 32) → ReLU → Dropout(0.2)
                   → Linear(32, 1) → Sigmoid → score [0, 1]
    """

    def __init__(self, input_dim: int = TOTAL_FEATURE_DIM, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout * 0.67),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPTriageModel:
    """
    Neural network triage model with training loop.
    More expressive than logistic regression for complex feature interactions.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        device: str = "auto",
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.feature_engineer = FeatureEngineer()

        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = None  # Created dynamically based on input_dim
        self._input_dim = None
        self._is_trained = False

    def _create_model(self, input_dim: int) -> None:
        """Create MLP with the correct input dimension."""
        self._input_dim = input_dim
        self.model = MLPTriageNet(input_dim=input_dim).to(self.device)

    def train(
        self, X: np.ndarray, y: np.ndarray, val_split: float = 0.2
    ) -> Dict:
        """Train the MLP on feature matrix X and labels y ∈ {0, 1}."""
        # Create model with correct input dimension
        self._create_model(X.shape[1])

        self.feature_engineer.fit_scaler(X)
        X_norm = self.feature_engineer.normalize(X)

        # ── Create dataset ────────────────────────────────────────────────────
        X_t = torch.tensor(X_norm, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)

        n_val = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        # ── Training loop ─────────────────────────────────────────────────────
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.BCELoss()
        best_val_loss = float("inf")
        best_weights = None

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    val_loss += criterion(pred, yb).item()
            val_loss /= max(len(val_loader), 1)
            self.model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(self.model.state_dict())

            if epoch % 10 == 0 or epoch == 1:
                log.info(f"  Epoch {epoch:3d}/{self.epochs} | "
                         f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # Restore best weights
        if best_weights:
            self.model.load_state_dict(best_weights)
        self.model.eval()
        self._is_trained = True

        # Final metrics
        y_prob = self.predict(X)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "auroc": roc_auc_score(y, y_prob) if len(set(y)) > 1 else 0.0,
            "f1": f1_score(y, y_pred, zero_division=0),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "mae": mean_absolute_error(y, y_prob),
        }
        log.info(f"MLP trained | Acc={metrics['accuracy']:.3f} | "
                 f"AUC={metrics['auroc']:.3f} | F1={metrics['f1']:.3f}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return triage scores for feature matrix X."""
        X_norm = self.feature_engineer.normalize(X)
        X_t = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            scores = self.model(X_t).cpu().numpy()
        return scores

    def predict_single(
        self, image_probs: Dict, ner_entities: Dict,
        summary_text: Optional[str] = None,
    ) -> float:
        """Predict triage score for a single patient."""
        # Only include summary if model was trained with summary features
        use_summary = summary_text
        if self._input_dim is not None and self._input_dim == TOTAL_FEATURE_DIM:
            use_summary = None
        vec = self.feature_engineer.build_feature_vector(
            image_probs, ner_entities, use_summary
        )
        return float(self.predict(vec.reshape(1, -1))[0])


# ==============================================================================
# SECTION 4 — RULE-BASED URGENCY ESTIMATOR (NO TRAINING NEEDED)
# ==============================================================================

def estimate_urgency_rule_based(
    image_probs: Dict[str, float],
    ner_entities: Dict[str, List[Dict]],
    threshold: float = 0.5,
) -> Dict:
    """
    Rule-based triage urgency estimation — works without training data.

    Combines:
      1. Severity-weighted image probability score
      2. Symptom severity from NER
      3. Risk factor count
      4. Critical finding detection

    Returns dict with score and reasoning.

    Example:
        >>> result = estimate_urgency_rule_based(image_probs, ner_entities)
        >>> print(f"Score: {result['score']:.2f} | Level: {result['level']}")
    """
    fe = FeatureEngineer(disease_threshold=threshold)

    # ── Image component (0–1) ─────────────────────────────────────────────────
    # Dynamically determine classes from input or fall back to defaults
    classes = list(image_probs.keys()) if image_probs else DISEASE_CLASSES
    
    probs = np.array([float(image_probs.get(cls, 0.0)) for cls in classes])
    severity_w = np.array([DISEASE_SEVERITY.get(cls, 0.5) for cls in classes])
    
    image_score = 0.0
    if severity_w.sum() > 0:
        image_score = float(np.dot(probs, severity_w) / severity_w.sum())

    # Critical findings
    critical_findings = []
    for cls in ["Pneumothorax", "Edema", "Pneumonia"]:
        p = image_probs.get(cls, 0.0)
        if p >= threshold:
            critical_findings.append(f"{cls} ({p:.0%})")

    # ── NER component (0–1) ───────────────────────────────────────────────────
    symptom_texts = [e.get("text", "").lower() for e in ner_entities.get("symptoms", [])]
    symptom_score = max(
        [SYMPTOM_SEVERITY.get(s, 0.4) for s in symptom_texts],
        default=0.0,
    )
    n_diseases = len(ner_entities.get("diseases", []))
    n_risks = len(ner_entities.get("risk_factors", []))
    n_meds = len(ner_entities.get("medications", []))
    comorbidity_score = min(n_diseases + n_risks, 6) / 6.0

    # ── Fused score ───────────────────────────────────────────────────────────
    # Weighted combination: image evidence is strongest signal
    score = (
        0.45 * image_score +
        0.30 * symptom_score +
        0.15 * comorbidity_score +
        0.10 * (1.0 if critical_findings else 0.0)
    )
    score = float(np.clip(score, 0.0, 1.0))

    # ── Triage level ──────────────────────────────────────────────────────────
    if score >= 0.8:
        level, label = 1, "🔴 EMERGENT"
    elif score >= 0.6:
        level, label = 2, "🟠 URGENT"
    elif score >= 0.4:
        level, label = 3, "🟡 SEMI-URGENT"
    elif score >= 0.2:
        level, label = 4, "🟢 NON-URGENT"
    else:
        level, label = 5, "⚪ ROUTINE"

    return {
        "score": score,
        "level": level,
        "label": label,
        "components": {
            "image_score": round(image_score, 4),
            "symptom_score": round(symptom_score, 4),
            "comorbidity_score": round(comorbidity_score, 4),
            "critical_findings": critical_findings,
        },
        "details": {
            "n_symptoms": len(symptom_texts),
            "n_diseases": n_diseases,
            "n_medications": n_meds,
            "n_risk_factors": n_risks,
            "top_disease": max(image_probs, key=image_probs.get) if image_probs else "none",
        },
    }


# ==============================================================================
# SECTION 5 — MAIN API: predict_triage()
# ==============================================================================

# Module-level singleton
_triage_model = None


def predict_triage(
    image_output: Union[Dict[str, float], List[float]],
    text_output: Dict[str, List[Dict]],
    model: Optional[Union[LogisticTriageModel, MLPTriageModel]] = None,
    mode: str = "rule_based",
) -> Union[float, Dict]:
    """
    Predict triage urgency from multi-modal inputs.

    This is the primary API function. It accepts outputs from the image model
    (Step 2) and NER pipeline (Step 4) and returns an urgency score.

    Args:
        image_output : Disease probabilities from predict_image().
                       Dict {disease: prob} or list of 14 floats.
        text_output  : Entity dict from extract_entities().
                       {"symptoms": [...], "diseases": [...], ...}
        model        : Trained LogisticTriageModel or MLPTriageModel.
                       If None, uses rule-based estimation.
        mode         : "rule_based" (default, no training needed),
                       "logistic", or "mlp" (require trained model).

    Returns:
        If mode="rule_based": Dict with score, level, label, and reasoning.
        If mode="logistic"/"mlp": float score in [0, 1].

    Example:
        >>> from triage_fusion import predict_triage
        >>> image_probs = {"Pneumonia": 0.87, "Effusion": 0.45, ...}
        >>> ner_entities = {"symptoms": [{"text": "fever", ...}], ...}
        >>> result = predict_triage(image_probs, ner_entities)
        >>> print(f"Urgency: {result['score']:.2f} — {result['label']}")
    """
    if mode == "rule_based" or model is None:
        # Convert list input to dict
        if isinstance(image_output, (list, np.ndarray)):
            image_dict = {
                cls: float(p) for cls, p in
                zip(DISEASE_CLASSES, image_output)
            }
        else:
            image_dict = image_output

        return estimate_urgency_rule_based(image_dict, text_output)

    # Trained model prediction
    return model.predict_single(image_output, text_output)


# ==============================================================================
# SECTION 6 — SYNTHETIC DATA GENERATOR (FOR TRAINING/TESTING)
# ==============================================================================

def generate_synthetic_dataset(
    n_samples: int = 500,
    seed: int = 42,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Generate synthetic training data for the triage model.

    Creates realistic patient scenarios with correlated image findings,
    NER entities, and urgency labels. Used for development and testing
    when real labeled data is unavailable.

    Returns:
        (samples, labels)
        - samples: List of dicts with "image_probs" and "ner_entities"
        - labels: numpy array of shape [n_samples] with values in {0, 1}
    """
    rng = np.random.RandomState(seed)
    samples = []
    labels = []

    all_symptoms = list(SYMPTOM_SEVERITY.keys())
    all_diseases = list(DISEASE_SEVERITY.keys())
    all_meds = ["amoxicillin", "levofloxacin", "albuterol", "furosemide",
                "metoprolol", "prednisone", "lisinopril", "aspirin"]
    all_risks = ["smoking", "diabetes", "hypertension", "obesity",
                 "elderly", "immunocompromised", "former smoker"]

    for _ in range(n_samples):
        # Decide urgency level first, then generate correlated features
        is_urgent = rng.random() < 0.4  # 40% urgent cases

        # ── Generate image probabilities ──────────────────────────────────────
        image_probs = {}
        for cls in DISEASE_CLASSES:
            if is_urgent:
                # Urgent: higher probs for severe diseases
                sev = DISEASE_SEVERITY[cls]
                base = sev * 0.5 + rng.beta(2, 3) * 0.5
                image_probs[cls] = float(np.clip(base + rng.normal(0, 0.1), 0, 1))
            else:
                # Non-urgent: generally low probabilities
                image_probs[cls] = float(np.clip(rng.beta(1, 5), 0, 1))

        # ── Generate NER entities ─────────────────────────────────────────────
        if is_urgent:
            n_symptoms = rng.randint(2, 5)
            n_diseases = rng.randint(1, 4)
            n_meds = rng.randint(2, 5)
            n_risks = rng.randint(1, 4)
        else:
            n_symptoms = rng.randint(0, 2)
            n_diseases = rng.randint(0, 2)
            n_meds = rng.randint(0, 3)
            n_risks = rng.randint(0, 2)

        def make_entities(pool, n, conf_range):
            chosen = rng.choice(pool, size=min(n, len(pool)), replace=False)
            return [
                {"text": t, "start": 0, "end": len(t),
                 "confidence": float(rng.uniform(*conf_range))}
                for t in chosen
            ]

        ner_entities = {
            "symptoms": make_entities(all_symptoms, n_symptoms, (0.6, 0.95)),
            "diseases": make_entities(all_diseases, n_diseases, (0.5, 0.9)),
            "medications": make_entities(all_meds, n_meds, (0.7, 0.95)),
            "risk_factors": make_entities(all_risks, n_risks, (0.6, 0.9)),
        }

        samples.append({
            "image_probs": image_probs,
            "ner_entities": ner_entities,
        })
        labels.append(1 if is_urgent else 0)

    return samples, np.array(labels, dtype=np.int32)


# ==============================================================================
# SECTION 7 — TRAINING PIPELINE
# ==============================================================================

def train_triage_pipeline(
    model_type: str = "logistic",
    n_synthetic: int = 500,
    seed: int = 42,
    test_size: float = 0.2,
    save_path: Optional[str] = None,
) -> Tuple[Union[LogisticTriageModel, MLPTriageModel], Dict]:
    """
    End-to-end training pipeline for the triage model.

    Steps:
      1. Generate synthetic data (or use provided real data)
      2. Extract features from all samples
      3. Stratified train/test split
      4. Train the selected model on training set
      5. Evaluate on held-out test set
      6. Show feature importance (logistic only)
      7. Optionally save trained model

    Args:
        model_type  : "logistic" or "mlp".
        n_synthetic : Number of synthetic samples.
        seed        : Random seed.
        test_size   : Fraction of data for test set (default 0.2).
        save_path   : If provided, save trained model to this path.

    Returns:
        (trained_model, test_metrics_dict)
    """
    log.info("=" * 60)
    log.info(f"  TRIAGE FUSION TRAINING PIPELINE ({model_type.upper()})")
    log.info("=" * 60)

    # ── Step 1: Generate data ─────────────────────────────────────────────────
    log.info(f"\n  Step 1: Generating {n_synthetic} synthetic samples…")
    samples, labels = generate_synthetic_dataset(n_synthetic, seed)
    log.info(f"  Labels: {(labels == 1).sum()} urgent, {(labels == 0).sum()} non-urgent")

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    log.info("  Step 2: Extracting features…")
    if model_type == "logistic":
        model = LogisticTriageModel()
    else:
        model = MLPTriageModel(epochs=50)

    X = model.feature_engineer.build_feature_matrix(samples)
    log.info(f"  Feature matrix shape: {X.shape}")

    # ── Step 3: Stratified train/test split ────────────────────────────────────
    log.info(f"  Step 3: Splitting data (test_size={test_size})…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=seed, stratify=labels,
    )
    log.info(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # ── Step 4: Train on training set only ────────────────────────────────────
    log.info(f"  Step 4: Training {model_type} model…")
    train_metrics = model.train(X_train, y_train)
    log.info("\n  Training Metrics:")
    for k, v in train_metrics.items():
        log.info(f"    {k:<12s}: {v:.4f}")

    # ── Step 5: Evaluate on held-out test set ─────────────────────────────────
    log.info("\n  Step 5: Evaluating on test set…")
    y_prob_test = model.predict(X_test)
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred_test),
        "auroc": roc_auc_score(y_test, y_prob_test) if len(set(y_test)) > 1 else 0.0,
        "f1": f1_score(y_test, y_pred_test, zero_division=0),
        "precision": precision_score(y_test, y_pred_test, zero_division=0),
        "recall": recall_score(y_test, y_pred_test, zero_division=0),
        "mae": mean_absolute_error(y_test, y_prob_test),
    }
    log.info("\n  Test Metrics (held-out):")
    for k, v in test_metrics.items():
        log.info(f"    {k:<12s}: {v:.4f}")

    # ── Step 6: Feature importance (logistic only) ────────────────────────────
    if model_type == "logistic":
        log.info("\n  Feature Importance (top 10):")
        importance = model.get_feature_importance()
        for name, coef in importance[:10]:
            bar = "+" * int(abs(coef) * 5) if coef > 0 else "-" * int(abs(coef) * 5)
            log.info(f"    {name:<30s}: {coef:+.4f}  {bar}")

    # ── Step 7: Save model ────────────────────────────────────────────────────
    if save_path:
        save_triage_model(model, save_path, model_type=model_type)
        log.info(f"  Model saved → {save_path}")

    return model, test_metrics


# ==============================================================================
# SECTION 8 — MODEL PERSISTENCE (SAVE / LOAD)
# ==============================================================================

def save_triage_model(
    model: Union[LogisticTriageModel, MLPTriageModel],
    path: str,
    model_type: str = "logistic",
) -> None:
    """
    Save a trained triage model to disk.

    Serializes: model weights, fitted scaler, model type, and feature config.

    Args:
        model      : Trained LogisticTriageModel or MLPTriageModel.
        path       : File path (e.g., "triage_model.pkl").
        model_type : "logistic" or "mlp".
    """
    save_dir = Path(path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_type": model_type,
        "scaler": model.feature_engineer.scaler,
        "scaler_fitted": model.feature_engineer._is_fitted,
        "threshold": model.feature_engineer.threshold,
    }

    if model_type == "logistic":
        payload["sklearn_model"] = model.model
    else:
        payload["mlp_state_dict"] = model.model.state_dict()
        payload["mlp_input_dim"] = model._input_dim

    with open(path, "wb") as f:
        pickle.dump(payload, f)

    log.info(f"Model saved → {path}")


def load_triage_model(
    path: str,
) -> Union[LogisticTriageModel, MLPTriageModel]:
    """
    Load a trained triage model from disk.

    Args:
        path: File path to the saved model.

    Returns:
        Restored LogisticTriageModel or MLPTriageModel ready for inference.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    model_type = payload["model_type"]

    if model_type == "logistic":
        model = LogisticTriageModel()
        model.model = payload["sklearn_model"]
        model._is_trained = True
    else:
        input_dim = payload.get("mlp_input_dim", TOTAL_FEATURE_DIM)
        model = MLPTriageModel()
        model._create_model(input_dim)
        model.model.load_state_dict(payload["mlp_state_dict"])
        model.model.eval()
        model._is_trained = True

    model.feature_engineer.scaler = payload["scaler"]
    model.feature_engineer._is_fitted = payload["scaler_fitted"]
    model.feature_engineer.threshold = payload.get("threshold", 0.5)

    log.info(f"Model loaded ← {path} (type={model_type})")
    return model


# ==============================================================================
# SECTION 9 — CROSS-VALIDATION
# ==============================================================================

def cross_validate_triage(
    model_type: str = "logistic",
    n_folds: int = 5,
    n_synthetic: int = 500,
    seed: int = 42,
) -> Dict:
    """
    Stratified k-fold cross-validation for the triage model.

    Provides robust performance estimates with mean ± std for each metric.

    Args:
        model_type  : "logistic" or "mlp".
        n_folds     : Number of CV folds.
        n_synthetic : Number of synthetic samples.
        seed        : Random seed.

    Returns:
        Dict with per-fold and aggregated metrics.
    """
    log.info("=" * 60)
    log.info(f"  {n_folds}-FOLD CROSS-VALIDATION ({model_type.upper()})")
    log.info("=" * 60)

    # Generate data once
    samples, labels = generate_synthetic_dataset(n_synthetic, seed)
    fe = FeatureEngineer()
    X = fe.build_feature_matrix(samples)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, labels), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        if model_type == "logistic":
            model = LogisticTriageModel()
        else:
            model = MLPTriageModel(epochs=30)

        model.train(X_train, y_train)
        y_prob = model.predict(X_val)
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "auroc": roc_auc_score(y_val, y_prob) if len(set(y_val)) > 1 else 0.0,
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
        }
        fold_metrics.append(metrics)
        log.info(f"  Fold {fold}/{n_folds} | Acc={metrics['accuracy']:.3f} | "
                 f"AUC={metrics['auroc']:.3f} | F1={metrics['f1']:.3f}")

    # Aggregate
    agg = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics]
        agg[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    log.info("\n  Aggregated Results (mean ± std):")
    for key, stats in agg.items():
        log.info(f"    {key:<12s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    return {"per_fold": fold_metrics, "aggregated": agg}


# ==============================================================================
# SECTION 10 — COMPREHENSIVE EVALUATION WITH PLOTS
# ==============================================================================

class EvaluationReport:
    """
    Generates a comprehensive evaluation report with visualizations.

    Includes:
      - Confusion matrix
      - ROC curve + AUC
      - Precision-Recall curve
      - Calibration analysis
      - Feature importance plot (logistic only)
      - Classification report
    """

    def __init__(self, save_dir: str = "./triage_evaluation"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model: Optional[Union[LogisticTriageModel, MLPTriageModel]] = None,
        model_name: str = "Triage Model",
    ) -> Dict:
        """
        Generate all evaluation plots and metrics.

        Args:
            y_true     : Ground truth binary labels.
            y_prob     : Predicted probabilities.
            model      : Trained model (for feature importance).
            model_name : Display name for plots.

        Returns:
            Dict with all computed metrics and file paths.
        """
        y_pred = (y_prob >= 0.5).astype(int)

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "auroc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.0,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "mae": mean_absolute_error(y_true, y_prob),
        }

        # Generate plots
        files = {}
        files["confusion_matrix"] = self._plot_confusion_matrix(y_true, y_pred, model_name)
        files["roc_curve"] = self._plot_roc_curve(y_true, y_prob, model_name)
        files["pr_curve"] = self._plot_precision_recall(y_true, y_prob, model_name)
        files["calibration"] = self._plot_calibration(y_true, y_prob, model_name)

        if model and hasattr(model, "get_feature_importance"):
            imp = model.get_feature_importance()
            if imp:
                files["feature_importance"] = self._plot_feature_importance(
                    imp, model_name
                )

        # Save metrics JSON
        report_path = self.save_dir / "evaluation_metrics.json"
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        files["metrics_json"] = str(report_path)

        # Classification report
        cls_report = classification_report(
            y_true, y_pred, target_names=["Non-Urgent", "Urgent"],
            output_dict=True, zero_division=0,
        )
        cls_path = self.save_dir / "classification_report.json"
        with open(cls_path, "w") as f:
            json.dump(cls_report, f, indent=2)
        files["classification_report"] = str(cls_path)

        log.info(f"  Evaluation report saved → {self.save_dir}")
        return {"metrics": metrics, "files": files}

    def _plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, title: str
    ) -> str:
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.colorbar(im, ax=ax)

        labels = ["Non-Urgent", "Urgent"]
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_yticklabels(labels, fontsize=11)

        # Annotate cells
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=16, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix — {title}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        path = str(self.save_dir / "confusion_matrix.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_roc_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, title: str
    ) -> str:
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color="#2980b9", lw=2.5,
                label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#2980b9")

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve — {title}", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = str(self.save_dir / "roc_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_precision_recall(
        self, y_true: np.ndarray, y_prob: np.ndarray, title: str
    ) -> str:
        """Plot and save Precision-Recall curve."""
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec, prec)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec, prec, color="#e74c3c", lw=2.5,
                label=f"AP = {pr_auc:.3f}")
        ax.fill_between(rec, prec, alpha=0.1, color="#e74c3c")

        baseline = y_true.sum() / len(y_true)
        ax.axhline(y=baseline, color="gray", ls="--", lw=1,
                    label=f"Baseline = {baseline:.2f}")

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall — {title}", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right", fontsize=11)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = str(self.save_dir / "precision_recall_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_calibration(
        self, y_true: np.ndarray, y_prob: np.ndarray, title: str,
        n_bins: int = 10,
    ) -> str:
        """Plot and save calibration curve (reliability diagram)."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means = []
        bin_true_fracs = []
        bin_counts = []

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_means.append(y_prob[mask].mean())
                bin_true_fracs.append(y_true[mask].mean())
                bin_counts.append(mask.sum())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7),
                                        gridspec_kw={"height_ratios": [3, 1]})

        # Calibration curve
        ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax1.plot(bin_means, bin_true_fracs, "o-", color="#27ae60", lw=2,
                 markersize=6, label="Model")
        ax1.set_xlabel("Mean Predicted Probability", fontsize=11)
        ax1.set_ylabel("Fraction of Positives", fontsize=11)
        ax1.set_title(f"Calibration — {title}", fontsize=13, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Histogram of predictions
        ax2.hist(y_prob, bins=n_bins, range=(0, 1), color="#3498db",
                 alpha=0.7, edgecolor="white")
        ax2.set_xlabel("Predicted Probability", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = str(self.save_dir / "calibration_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _plot_feature_importance(
        self, importance: List[Tuple[str, float]], title: str,
        top_k: int = 15,
    ) -> str:
        """Plot and save feature importance (logistic regression coefficients)."""
        top = importance[:top_k]
        names = [n for n, _ in top]
        coefs = [c for _, c in top]
        colors = ["#e74c3c" if c > 0 else "#3498db" for c in coefs]

        fig, ax = plt.subplots(figsize=(8, 6))
        y_pos = range(len(names))
        ax.barh(y_pos, coefs, color=colors, height=0.6, edgecolor="white")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Coefficient Value", fontsize=11)
        ax.set_title(f"Feature Importance — {title}", fontsize=13, fontweight="bold")
        ax.axvline(x=0, color="black", lw=0.8)
        ax.grid(True, axis="x", alpha=0.3)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e74c3c", label="Increases urgency (+)"),
            Patch(facecolor="#3498db", label="Decreases urgency (−)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        plt.tight_layout()
        path = str(self.save_dir / "feature_importance.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path


# ==============================================================================
# SECTION 11 — END-TO-END PIPELINE INTEGRATION
# ==============================================================================

def run_full_pipeline(
    image_probs: Optional[Dict[str, float]] = None,
    clinical_text: Optional[str] = None,
    ner_entities: Optional[Dict[str, List[Dict]]] = None,
    summary_text: Optional[str] = None,
    model: Optional[Union[LogisticTriageModel, MLPTriageModel]] = None,
) -> Dict:
    """
    End-to-end multi-modal fusion demonstrating the complete pipeline.

    Connects all four pipeline modules:
      1. chest_xray_model.py   → image_probs  (disease probabilities)
      2. clinical_ner.py       → ner_entities  (structured entities)
      3. clinical_summarizer.py → summary_text (optional summary)
      4. triage_fusion.py      → triage score  (urgency 0–1)

    If inputs are not provided, uses realistic demo data.

    Args:
        image_probs  : Disease probability dict from image model.
        clinical_text: Raw clinical note (used for NER if ner_entities not given).
        ner_entities : Pre-extracted NER entities.
        summary_text : Pre-generated clinical summary.
        model        : Trained triage model (uses rule-based if None).

    Returns:
        Dict with all intermediate and final results.
    """
    log.info("=" * 60)
    log.info("  END-TO-END MULTI-MODAL FUSION PIPELINE")
    log.info("=" * 60)

    results = {}

    # ── Step 1: Image model output ────────────────────────────────────────────
    if image_probs is None:
        log.info("\n  Step 1: Using demo image probabilities…")
        image_probs = {
            "Pneumonia": 0.87, "Effusion": 0.45, "Edema": 0.62,
            "Cardiomegaly": 0.33, "Consolidation": 0.41,
            "Infiltration": 0.55, "Atelectasis": 0.21,
            "Pneumothorax": 0.05, "Mass": 0.08, "Nodule": 0.12,
            "Emphysema": 0.15, "Fibrosis": 0.09,
            "Pleural_Thickening": 0.07, "Hernia": 0.03,
        }
    else:
        log.info("\n  Step 1: Using provided image probabilities…")

    results["image_probs"] = image_probs
    top_finding = max(image_probs, key=image_probs.get)
    log.info(f"  Top finding: {top_finding} ({image_probs[top_finding]:.0%})")

    # ── Step 2: NER entities ──────────────────────────────────────────────────
    if ner_entities is None:
        if clinical_text:
            log.info("  Step 2: Extracting NER entities from clinical text…")
            try:
                from clinical_ner import extract_entities
                ner_entities = extract_entities(clinical_text)
                log.info(f"  Extracted: {sum(len(v) for v in ner_entities.values())} entities")
            except ImportError:
                log.warning("  clinical_ner not available, using demo entities")
                ner_entities = _demo_ner_entities()
        else:
            log.info("  Step 2: Using demo NER entities…")
            ner_entities = _demo_ner_entities()
    else:
        log.info("  Step 2: Using provided NER entities…")

    results["ner_entities"] = ner_entities
    for cat, ents in ner_entities.items():
        if ents:
            log.info(f"    {cat}: {[e['text'] for e in ents]}")

    # ── Step 3: Summary (optional) ────────────────────────────────────────────
    if summary_text is None and clinical_text:
        log.info("  Step 3: Generating clinical summary…")
        try:
            from clinical_summarizer import summarize_text
            summary_text = summarize_text(clinical_text)
            log.info(f"  Summary: {summary_text[:100]}…")
        except ImportError:
            log.info("  clinical_summarizer not available, skipping summary")
    elif summary_text:
        log.info(f"  Step 3: Using provided summary ({len(summary_text)} chars)")

    results["summary_text"] = summary_text

    # ── Step 4: Triage fusion ─────────────────────────────────────────────────
    log.info("\n  Step 4: Computing triage score…")

    # Rule-based result (always computed for comparison)
    rule_result = estimate_urgency_rule_based(image_probs, ner_entities)
    results["rule_based"] = rule_result
    log.info(f"  Rule-based: score={rule_result['score']:.3f} | {rule_result['label']}")

    # Model-based result (if model provided)
    if model is not None:
        model_score = model.predict_single(image_probs, ner_entities, summary_text)
        results["model_score"] = model_score
        log.info(f"  Model-based: score={model_score:.3f}")

    # ── Feature vector details ────────────────────────────────────────────────
    fe = FeatureEngineer()
    vec = fe.build_feature_vector(image_probs, ner_entities, summary_text)
    results["feature_vector"] = vec
    results["feature_dim"] = len(vec)
    log.info(f"\n  Feature vector: {len(vec)} dimensions")

    log.info("\n  Pipeline complete ✓")
    return results


def _demo_ner_entities() -> Dict[str, List[Dict]]:
    """Return realistic demo NER entities for testing."""
    return {
        "symptoms": [
            {"text": "shortness of breath", "start": 0, "end": 19, "confidence": 0.92},
            {"text": "fever", "start": 25, "end": 30, "confidence": 0.88},
            {"text": "productive cough", "start": 35, "end": 51, "confidence": 0.85},
        ],
        "diseases": [
            {"text": "pneumonia", "start": 60, "end": 69, "confidence": 0.95},
            {"text": "hypertension", "start": 75, "end": 87, "confidence": 0.80},
        ],
        "medications": [
            {"text": "levofloxacin", "start": 100, "end": 112, "confidence": 0.90},
            {"text": "albuterol", "start": 115, "end": 124, "confidence": 0.85},
        ],
        "risk_factors": [
            {"text": "smoking", "start": 130, "end": 137, "confidence": 0.88},
            {"text": "elderly", "start": 140, "end": 147, "confidence": 0.70},
        ],
    }


# ==============================================================================
# SECTION 12 — SMOKE TEST
# ==============================================================================

def _run_smoke_test() -> None:
    """Validate the full pipeline without external dependencies."""
    log.info("=" * 60)
    log.info("  TRIAGE FUSION SMOKE TEST")
    log.info("=" * 60)

    # ── Test 1: Feature engineering ───────────────────────────────────────────
    log.info("  Test 1: Feature engineering…")
    fe = FeatureEngineer()

    image_probs = {"Pneumonia": 0.87, "Effusion": 0.45, "Cardiomegaly": 0.12}
    ner_entities = {
        "symptoms": [
            {"text": "fever", "start": 0, "end": 5, "confidence": 0.9},
            {"text": "cough", "start": 10, "end": 15, "confidence": 0.85},
        ],
        "diseases": [{"text": "pneumonia", "start": 20, "end": 29, "confidence": 0.95}],
        "medications": [{"text": "levofloxacin", "start": 30, "end": 42, "confidence": 0.9}],
        "risk_factors": [{"text": "smoking", "start": 50, "end": 57, "confidence": 0.8}],
    }

    img_feat = fe.extract_image_features(image_probs)
    assert img_feat.shape[0] == N_IMAGE_FEATURES + N_IMAGE_DERIVED
    log.info(f"  ✓ Image features: {img_feat.shape}")

    ner_feat = fe.extract_ner_features(ner_entities)
    assert ner_feat.shape[0] == N_NER_FEATURES + N_NER_DERIVED
    log.info(f"  ✓ NER features: {ner_feat.shape}")

    full_vec = fe.build_feature_vector(image_probs, ner_entities)
    assert full_vec.shape[0] == TOTAL_FEATURE_DIM
    log.info(f"  ✓ Full feature vector: {full_vec.shape} (dim={TOTAL_FEATURE_DIM})")

    names = fe.get_feature_names()
    assert len(names) == TOTAL_FEATURE_DIM
    log.info(f"  ✓ Feature names: {len(names)} names")

    # ── Test 2: Summary embedding ─────────────────────────────────────────────
    log.info("  Test 2: Summary embedding…")
    summary = (
        "65-year-old male with acute pneumonia and severe shortness of breath. "
        "Chest X-ray shows bilateral infiltrates. Started on IV levofloxacin. "
        "Patient is hemodynamically unstable with deteriorating respiratory status."
    )
    summary_feat = fe.extract_summary_embedding(summary)
    assert summary_feat.shape[0] == N_SUMMARY_FEATURES
    assert all(0 <= v <= 1 for v in summary_feat), "Summary features should be in [0, 1]"
    log.info(f"  ✓ Summary embedding: {summary_feat.shape}")
    log.info(f"    Values: {[f'{v:.3f}' for v in summary_feat]}")

    # With summary
    full_vec_with_summary = fe.build_feature_vector(image_probs, ner_entities, summary)
    assert full_vec_with_summary.shape[0] == TOTAL_FEATURE_DIM + N_SUMMARY_FEATURES
    log.info(f"  ✓ Full vector with summary: {full_vec_with_summary.shape}")

    # Empty summary should return zeros
    empty_feat = fe.extract_summary_embedding("")
    assert all(v == 0 for v in empty_feat)
    log.info("  ✓ Empty summary returns zero vector")

    names_with_summary = fe.get_feature_names(include_summary=True)
    assert len(names_with_summary) == TOTAL_FEATURE_DIM + N_SUMMARY_FEATURES
    log.info(f"  ✓ Feature names with summary: {len(names_with_summary)} names")

    # ── Test 3: Rule-based estimation ─────────────────────────────────────────
    log.info("  Test 3: Rule-based triage…")
    result = estimate_urgency_rule_based(image_probs, ner_entities)
    assert 0 <= result["score"] <= 1
    assert result["level"] in [1, 2, 3, 4, 5]
    assert "label" in result
    log.info(f"  ✓ Score={result['score']:.3f} | {result['label']}")

    # ── Test 4: predict_triage API ────────────────────────────────────────────
    log.info("  Test 4: predict_triage API…")
    result2 = predict_triage(image_probs, ner_entities)
    assert "score" in result2
    log.info(f"  ✓ API returned score={result2['score']:.3f}")

    # ── Test 5: Synthetic data generation ─────────────────────────────────────
    log.info("  Test 5: Synthetic data…")
    samples, labels = generate_synthetic_dataset(100, seed=42)
    assert len(samples) == 100
    assert len(labels) == 100
    assert set(labels).issubset({0, 1})
    log.info(f"  ✓ Generated {len(samples)} samples | "
             f"{(labels == 1).sum()} urgent, {(labels == 0).sum()} non-urgent")

    # ── Test 6: Logistic Regression training (with train/test split) ──────────
    log.info("  Test 6: Logistic Regression training…")
    lr_model = LogisticTriageModel()
    X = lr_model.feature_engineer.build_feature_matrix(samples)
    assert X.shape == (100, TOTAL_FEATURE_DIM)
    metrics = lr_model.train(X, labels)
    assert metrics["accuracy"] > 0.5
    score = lr_model.predict_single(image_probs, ner_entities)
    assert 0 <= score <= 1
    log.info(f"  ✓ LR trained | Acc={metrics['accuracy']:.3f} | "
             f"Single prediction={score:.3f}")

    # ── Test 7: MLP training ──────────────────────────────────────────────────
    log.info("  Test 7: MLP training…")
    mlp_model = MLPTriageModel(epochs=10)
    X2 = mlp_model.feature_engineer.build_feature_matrix(samples)
    metrics2 = mlp_model.train(X2, labels)
    score2 = mlp_model.predict_single(image_probs, ner_entities)
    assert 0 <= score2 <= 1
    log.info(f"  ✓ MLP trained | Acc={metrics2['accuracy']:.3f} | "
             f"Single prediction={score2:.3f}")

    # ── Test 8: Feature importance ────────────────────────────────────────────
    log.info("  Test 8: Feature importance…")
    importance = lr_model.get_feature_importance()
    assert len(importance) == TOTAL_FEATURE_DIM
    top3 = importance[:3]
    for name, coef in top3:
        log.info(f"    {name}: {coef:+.4f}")
    log.info("  ✓ Feature importance extracted")

    # ── Test 9: Model save/load round-trip ────────────────────────────────────
    log.info("  Test 9: Model save/load…")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_triage_model(lr_model, tmp_path, model_type="logistic")
        loaded_model = load_triage_model(tmp_path)
        score_original = lr_model.predict_single(image_probs, ner_entities)
        score_loaded = loaded_model.predict_single(image_probs, ner_entities)
        assert abs(score_original - score_loaded) < 1e-6, \
            f"Save/load mismatch: {score_original} vs {score_loaded}"
        log.info(f"  ✓ Save/load round-trip: {score_original:.4f} == {score_loaded:.4f}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # ── Test 10: Cross-validation ─────────────────────────────────────────────
    log.info("  Test 10: Cross-validation…")
    cv_results = cross_validate_triage(
        model_type="logistic", n_folds=3, n_synthetic=150, seed=42,
    )
    assert "aggregated" in cv_results
    assert "per_fold" in cv_results
    assert len(cv_results["per_fold"]) == 3
    mean_auc = cv_results["aggregated"]["auroc"]["mean"]
    log.info(f"  ✓ 3-fold CV complete | Mean AUC={mean_auc:.3f}")

    # ── Test 11: Evaluation report ────────────────────────────────────────────
    log.info("  Test 11: Evaluation report…")
    import tempfile as _tf
    with _tf.TemporaryDirectory() as eval_dir:
        evaluator = EvaluationReport(save_dir=eval_dir)
        y_prob_all = lr_model.predict(X)
        report = evaluator.generate_full_report(
            labels, y_prob_all, model=lr_model, model_name="LR Smoke Test",
        )
        assert "metrics" in report
        assert "files" in report
        assert report["metrics"]["accuracy"] > 0.5
        log.info(f"  ✓ Evaluation report generated | "
                 f"Acc={report['metrics']['accuracy']:.3f}")

    # ── Test 12: End-to-end pipeline ──────────────────────────────────────────
    log.info("  Test 12: End-to-end pipeline…")
    pipeline_result = run_full_pipeline(
        image_probs=image_probs,
        ner_entities=ner_entities,
        summary_text=summary,
        model=lr_model,
    )
    assert "rule_based" in pipeline_result
    assert "model_score" in pipeline_result
    assert "feature_vector" in pipeline_result
    assert pipeline_result["feature_dim"] == TOTAL_FEATURE_DIM + N_SUMMARY_FEATURES
    log.info(f"  ✓ Pipeline: rule={pipeline_result['rule_based']['score']:.3f} | "
             f"model={pipeline_result['model_score']:.3f}")

    log.info("\n  ALL TRIAGE FUSION SMOKE TESTS PASSED ✓ (12/12)")


# ==============================================================================
# SECTION 13 — CLI ENTRY POINT
# ==============================================================================

def main():
    """
    CLI for triage fusion.

    Usage:
        python triage_fusion.py --smoke-test
        python triage_fusion.py --train logistic
        python triage_fusion.py --train mlp
        python triage_fusion.py --cross-validate logistic
        python triage_fusion.py --evaluate
        python triage_fusion.py --pipeline
        python triage_fusion.py --demo
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Modal Triage Fusion — Urgency Prediction",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke-test", action="store_true",
                       help="Run all smoke tests (12 tests)")
    group.add_argument("--train", choices=["logistic", "mlp"],
                       help="Train a triage model with train/test split")
    group.add_argument("--cross-validate", choices=["logistic", "mlp"],
                       dest="cv_model",
                       help="Run 5-fold cross-validation")
    group.add_argument("--evaluate", action="store_true",
                       help="Train + generate full evaluation report with plots")
    group.add_argument("--pipeline", action="store_true",
                       help="Run end-to-end pipeline integration demo")
    group.add_argument("--demo", action="store_true",
                       help="Quick demo with rule-based + trained models")

    args = parser.parse_args()

    if args.smoke_test:
        _run_smoke_test()

    elif args.train:
        train_triage_pipeline(model_type=args.train)

    elif args.cv_model:
        cross_validate_triage(model_type=args.cv_model, n_folds=5)

    elif args.evaluate:
        # Train and generate comprehensive evaluation
        model, metrics = train_triage_pipeline("logistic", n_synthetic=500)
        samples, labels = generate_synthetic_dataset(500, seed=42)
        X = model.feature_engineer.build_feature_matrix(samples)
        y_prob = model.predict(X)
        evaluator = EvaluationReport(save_dir="./triage_evaluation")
        evaluator.generate_full_report(
            labels, y_prob, model=model, model_name="Logistic Regression",
        )
        log.info("Evaluation report saved to ./triage_evaluation/")

    elif args.pipeline:
        run_full_pipeline()

    elif args.demo:
        # Quick demo with rule-based estimation
        image_probs = {
            "Pneumonia": 0.87, "Effusion": 0.45, "Edema": 0.62,
            "Cardiomegaly": 0.33, "Atelectasis": 0.21,
            "Pneumothorax": 0.05, "Mass": 0.08, "Nodule": 0.12,
            "Infiltration": 0.55, "Consolidation": 0.41,
            "Emphysema": 0.15, "Fibrosis": 0.09,
            "Pleural_Thickening": 0.07, "Hernia": 0.03,
        }
        ner_entities = _demo_ner_entities()

        print("\n" + "=" * 60)
        print("  TRIAGE FUSION DEMO")
        print("=" * 60)

        result = predict_triage(image_probs, ner_entities)
        print(f"\n  Triage Score: {result['score']:.3f}")
        print(f"  Triage Level: {result['label']}")
        print(f"\n  Components:")
        for k, v in result["components"].items():
            print(f"    {k}: {v}")
        print(f"\n  Details:")
        for k, v in result["details"].items():
            print(f"    {k}: {v}")

        # Also train both models
        print("\n" + "─" * 60)
        train_triage_pipeline("logistic", n_synthetic=300)
        print("\n" + "─" * 60)
        train_triage_pipeline("mlp", n_synthetic=300)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
