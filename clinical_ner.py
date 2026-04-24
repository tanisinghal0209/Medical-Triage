"""
================================================================================
  STEP 4 — CLINICAL NER: MEDICAL ENTITY EXTRACTION
  Chest X-Ray Multi-Modal Pipeline
  Module: Extract structured medical entities from clinical notes

  Uses: HuggingFace Transformers (Bio-ClinicalBERT / BioBERT backbone)
================================================================================

ENTITIES EXTRACTED:
  - Symptoms       (e.g., "shortness of breath", "fever", "cough")
  - Diseases       (e.g., "pneumonia", "tuberculosis", "COPD")
  - Medications    (e.g., "amoxicillin", "albuterol", "prednisone")
  - Risk Factors   (e.g., "smoking history", "diabetes", "immunocompromised")

OUTPUT FORMAT:
  {
      "symptoms":     [{"text": "...", "start": int, "end": int, "confidence": float}],
      "diseases":     [{"text": "...", "start": int, "end": int, "confidence": float}],
      "medications":  [{"text": "...", "start": int, "end": int, "confidence": float}],
      "risk_factors": [{"text": "...", "start": int, "end": int, "confidence": float}]
  }

QUICK START:
  from clinical_ner import extract_entities
  result = extract_entities("Patient presents with fever and productive cough.")
  print(result)

DEPENDS ON:
  pip install transformers torch sentencepiece
================================================================================
"""

# ==============================================================================
# SECTION 0 — IMPORTS
# ==============================================================================

import re
import json
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
    Pipeline,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1 — ENTITY CATEGORY DEFINITIONS
# ==============================================================================

# Comprehensive medical vocabularies for entity classification.
# These gazetteers are used as a secondary signal alongside the NER model
# to categorize and validate detected entities into the four target groups.

SYMPTOM_TERMS: Set[str] = {
    # ── Respiratory ───────────────────────────────────────────────────────────
    "cough", "productive cough", "dry cough", "chronic cough",
    "shortness of breath", "dyspnea", "breathlessness", "wheezing",
    "hemoptysis", "stridor", "tachypnea", "orthopnea",
    "pleuritic chest pain", "chest tightness",

    # ── General / Systemic ────────────────────────────────────────────────────
    "fever", "chills", "night sweats", "fatigue", "malaise",
    "weight loss", "anorexia", "weakness", "lethargy",
    "diaphoresis", "rigors",

    # ── Pain ──────────────────────────────────────────────────────────────────
    "chest pain", "back pain", "headache", "myalgia", "arthralgia",
    "abdominal pain", "pleurisy", "pain",

    # ── GI ────────────────────────────────────────────────────────────────────
    "nausea", "vomiting", "diarrhea", "constipation",

    # ── Cardiovascular ────────────────────────────────────────────────────────
    "palpitations", "edema", "swelling", "syncope", "dizziness",
    "peripheral edema", "jugular venous distension",

    # ── Neurological ──────────────────────────────────────────────────────────
    "confusion", "altered mental status", "seizure", "tremor",

    # ── Other ─────────────────────────────────────────────────────────────────
    "sore throat", "nasal congestion", "rhinorrhea", "sputum",
    "purulent sputum", "blood-tinged sputum", "hoarseness",
    "loss of appetite", "inability to eat", "difficulty breathing",
    "rapid breathing", "shallow breathing",
}

DISEASE_TERMS: Set[str] = {
    # ── Pulmonary / Thoracic ──────────────────────────────────────────────────
    "pneumonia", "tuberculosis", "tb", "copd",
    "chronic obstructive pulmonary disease",
    "asthma", "bronchitis", "emphysema", "lung cancer",
    "pulmonary embolism", "pulmonary edema", "pulmonary fibrosis",
    "pneumothorax", "pleural effusion", "atelectasis",
    "acute respiratory distress syndrome", "ards",
    "interstitial lung disease", "sarcoidosis",
    "lung abscess", "bronchiectasis", "mesothelioma",
    "covid-19", "covid", "influenza", "rsv",

    # ── Cardiac ───────────────────────────────────────────────────────────────
    "heart failure", "congestive heart failure", "chf",
    "myocardial infarction", "mi", "coronary artery disease", "cad",
    "cardiomegaly", "pericarditis", "endocarditis",
    "aortic aneurysm", "hypertension", "htn",
    "atrial fibrillation", "afib",

    # ── Infectious ────────────────────────────────────────────────────────────
    "sepsis", "septicemia", "meningitis", "cellulitis",
    "urinary tract infection", "uti",

    # ── Oncology ──────────────────────────────────────────────────────────────
    "carcinoma", "adenocarcinoma", "lymphoma", "metastasis",
    "malignancy", "neoplasm", "tumor", "mass",

    # ── Other ─────────────────────────────────────────────────────────────────
    "diabetes", "diabetes mellitus", "type 2 diabetes",
    "chronic kidney disease", "ckd", "renal failure",
    "anemia", "hepatitis", "cirrhosis",
    "hernia", "fibrosis", "consolidation", "infiltration",
    "edema", "effusion", "nodule", "pleural thickening",
}

MEDICATION_TERMS: Set[str] = {
    # ── Antibiotics ───────────────────────────────────────────────────────────
    "amoxicillin", "azithromycin", "zithromax", "levofloxacin",
    "levaquin", "ciprofloxacin", "cipro", "doxycycline",
    "metronidazole", "flagyl", "vancomycin", "meropenem",
    "ceftriaxone", "rocephin", "piperacillin", "tazobactam",
    "zosyn", "ampicillin", "sulbactam", "unasyn",
    "trimethoprim", "sulfamethoxazole", "bactrim",
    "clindamycin", "erythromycin", "penicillin",
    "cephalosporin", "fluoroquinolone", "macrolide",

    # ── Respiratory ───────────────────────────────────────────────────────────
    "albuterol", "salbutamol", "ipratropium", "tiotropium",
    "budesonide", "fluticasone", "montelukast", "singulair",
    "theophylline", "roflumilast", "bronchodilator",
    "inhaler", "nebulizer",

    # ── Steroids / Anti-inflammatory ──────────────────────────────────────────
    "prednisone", "prednisolone", "methylprednisolone",
    "dexamethasone", "hydrocortisone", "solumedrol",
    "ibuprofen", "naproxen", "aspirin", "acetaminophen",
    "tylenol", "nsaid",

    # ── Cardiac ───────────────────────────────────────────────────────────────
    "lisinopril", "enalapril", "metoprolol", "atenolol",
    "amlodipine", "losartan", "valsartan", "furosemide",
    "lasix", "spironolactone", "hydrochlorothiazide", "hctz",
    "warfarin", "coumadin", "heparin", "enoxaparin", "lovenox",
    "apixaban", "eliquis", "rivaroxaban", "xarelto",
    "clopidogrel", "plavix", "digoxin", "amiodarone",
    "nitroglycerin", "statin", "atorvastatin", "lipitor",

    # ── Analgesics / Sedatives ────────────────────────────────────────────────
    "morphine", "fentanyl", "hydrocodone", "oxycodone",
    "tramadol", "codeine", "gabapentin", "pregabalin",
    "lorazepam", "ativan", "midazolam", "propofol",

    # ── Diabetes ──────────────────────────────────────────────────────────────
    "metformin", "insulin", "glipizide", "glyburide",
    "sitagliptin", "januvia", "empagliflozin", "jardiance",

    # ── Antivirals ────────────────────────────────────────────────────────────
    "oseltamivir", "tamiflu", "remdesivir", "acyclovir",

    # ── Other ─────────────────────────────────────────────────────────────────
    "oxygen", "supplemental oxygen", "o2", "epinephrine",
    "norepinephrine", "dopamine", "vasopressor",
    "pantoprazole", "omeprazole", "famotidine",
}

RISK_FACTOR_TERMS: Set[str] = {
    # ── Behavioral ────────────────────────────────────────────────────────────
    "smoking", "smoker", "tobacco use", "tobacco abuse",
    "smoking history", "pack-year", "former smoker", "current smoker",
    "alcohol use", "alcohol abuse", "substance abuse",
    "drug use", "iv drug use", "illicit drug use",
    "sedentary lifestyle", "obesity", "obese",

    # ── Comorbidities as risk ─────────────────────────────────────────────────
    "diabetes", "hypertension", "immunocompromised",
    "immunosuppressed", "hiv", "aids",
    "chronic kidney disease", "liver disease",
    "organ transplant", "transplant recipient",
    "cancer history", "malignancy history",

    # ── Demographics ──────────────────────────────────────────────────────────
    "elderly", "advanced age", "geriatric",
    "male", "female", "pregnancy", "pregnant",
    "pediatric", "neonatal",

    # ── Occupational / Environmental ──────────────────────────────────────────
    "asbestos exposure", "occupational exposure",
    "chemical exposure", "radiation exposure",
    "travel history", "recent travel",
    "animal contact", "bird exposure",
    "crowded living", "homeless", "incarcerated",
    "healthcare worker", "close contact",

    # ── Medical history ───────────────────────────────────────────────────────
    "family history", "genetic predisposition",
    "previous hospitalization", "recent surgery",
    "mechanical ventilation", "intubation",
    "central line", "indwelling catheter",
    "bed rest", "immobility", "prolonged immobilization",
    "dvt history", "blood clot history",
    "steroid use", "chronic steroid use", "long-term steroid use",
    "chemotherapy", "radiation therapy",
}


# ==============================================================================
# SECTION 2 — TEXT PREPROCESSING FOR NOISY CLINICAL NOTES
# ==============================================================================

class ClinicalTextCleaner:
    """
    Handles the messiness of real-world clinical text:
      - Inconsistent abbreviations
      - Missing punctuation
      - Mixed case
      - Medical shorthand
      - Templated note fragments

    Cleans text while preserving character offsets for entity mapping.
    """

    # Common clinical abbreviations → full forms
    ABBREVIATION_MAP: Dict[str, str] = {
        r"\bSOB\b":    "shortness of breath",
        r"\bDOE\b":    "dyspnea on exertion",
        r"\bCP\b":     "chest pain",
        r"\bHTN\b":    "hypertension",
        r"\bDM\b":     "diabetes mellitus",
        r"\bCHF\b":    "congestive heart failure",
        r"\bCAD\b":    "coronary artery disease",
        r"\bCOPD\b":   "chronic obstructive pulmonary disease",
        r"\bMI\b":     "myocardial infarction",
        r"\bCKD\b":    "chronic kidney disease",
        r"\bUTI\b":    "urinary tract infection",
        r"\bAFib\b":   "atrial fibrillation",
        r"\bPE\b":     "pulmonary embolism",
        r"\bDVT\b":    "deep vein thrombosis",
        r"\bARDS\b":   "acute respiratory distress syndrome",
        r"\bBID\b":    "twice daily",
        r"\bTID\b":    "three times daily",
        r"\bQID\b":    "four times daily",
        r"\bPRN\b":    "as needed",
        r"\bPO\b":     "by mouth",
        r"\bIV\b":     "intravenous",
        r"\bIM\b":     "intramuscular",
        r"\bSQ\b":     "subcutaneous",
        r"\bBP\b":     "blood pressure",
        r"\bHR\b":     "heart rate",
        r"\bRR\b":     "respiratory rate",
        r"\bO2\s*sat\b": "oxygen saturation",
        r"\bWBC\b":    "white blood cell count",
        r"\bHgb\b":    "hemoglobin",
        r"\bCBC\b":    "complete blood count",
        r"\bBMP\b":    "basic metabolic panel",
        r"\bCXR\b":    "chest x-ray",
        r"\bCT\b":     "computed tomography",
        r"\bABG\b":    "arterial blood gas",
    }

    @staticmethod
    def clean(text: str) -> str:
        """
        Clean clinical text while making it more parseable for NER.

        Steps:
          1. Normalize whitespace (tabs, multiple spaces → single space)
          2. Fix common punctuation issues
          3. Normalize section headers
          4. Clean up list markers

        Note: We do NOT expand abbreviations by default because the NER
        model is trained on text containing abbreviations. Expansion is
        available via expand_abbreviations() for dictionary-based matching.
        """
        if not text or not text.strip():
            return ""

        # ── Normalize whitespace ──────────────────────────────────────────────
        text = re.sub(r"\t", " ", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # ── Fix missing spaces after periods (common in clinical notes) ───────
        text = re.sub(r"\.([A-Z])", r". \1", text)

        # ── Normalize dashes in lists ─────────────────────────────────────────
        text = re.sub(r"^[-•●]\s*", "- ", text, flags=re.MULTILINE)

        # ── Strip leading/trailing whitespace per line ────────────────────────
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        return text.strip()

    @classmethod
    def expand_abbreviations(cls, text: str) -> str:
        """
        Expand common clinical abbreviations.
        Use cautiously — may introduce noise if abbreviation is ambiguous.
        """
        for pattern, expansion in cls.ABBREVIATION_MAP.items():
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text


# ==============================================================================
# SECTION 3 — NER MODEL MANAGER
# ==============================================================================

# Default model: a biomedical NER model fine-tuned on clinical text
# This model recognizes standard biomedical entity types which we then
# map to our four target categories.
DEFAULT_MODEL = "d4data/biomedical-ner-all"

# Fallback model if the primary isn't available
FALLBACK_MODEL = "samrawal/bert-base-uncased_clinical-ner"

# Entity label → our category mapping
# Different NER models use different label schemes; this mapping handles
# the most common biomedical NER label sets.
LABEL_TO_CATEGORY: Dict[str, str] = {
    # ── d4data/biomedical-ner-all labels ──────────────────────────────────────
    "Sign_symptom":       "symptoms",
    "Disease_disorder":   "diseases",
    "Medication":         "medications",
    "Drug":               "medications",
    "Clinical_event":     "risk_factors",
    "Activity":           "risk_factors",
    "Diagnostic_procedure": "risk_factors",    # Procedures can be risk-relevant
    "Lab_value":          "risk_factors",
    "Biological_structure": None,              # Skip anatomy terms
    "Therapeutic_procedure": "medications",    # e.g., "oxygen therapy"
    "Nonbiological_location": None,
    "Detailed_description": None,
    "Shape": None,
    "Subject": None,
    "Qualitative_concept": None,
    "Quantitative_concept": None,
    "Severity": None,
    "Date": None,
    "Time": None,
    "Duration": None,
    "Distance": None,
    "Area": None,
    "Volume": None,
    "Mass": None,
    "Dosage": None,
    "Frequency": None,
    "Age": None,
    "Biological_attribute": None,
    "Color": None,
    "Texture": None,
    "Coreference": None,
    "Other_entity": None,
    "Other_event": None,

    # ── Generic BIO labels (some models) ──────────────────────────────────────
    "PROBLEM":    "diseases",
    "TREATMENT":  "medications",
    "TEST":       "risk_factors",

    # ── Misc labels ───────────────────────────────────────────────────────────
    "DISEASE":    "diseases",
    "SYMPTOM":    "symptoms",
    "CHEMICAL":   "medications",
    "GENE":       None,
    "SPECIES":    None,
}


class NERModelManager:
    """
    Manages the HuggingFace NER pipeline.

    Handles:
      - Model loading with automatic fallback
      - GPU/CPU device selection
      - Tokenizer configuration for clinical text
      - Aggregation strategy for multi-token entities

    Attributes:
        pipe: HuggingFace NER pipeline ready for inference.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[int] = None,
        aggregation_strategy: str = "simple",
    ):
        """
        Args:
            model_name           : HuggingFace model ID or local path.
            device               : GPU index (0, 1, ...) or -1/None for CPU.
            aggregation_strategy : How to merge sub-word tokens.
                                   "simple" → merge adjacent same-label tokens.
                                   "first"  → use first sub-word's label.
                                   "average" → average scores across sub-words.
        """
        self.model_name = model_name
        self.aggregation = aggregation_strategy

        # ── Auto-detect device ────────────────────────────────────────────────
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        device_name = "GPU" if self.device >= 0 else "CPU"
        log.info(f"NER device: {device_name}")

        # ── Load pipeline ─────────────────────────────────────────────────────
        self.pipe = self._load_pipeline()

    def _load_pipeline(self) -> Pipeline:
        """Load the NER pipeline with automatic fallback on failure."""
        try:
            log.info(f"Loading NER model: {self.model_name}")
            pipe = pipeline(
                "ner",
                model=self.model_name,
                tokenizer=self.model_name,
                aggregation_strategy=self.aggregation,
                device=self.device,
            )
            log.info(f"NER model loaded successfully: {self.model_name}")
            return pipe

        except Exception as e:
            log.warning(f"Failed to load {self.model_name}: {e}")
            log.info(f"Attempting fallback model: {FALLBACK_MODEL}")

            try:
                pipe = pipeline(
                    "ner",
                    model=FALLBACK_MODEL,
                    tokenizer=FALLBACK_MODEL,
                    aggregation_strategy=self.aggregation,
                    device=self.device,
                )
                self.model_name = FALLBACK_MODEL
                log.info(f"Fallback model loaded: {FALLBACK_MODEL}")
                return pipe

            except Exception as e2:
                log.error(f"Fallback also failed: {e2}")
                log.info("Falling back to dictionary-based extraction only.")
                return None

    def predict(self, text: str) -> List[Dict]:
        """
        Run NER inference on text.

        Returns list of dicts:
          [{"entity_group": "Disease_disorder", "word": "pneumonia",
            "start": 42, "end": 51, "score": 0.97}, ...]
        """
        if self.pipe is None:
            return []

        try:
            # HuggingFace pipeline handles tokenization internally
            results = self.pipe(text)
            return results
        except Exception as e:
            log.error(f"NER inference error: {e}")
            return []


# ==============================================================================
# SECTION 4 — DICTIONARY-BASED ENTITY MATCHER (GAZETTEER)
# ==============================================================================

class GazetteerMatcher:
    """
    Dictionary-based entity matching using medical term gazetteers.

    Serves two purposes:
      1. Catches entities the NER model misses (high recall)
      2. Helps categorize model-detected entities into our 4 target classes

    Uses efficient regex-based matching with word boundary detection.
    """

    def __init__(self):
        # ── Build compiled regex patterns per category ────────────────────────
        self._patterns: Dict[str, List[re.Pattern]] = {}

        for category, terms in [
            ("symptoms",     SYMPTOM_TERMS),
            ("diseases",     DISEASE_TERMS),
            ("medications",  MEDICATION_TERMS),
            ("risk_factors", RISK_FACTOR_TERMS),
        ]:
            # Sort by length (longest first) for greedy matching
            sorted_terms = sorted(terms, key=len, reverse=True)
            patterns = []
            for term in sorted_terms:
                # Escape special regex chars, then build word-boundary pattern
                escaped = re.escape(term)
                # Allow flexible whitespace/hyphens between words
                escaped = re.sub(r"\\ ", r"[\\s\\-]+", escaped)
                pattern = re.compile(
                    rf"\b{escaped}\b",
                    re.IGNORECASE,
                )
                patterns.append(pattern)
            self._patterns[category] = patterns

    def match(self, text: str) -> Dict[str, List[Dict]]:
        """
        Find all gazetteer matches in the text.

        Returns:
            {
                "symptoms": [{"text": "...", "start": int, "end": int, "confidence": 1.0}],
                ...
            }
        """
        results: Dict[str, List[Dict]] = {
            "symptoms": [], "diseases": [], "medications": [], "risk_factors": []
        }
        # Track matched spans to avoid duplicates
        matched_spans: Set[Tuple[int, int]] = set()

        for category, patterns in self._patterns.items():
            for pattern in patterns:
                for m in pattern.finditer(text):
                    span = (m.start(), m.end())
                    # Skip if this span overlaps with an already-matched span
                    if any(
                        s <= span[0] < e or s < span[1] <= e
                        for s, e in matched_spans
                    ):
                        continue

                    matched_spans.add(span)
                    results[category].append({
                        "text": m.group(),
                        "start": m.start(),
                        "end": m.end(),
                        "confidence": 0.85,  # Gazetteer matches get fixed confidence
                        "source": "gazetteer",
                    })

        return results

    def categorize_entity(self, entity_text: str) -> Optional[str]:
        """
        Determine which category a given entity string belongs to.
        Used to re-categorize NER-detected entities.
        """
        text_lower = entity_text.lower().strip()

        # Check each category's terms
        for category, terms in [
            ("symptoms",     SYMPTOM_TERMS),
            ("diseases",     DISEASE_TERMS),
            ("medications",  MEDICATION_TERMS),
            ("risk_factors", RISK_FACTOR_TERMS),
        ]:
            if text_lower in {t.lower() for t in terms}:
                return category

        return None


# ==============================================================================
# SECTION 5 — ENTITY MERGER & DEDUPLICATION
# ==============================================================================

def merge_entities(
    ner_entities: Dict[str, List[Dict]],
    gaz_entities: Dict[str, List[Dict]],
    overlap_threshold: float = 0.5,
) -> Dict[str, List[Dict]]:
    """
    Merge NER model predictions with gazetteer matches.

    Strategy:
      - NER entities take priority (higher confidence, contextual)
      - Gazetteer entities fill gaps the model missed
      - Overlapping spans are deduplicated (keep higher confidence)
      - Final sort by position in text

    Args:
        ner_entities      : Entities from the NER model.
        gaz_entities      : Entities from the gazetteer.
        overlap_threshold : IoU threshold for considering spans as duplicates.

    Returns:
        Merged and deduplicated entity dict.
    """
    merged: Dict[str, List[Dict]] = {
        "symptoms": [], "diseases": [], "medications": [], "risk_factors": []
    }

    for category in merged:
        all_entities = []

        # Add NER entities first (higher priority)
        for ent in ner_entities.get(category, []):
            ent["source"] = ent.get("source", "model")
            all_entities.append(ent)

        # Add gazetteer entities
        for ent in gaz_entities.get(category, []):
            all_entities.append(ent)

        # ── Deduplicate overlapping spans ─────────────────────────────────────
        # Sort by confidence descending — keep higher confidence spans
        all_entities.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        kept: List[Dict] = []
        used_spans: List[Tuple[int, int]] = []

        for ent in all_entities:
            s, e = ent["start"], ent["end"]
            span_len = e - s

            is_duplicate = False
            for us, ue in used_spans:
                # Calculate overlap
                overlap_start = max(s, us)
                overlap_end = min(e, ue)
                overlap = max(0, overlap_end - overlap_start)

                if span_len > 0 and overlap / span_len >= overlap_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(ent)
                used_spans.append((s, e))

        # Sort by position in text
        kept.sort(key=lambda x: x["start"])
        merged[category] = kept

    return merged


# ==============================================================================
# SECTION 6 — MAIN API: extract_entities()
# ==============================================================================

# Module-level singleton for the NER model (lazy-loaded)
_ner_manager: Optional[NERModelManager] = None
_gazetteer: Optional[GazetteerMatcher] = None


def _get_ner_manager(model_name: str = DEFAULT_MODEL) -> NERModelManager:
    """Lazy-load the NER model (singleton pattern)."""
    global _ner_manager
    if _ner_manager is None or _ner_manager.model_name != model_name:
        _ner_manager = NERModelManager(model_name=model_name)
    return _ner_manager


def _get_gazetteer() -> GazetteerMatcher:
    """Lazy-load the gazetteer matcher (singleton pattern)."""
    global _gazetteer
    if _gazetteer is None:
        _gazetteer = GazetteerMatcher()
    return _gazetteer


def extract_entities(
    text: str,
    model: Optional[NERModelManager] = None,
    model_name: str = DEFAULT_MODEL,
    clean_text: bool = True,
    expand_abbrevs: bool = True,
    use_gazetteer: bool = True,
    min_confidence: float = 0.3,
) -> Dict[str, List[Dict]]:
    """
    Extract structured medical entities from clinical text.

    This is the main entry point. It combines:
      1. Text cleaning (handles noisy clinical text)
      2. Abbreviation expansion
      3. NER model inference (HuggingFace transformer)
      4. Dictionary-based matching (gazetteers)
      5. Entity merging & deduplication

    Args:
        text           : Raw clinical note text.
        model          : Pre-loaded NERModelManager (optional, for reuse).
        model_name     : HuggingFace model ID (used if model is None).
        clean_text     : Whether to clean the input text.
        expand_abbrevs : Whether to expand clinical abbreviations.
        use_gazetteer  : Whether to use dictionary-based matching.
        min_confidence : Minimum confidence threshold for entities.

    Returns:
        {
            "symptoms":     [{"text": str, "start": int, "end": int, "confidence": float}],
            "diseases":     [{"text": str, "start": int, "end": int, "confidence": float}],
            "medications":  [{"text": str, "start": int, "end": int, "confidence": float}],
            "risk_factors": [{"text": str, "start": int, "end": int, "confidence": float}]
        }

    Example:
        >>> result = extract_entities(
        ...     "65 y/o male with HTN, DM presents with SOB and productive cough. "
        ...     "Started on amoxicillin 500mg PO TID. CXR shows bilateral infiltrates "
        ...     "consistent with pneumonia. History of smoking 30 pack-years."
        ... )
        >>> print(json.dumps(result, indent=2))
    """
    if not text or not text.strip():
        return {
            "symptoms": [], "diseases": [],
            "medications": [], "risk_factors": []
        }

    # ── Step 1: Clean text ────────────────────────────────────────────────────
    processed_text = text
    if clean_text:
        processed_text = ClinicalTextCleaner.clean(processed_text)
    if expand_abbrevs:
        processed_text = ClinicalTextCleaner.expand_abbreviations(processed_text)

    log.debug(f"Processed text ({len(processed_text)} chars): {processed_text[:100]}…")

    # ── Step 2: NER model inference ───────────────────────────────────────────
    ner_entities: Dict[str, List[Dict]] = {
        "symptoms": [], "diseases": [], "medications": [], "risk_factors": []
    }

    if model is None:
        model = _get_ner_manager(model_name)

    raw_predictions = model.predict(processed_text)
    gazetteer = _get_gazetteer()

    for pred in raw_predictions:
        entity_text = pred.get("word", "").strip()
        if not entity_text or len(entity_text) < 2:
            continue

        score = float(pred.get("score", 0))
        if score < min_confidence:
            continue

        # ── Map NER label → our category ──────────────────────────────────────
        label = pred.get("entity_group", pred.get("entity", ""))
        # Strip B-, I- prefixes from BIO tags
        label = re.sub(r"^[BI]-", "", label)

        category = LABEL_TO_CATEGORY.get(label, None)

        # If model label doesn't map, try gazetteer lookup
        if category is None:
            category = gazetteer.categorize_entity(entity_text)

        if category is None:
            continue  # Skip entities that don't fit our categories

        ner_entities[category].append({
            "text": entity_text,
            "start": pred.get("start", 0),
            "end": pred.get("end", 0),
            "confidence": round(score, 4),
            "source": "model",
        })

    # ── Step 3: Gazetteer matching ────────────────────────────────────────────
    gaz_entities: Dict[str, List[Dict]] = {
        "symptoms": [], "diseases": [], "medications": [], "risk_factors": []
    }
    if use_gazetteer:
        gaz_entities = gazetteer.match(processed_text)

    # ── Step 4: Merge & deduplicate ───────────────────────────────────────────
    final = merge_entities(ner_entities, gaz_entities)

    # ── Step 5: Apply confidence threshold ────────────────────────────────────
    for category in final:
        final[category] = [
            {k: v for k, v in ent.items() if k != "source"}  # Remove internal field
            for ent in final[category]
            if ent.get("confidence", 0) >= min_confidence
        ]

    return final


# ==============================================================================
# SECTION 7 — BATCH EXTRACTION
# ==============================================================================

def extract_entities_batch(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    **kwargs,
) -> List[Dict[str, List[Dict]]]:
    """
    Extract entities from multiple clinical notes.

    Args:
        texts      : List of clinical note strings.
        model_name : HuggingFace model ID.
        **kwargs   : Additional arguments passed to extract_entities().

    Returns:
        List of entity dicts, one per input text.

    Example:
        >>> notes = [
        ...     "Patient with pneumonia started on azithromycin.",
        ...     "History of COPD, current smoker, presenting with cough."
        ... ]
        >>> results = extract_entities_batch(notes)
    """
    # Pre-load model once for all texts
    ner_model = _get_ner_manager(model_name)

    results = []
    for idx, text in enumerate(texts):
        log.info(f"Processing note [{idx + 1}/{len(texts)}]…")
        result = extract_entities(text, model=ner_model, **kwargs)
        results.append(result)

    return results


# ==============================================================================
# SECTION 8 — PRETTY PRINTING & FORMATTING
# ==============================================================================

def format_entities(
    entities: Dict[str, List[Dict]],
    include_metadata: bool = False,
) -> str:
    """
    Format extracted entities as a readable string.

    Args:
        entities         : Output from extract_entities().
        include_metadata : Include start/end/confidence details.

    Returns:
        Formatted multi-line string.
    """
    lines = []
    category_icons = {
        "symptoms":     "🩺",
        "diseases":     "🦠",
        "medications":  "💊",
        "risk_factors": "⚠️",
    }

    for category, ents in entities.items():
        icon = category_icons.get(category, "•")
        header = category.replace("_", " ").title()
        lines.append(f"\n{icon}  {header} ({len(ents)} found)")
        lines.append("─" * 40)

        if not ents:
            lines.append("   (none detected)")
            continue

        for ent in ents:
            text = ent["text"]
            conf = ent.get("confidence", 0)
            conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))

            if include_metadata:
                lines.append(
                    f"   • {text:<30s}  [{conf_bar}] {conf:.1%}  "
                    f"(pos: {ent.get('start', '?')}–{ent.get('end', '?')})"
                )
            else:
                lines.append(f"   • {text:<30s}  [{conf_bar}] {conf:.1%}")

    return "\n".join(lines)


def entities_to_json(
    entities: Dict[str, List[Dict]],
    simple: bool = False,
) -> str:
    """
    Convert entities to JSON string.

    Args:
        entities : Output from extract_entities().
        simple   : If True, return only entity text lists (no metadata).

    Returns:
        JSON string.
    """
    if simple:
        simple_dict = {
            cat: [ent["text"] for ent in ents]
            for cat, ents in entities.items()
        }
        return json.dumps(simple_dict, indent=2)

    return json.dumps(entities, indent=2)


# ==============================================================================
# SECTION 9 — EVALUATION METRICS
# ==============================================================================

@dataclass
class EvalMetrics:
    """Evaluation metrics for entity extraction."""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def __str__(self) -> str:
        return (
            f"Precision: {self.precision:.3f} | "
            f"Recall: {self.recall:.3f} | "
            f"F1: {self.f1:.3f} | "
            f"TP={self.true_positives} FP={self.false_positives} FN={self.false_negatives}"
        )


def evaluate_extraction(
    predicted: Dict[str, List[Dict]],
    gold: Dict[str, List[str]],
    match_mode: str = "partial",
) -> Dict[str, EvalMetrics]:
    """
    Evaluate entity extraction against gold-standard annotations.

    Args:
        predicted  : Output from extract_entities().
        gold       : Ground truth: {category: [entity_text, ...]}.
        match_mode : "exact" → full string match.
                     "partial" → substring match (more lenient).

    Returns:
        Dict mapping category → EvalMetrics.

    Example:
        >>> gold = {
        ...     "symptoms": ["fever", "cough"],
        ...     "diseases": ["pneumonia"],
        ...     "medications": ["amoxicillin"],
        ...     "risk_factors": ["smoking"]
        ... }
        >>> metrics = evaluate_extraction(predicted, gold)
        >>> for cat, m in metrics.items():
        ...     print(f"{cat}: {m}")
    """
    results: Dict[str, EvalMetrics] = {}

    for category in ["symptoms", "diseases", "medications", "risk_factors"]:
        pred_texts = {ent["text"].lower().strip() for ent in predicted.get(category, [])}
        gold_texts = {t.lower().strip() for t in gold.get(category, [])}

        if match_mode == "partial":
            # Partial matching: predicted entity matches if it contains or
            # is contained by a gold entity
            tp = 0
            matched_gold = set()
            matched_pred = set()

            for pred_t in pred_texts:
                for gold_t in gold_texts:
                    if pred_t in gold_t or gold_t in pred_t:
                        if gold_t not in matched_gold:
                            tp += 1
                            matched_gold.add(gold_t)
                            matched_pred.add(pred_t)
                            break

            fp = len(pred_texts) - len(matched_pred)
            fn = len(gold_texts) - len(matched_gold)

        else:  # exact match
            tp = len(pred_texts & gold_texts)
            fp = len(pred_texts - gold_texts)
            fn = len(gold_texts - pred_texts)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        results[category] = EvalMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )

    return results


# ==============================================================================
# SECTION 10 — EXAMPLE INPUTS / OUTPUTS & DEMO
# ==============================================================================

EXAMPLE_NOTES = [
    # ── Example 1: Standard clinical note ─────────────────────────────────────
    {
        "text": (
            "65 y/o male presents to ED with chief complaint of SOB and "
            "productive cough for 3 days. Associated fever (101.2F) and chills. "
            "PMH significant for HTN, DM type 2, and COPD. "
            "Current smoker with 30 pack-year history. "
            "CXR shows bilateral infiltrates consistent with pneumonia. "
            "Started on levofloxacin 750mg IV daily and albuterol nebulizer PRN. "
            "O2 sat 89% on room air, placed on supplemental oxygen 2L NC."
        ),
        "gold": {
            "symptoms": ["shortness of breath", "productive cough", "fever", "chills"],
            "diseases": ["hypertension", "diabetes", "COPD", "pneumonia"],
            "medications": ["levofloxacin", "albuterol", "supplemental oxygen"],
            "risk_factors": ["smoker", "smoking"],
        },
    },

    # ── Example 2: Radiology report ───────────────────────────────────────────
    {
        "text": (
            "FINDINGS: There is a large right-sided pleural effusion with "
            "associated compressive atelectasis. Cardiomegaly is noted. "
            "No pneumothorax identified. Small bilateral nodules measuring "
            "3-5mm, recommend follow-up CT in 3 months. "
            "IMPRESSION: 1. Large right pleural effusion. "
            "2. Cardiomegaly suggesting possible congestive heart failure. "
            "3. Pulmonary nodules - recommend follow-up."
        ),
        "gold": {
            "symptoms": [],
            "diseases": [
                "pleural effusion", "atelectasis", "cardiomegaly",
                "congestive heart failure", "nodules"
            ],
            "medications": [],
            "risk_factors": [],
        },
    },

    # ── Example 3: Noisy / abbreviated note ───────────────────────────────────
    {
        "text": (
            "pt c/o CP x2d, worse w/ deep inspiration.hx of PE 2yrs ago. "
            "on warfarin.also reports DOE and LE edema. "
            "pmh: CHF, afib, CKD stage 3 "
            "meds: furosemide 40mg bid, metoprolol 25mg bid, warfarin 5mg daily "
            "social: former smoker quit 5y ago, no etoh"
        ),
        "gold": {
            "symptoms": ["chest pain", "dyspnea on exertion", "edema"],
            "diseases": ["pulmonary embolism", "CHF", "atrial fibrillation", "CKD"],
            "medications": ["warfarin", "furosemide", "metoprolol"],
            "risk_factors": ["former smoker"],
        },
    },
]


def run_demo(verbose: bool = True) -> None:
    """
    Run the entity extraction pipeline on example clinical notes.
    Demonstrates the full pipeline including evaluation.
    """
    log.info("=" * 70)
    log.info("  CLINICAL NER DEMO — Medical Entity Extraction")
    log.info("=" * 70)

    for idx, example in enumerate(EXAMPLE_NOTES, 1):
        text = example["text"]
        gold = example["gold"]

        log.info(f"\n{'─' * 70}")
        log.info(f"  EXAMPLE {idx}")
        log.info(f"{'─' * 70}")

        if verbose:
            # Show truncated input
            display_text = text[:150] + ("…" if len(text) > 150 else "")
            log.info(f"  Input: {display_text}")

        # ── Extract entities ──────────────────────────────────────────────────
        result = extract_entities(text)

        # ── Display results ───────────────────────────────────────────────────
        formatted = format_entities(result, include_metadata=True)
        print(formatted)

        # ── Evaluate against gold standard ────────────────────────────────────
        metrics = evaluate_extraction(result, gold, match_mode="partial")
        print(f"\n📊  Evaluation (partial match):")
        for category, m in metrics.items():
            icon = {"symptoms": "🩺", "diseases": "🦠",
                    "medications": "💊", "risk_factors": "⚠️"}.get(category, "•")
            print(f"   {icon} {category:<15s}: {m}")

        # ── Show JSON output ──────────────────────────────────────────────────
        if verbose:
            print(f"\n📋  JSON (simple format):")
            print(entities_to_json(result, simple=True))

    log.info(f"\n{'=' * 70}")
    log.info("  DEMO COMPLETE")
    log.info(f"{'=' * 70}")


# ==============================================================================
# SECTION 11 — SMOKE TEST (NO NETWORK NEEDED FOR GAZETTEER-ONLY MODE)
# ==============================================================================

def _run_smoke_test() -> None:
    """
    Quick validation of the pipeline components.
    Uses gazetteer-only mode to avoid network dependency.
    """
    log.info("=" * 60)
    log.info("  CLINICAL NER SMOKE TEST")
    log.info("=" * 60)

    # ── Test 1: Text cleaner ──────────────────────────────────────────────────
    log.info("  Test 1: ClinicalTextCleaner…")
    cleaner = ClinicalTextCleaner()
    raw = "pt c/o  SOB   x3d.\nHTN.DM\n\n\n\nfever"
    cleaned = cleaner.clean(raw)
    assert len(cleaned) > 0
    expanded = cleaner.expand_abbreviations(cleaned)
    assert "shortness of breath" in expanded.lower()
    assert "hypertension" in expanded.lower()
    log.info(f"  ✓ Clean: '{cleaned}' → Expanded: '{expanded[:60]}…'")

    # ── Test 2: Gazetteer matcher ─────────────────────────────────────────────
    log.info("  Test 2: GazetteerMatcher…")
    gaz = GazetteerMatcher()
    matches = gaz.match(
        "Patient has pneumonia with fever and cough. "
        "Taking amoxicillin. History of smoking."
    )
    assert len(matches["diseases"]) >= 1, "Should find pneumonia"
    assert len(matches["symptoms"]) >= 1, "Should find fever or cough"
    assert len(matches["medications"]) >= 1, "Should find amoxicillin"
    assert len(matches["risk_factors"]) >= 1, "Should find smoking"
    log.info(f"  ✓ Diseases: {[e['text'] for e in matches['diseases']]}")
    log.info(f"  ✓ Symptoms: {[e['text'] for e in matches['symptoms']]}")
    log.info(f"  ✓ Medications: {[e['text'] for e in matches['medications']]}")
    log.info(f"  ✓ Risk factors: {[e['text'] for e in matches['risk_factors']]}")

    # ── Test 3: Entity categorization ─────────────────────────────────────────
    log.info("  Test 3: Entity categorization…")
    assert gaz.categorize_entity("pneumonia") == "diseases"
    assert gaz.categorize_entity("fever") == "symptoms"
    assert gaz.categorize_entity("amoxicillin") == "medications"
    assert gaz.categorize_entity("smoking") == "risk_factors"
    log.info("  ✓ All categorizations correct")

    # ── Test 4: Merge logic ───────────────────────────────────────────────────
    log.info("  Test 4: Entity merging…")
    ner_ents = {
        "symptoms": [{"text": "fever", "start": 0, "end": 5, "confidence": 0.95}],
        "diseases": [],
        "medications": [],
        "risk_factors": [],
    }
    gaz_ents = {
        "symptoms": [{"text": "fever", "start": 0, "end": 5, "confidence": 0.85}],
        "diseases": [{"text": "pneumonia", "start": 10, "end": 19, "confidence": 0.85}],
        "medications": [],
        "risk_factors": [],
    }
    merged = merge_entities(ner_ents, gaz_ents)
    assert len(merged["symptoms"]) == 1, "Duplicate fever should be merged"
    assert len(merged["diseases"]) == 1, "Pneumonia from gazetteer should be kept"
    log.info(f"  ✓ Merged: {sum(len(v) for v in merged.values())} unique entities")

    # ── Test 5: Evaluation metrics ────────────────────────────────────────────
    log.info("  Test 5: Evaluation metrics…")
    predicted = {
        "symptoms": [
            {"text": "fever", "start": 0, "end": 5, "confidence": 0.9},
            {"text": "cough", "start": 10, "end": 15, "confidence": 0.8},
        ],
        "diseases": [
            {"text": "pneumonia", "start": 20, "end": 29, "confidence": 0.95},
        ],
        "medications": [],
        "risk_factors": [],
    }
    gold = {
        "symptoms": ["fever", "cough", "headache"],  # 2/3 recalled
        "diseases": ["pneumonia"],                     # 1/1
        "medications": ["amoxicillin"],                # 0/1
        "risk_factors": [],
    }
    metrics = evaluate_extraction(predicted, gold, match_mode="exact")
    assert metrics["symptoms"].true_positives == 2
    assert metrics["symptoms"].false_negatives == 1
    assert metrics["diseases"].f1 == 1.0
    assert metrics["medications"].recall == 0.0
    log.info(f"  ✓ Symptoms: {metrics['symptoms']}")
    log.info(f"  ✓ Diseases: {metrics['diseases']}")
    log.info(f"  ✓ Medications: {metrics['medications']}")

    # ── Test 6: Format & JSON output ──────────────────────────────────────────
    log.info("  Test 6: Output formatting…")
    formatted = format_entities(predicted)
    assert "fever" in formatted
    json_out = entities_to_json(predicted, simple=True)
    parsed = json.loads(json_out)
    assert "symptoms" in parsed
    log.info("  ✓ Formatting and JSON output OK")

    # ── Test 7: extract_entities with gazetteer-only ──────────────────────────
    log.info("  Test 7: extract_entities (gazetteer-only fallback)…")
    # Create a dummy NER manager that returns nothing (simulates model failure)
    class DummyNER:
        model_name = "dummy"
        def predict(self, text): return []

    result = extract_entities(
        "Patient has pneumonia with fever. On amoxicillin. Smoking history.",
        model=DummyNER(),
    )
    assert len(result["diseases"]) >= 1
    assert len(result["symptoms"]) >= 1
    assert len(result["medications"]) >= 1
    assert len(result["risk_factors"]) >= 1
    log.info(f"  ✓ Gazetteer-only extraction found entities in all 4 categories")

    log.info("\n  ALL CLINICAL NER SMOKE TESTS PASSED ✓")


# ==============================================================================
# SECTION 12 — CLI ENTRY POINT
# ==============================================================================

def main():
    """
    Command-line interface for clinical entity extraction.

    Usage:
        # Extract from text
        python clinical_ner.py --text "Patient has pneumonia with fever."

        # Extract from file
        python clinical_ner.py --file clinical_note.txt

        # Run demo with evaluation
        python clinical_ner.py --demo

        # Smoke test
        python clinical_ner.py --smoke-test

        # Save results to JSON
        python clinical_ner.py --text "..." --output results.json
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Clinical NER — Medical Entity Extraction from Clinical Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=main.__doc__,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", type=str, help="Clinical text to process.")
    group.add_argument("--file", type=str, help="Path to text file with clinical note.")
    group.add_argument("--demo", action="store_true", help="Run demo with examples.")
    group.add_argument("--smoke-test", action="store_true", help="Run smoke tests.")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"HuggingFace model name. Default: {DEFAULT_MODEL}")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON results to file.")
    parser.add_argument("--simple", action="store_true",
                        help="Output simple format (text lists only, no metadata).")
    parser.add_argument("--no-abbreviations", action="store_true",
                        help="Skip abbreviation expansion.")
    parser.add_argument("--no-gazetteer", action="store_true",
                        help="Skip dictionary-based matching (model only).")

    args = parser.parse_args()

    # ── Smoke test ────────────────────────────────────────────────────────────
    if args.smoke_test:
        _run_smoke_test()
        return

    # ── Demo ──────────────────────────────────────────────────────────────────
    if args.demo:
        run_demo()
        return

    # ── Get input text ────────────────────────────────────────────────────────
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r") as f:
            text = f.read()
    else:
        parser.error("Provide --text, --file, --demo, or --smoke-test.")
        return

    # ── Extract entities ──────────────────────────────────────────────────────
    result = extract_entities(
        text,
        model_name=args.model,
        expand_abbrevs=not args.no_abbreviations,
        use_gazetteer=not args.no_gazetteer,
    )

    # ── Output ────────────────────────────────────────────────────────────────
    print(format_entities(result, include_metadata=True))
    print(f"\n{entities_to_json(result, simple=args.simple)}")

    # ── Save to file ──────────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w") as f:
            f.write(entities_to_json(result, simple=args.simple))
        log.info(f"Results saved → {args.output}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
