"""
================================================================================
  STEP 5 — CLINICAL TEXT SUMMARIZATION
  Chest X-Ray Multi-Modal Pipeline
  Module: Summarize clinical notes into concise medical summaries

  Uses: HuggingFace Transformers (BART / T5 backbone)
================================================================================

PURPOSE:
  Takes verbose clinical notes (admission notes, radiology reports, discharge
  summaries) and produces concise 2-3 sentence summaries that preserve the
  key medical findings, diagnoses, and treatment decisions.

MODELS SUPPORTED (selectable):
  1. "facebook/bart-large-cnn"        — Best quality, larger (~1.6GB)
  2. "sshleifer/distilbart-cnn-12-6"  — Fast + good quality (~1.2GB)
  3. "google/flan-t5-base"            — Versatile T5 variant (~990MB)
  4. "Falconsai/medical_summarization" — Medical-domain fine-tuned

QUICK START:
  from clinical_summarizer import summarize_text
  summary = summarize_text("Patient presents with fever and cough...")
  print(summary)

DEPENDS ON:
  pip install transformers torch sentencepiece
================================================================================
"""

# ==============================================================================
# SECTION 0 — IMPORTS
# ==============================================================================

import re
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
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
# SECTION 1 — CONFIGURATION
# ==============================================================================

@dataclass
class SummarizerConfig:
    """
    All summarizer hyperparameters in one place.

    Attributes:
        model_name    : HuggingFace model identifier.
        max_input_len : Maximum input token length (longer text is truncated).
        min_output_len: Minimum summary length in tokens.
        max_output_len: Maximum summary length in tokens.
        num_beams     : Beam search width (higher = better quality, slower).
        length_penalty: > 1.0 favors longer summaries, < 1.0 favors shorter.
        no_repeat_ngram: Prevents repeated n-grams in output.
        early_stopping: Stop beam search when all beams reach EOS.
        device        : "auto", "cpu", "cuda", or "mps".
        use_half_precision: Use FP16 for faster inference on GPU.
    """
    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "sshleifer/distilbart-cnn-12-6"  # Good balance of speed & quality

    # ── Generation parameters ─────────────────────────────────────────────────
    max_input_len: int    = 1024
    min_output_len: int   = 30       # ~1 sentence minimum
    max_output_len: int   = 150      # ~2-3 sentences maximum
    num_beams: int        = 4
    length_penalty: float = 1.0
    no_repeat_ngram: int  = 3
    early_stopping: bool  = True

    # ── Inference optimization ────────────────────────────────────────────────
    device: str                = "auto"
    use_half_precision: bool   = True   # FP16 on GPU for 2x speedup
    batch_size: int            = 4      # Batch size for batch processing

    # ── Clinical text options ─────────────────────────────────────────────────
    add_clinical_prompt: bool  = True   # Prepend a medical-context prompt for T5


# Available model presets for easy switching
MODEL_PRESETS: Dict[str, Dict] = {
    "fast": {
        "model_name": "sshleifer/distilbart-cnn-12-6",
        "num_beams": 2,
        "max_output_len": 120,
        "description": "DistilBART — fastest inference, good quality",
    },
    "quality": {
        "model_name": "facebook/bart-large-cnn",
        "num_beams": 4,
        "max_output_len": 150,
        "description": "BART-Large — best quality, slower",
    },
    "balanced": {
        "model_name": "sshleifer/distilbart-cnn-12-6",
        "num_beams": 4,
        "max_output_len": 150,
        "description": "DistilBART with beam search — balanced",
    },
    "t5": {
        "model_name": "google/flan-t5-base",
        "num_beams": 4,
        "max_output_len": 150,
        "description": "Flan-T5 — versatile, instruction-tuned",
    },
}


# ==============================================================================
# SECTION 2 — CLINICAL TEXT PREPROCESSOR
# ==============================================================================

class ClinicalPreprocessor:
    """
    Prepares clinical notes for summarization by:
      1. Cleaning formatting artifacts
      2. Normalizing section headers
      3. Removing redundant template text
      4. Truncating to fit model context window

    Clinical notes often contain template boilerplate, inconsistent
    formatting, and section markers that confuse summarization models.
    """

    # Section headers commonly found in clinical notes
    SECTION_HEADERS = [
        "CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "HPI",
        "PAST MEDICAL HISTORY", "PMH", "MEDICATIONS", "MEDS",
        "ALLERGIES", "SOCIAL HISTORY", "FAMILY HISTORY",
        "REVIEW OF SYSTEMS", "ROS", "PHYSICAL EXAM", "PE",
        "ASSESSMENT", "PLAN", "ASSESSMENT AND PLAN", "A/P",
        "IMPRESSION", "FINDINGS", "CLINICAL HISTORY",
        "TECHNIQUE", "COMPARISON", "RECOMMENDATION",
        "DISCHARGE DIAGNOSIS", "DISCHARGE MEDICATIONS",
        "DISCHARGE INSTRUCTIONS", "FOLLOW UP",
        "LABORATORY DATA", "LABS", "VITALS", "VITAL SIGNS",
        "IMAGING", "RADIOLOGY", "PROCEDURES",
    ]

    @classmethod
    def preprocess(cls, text: str, max_chars: int = 4000) -> str:
        """
        Clean and prepare clinical text for summarization.

        Args:
            text      : Raw clinical note.
            max_chars : Maximum character length (soft limit for context window).

        Returns:
            Cleaned text ready for the summarization model.
        """
        if not text or not text.strip():
            return ""

        # ── Remove common template artifacts ──────────────────────────────────
        # EMR copy-paste artifacts
        text = re.sub(r"\*{3,}", "", text)           # ***
        text = re.sub(r"-{5,}", " ", text)           # -----
        text = re.sub(r"={5,}", " ", text)           # =====
        text = re.sub(r"_{5,}", " ", text)           # _____

        # Remove timestamps and EMR metadata lines
        text = re.sub(
            r"(?:Signed|Reviewed|Addendum|Cosigned)\s+by\s+.*?\d{4}",
            "", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*(?:AM|PM)?",
            "", text, flags=re.IGNORECASE
        )

        # ── Normalize section headers ─────────────────────────────────────────
        for header in cls.SECTION_HEADERS:
            # Convert standalone headers to inline markers
            pattern = re.compile(
                rf"^\s*{re.escape(header)}\s*:?\s*$",
                re.MULTILINE | re.IGNORECASE,
            )
            text = pattern.sub(f"\n{header}: ", text)

        # ── Clean whitespace ──────────────────────────────────────────────────
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)

        # ── Strip lines that are just numbers or short labels ─────────────────
        lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            # Keep lines with actual content
            if stripped and len(stripped) > 3:
                lines.append(stripped)

        text = " ".join(lines)

        # ── Truncate if too long ──────────────────────────────────────────────
        if len(text) > max_chars:
            # Try to truncate at a sentence boundary
            truncated = text[:max_chars]
            last_period = truncated.rfind(".")
            if last_period > max_chars * 0.7:
                truncated = truncated[:last_period + 1]
            text = truncated

        return text.strip()


# ==============================================================================
# SECTION 3 — SUMMARIZER ENGINE
# ==============================================================================

class ClinicalSummarizer:
    """
    Production-ready clinical text summarization engine.

    Wraps a HuggingFace seq2seq model (BART/T5) with optimizations:
      - Lazy model loading (load on first use)
      - FP16 inference on GPU for 2x speed
      - Intelligent text preprocessing for clinical notes
      - Batch processing with automatic padding
      - Configurable generation parameters

    Usage:
        summarizer = ClinicalSummarizer()
        summary = summarizer.summarize("Patient presents with...")
        summaries = summarizer.summarize_batch(["note1...", "note2..."])
    """

    def __init__(self, config: Optional[SummarizerConfig] = None, preset: Optional[str] = None):
        """
        Args:
            config : SummarizerConfig instance. If None, uses defaults.
            preset : One of "fast", "quality", "balanced", "t5".
                     Overrides config.model_name and generation params.
        """
        self.config = config or SummarizerConfig()

        # Apply preset if specified
        if preset and preset in MODEL_PRESETS:
            preset_cfg = MODEL_PRESETS[preset]
            self.config.model_name = preset_cfg["model_name"]
            self.config.num_beams = preset_cfg["num_beams"]
            self.config.max_output_len = preset_cfg["max_output_len"]
            log.info(f"Using preset '{preset}': {preset_cfg['description']}")

        # ── Resolve device ────────────────────────────────────────────────────
        self.device = self._resolve_device(self.config.device)
        log.info(f"Summarizer device: {self.device}")

        # ── Model components (lazy-loaded) ────────────────────────────────────
        self._tokenizer = None
        self._model = None
        self._pipeline = None
        self._preprocessor = ClinicalPreprocessor()

        # ── Performance tracking ──────────────────────────────────────────────
        self._inference_times: List[float] = []

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)

    # ── Lazy loading ──────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load tokenizer and model on first use."""
        if self._model is not None:
            return

        model_name = self.config.model_name
        log.info(f"Loading summarization model: {model_name}")

        t0 = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # ── Move to device + optimize ─────────────────────────────────────────
        self._model = self._model.to(self.device)
        self._model.eval()

        # FP16 optimization (GPU only, not for MPS which has limited FP16 support)
        if self.config.use_half_precision and self.device.type == "cuda":
            self._model = self._model.half()
            log.info("FP16 precision enabled for GPU inference.")

        load_time = time.time() - t0
        n_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        log.info(
            f"Model loaded in {load_time:.1f}s | "
            f"{n_params:.0f}M parameters | device: {self.device}"
        )

    def _is_t5_model(self) -> bool:
        """Check if the loaded model is a T5 variant."""
        return "t5" in self.config.model_name.lower()

    # ── Core summarization ────────────────────────────────────────────────────

    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> str:
        """
        Summarize a single clinical note.

        Args:
            text       : Clinical note to summarize.
            max_length : Override max output length.
            min_length : Override min output length.
            num_beams  : Override beam search width.

        Returns:
            Concise 2-3 sentence summary preserving key medical meaning.

        Example:
            >>> summarizer = ClinicalSummarizer()
            >>> summary = summarizer.summarize(
            ...     "65 y/o male presents to ED with chief complaint of SOB "
            ...     "and productive cough for 3 days. PMH: HTN, DM, COPD. "
            ...     "CXR shows bilateral infiltrates consistent with pneumonia. "
            ...     "Started on levofloxacin 750mg IV."
            ... )
            >>> print(summary)
            "A 65-year-old male presented with shortness of breath and cough.
             Imaging revealed bilateral infiltrates consistent with pneumonia.
             Treatment with IV levofloxacin was initiated."
        """
        self._load_model()

        # ── Preprocess ────────────────────────────────────────────────────────
        cleaned = self._preprocessor.preprocess(text)
        if not cleaned:
            return ""

        # ── Add clinical prompt for T5 models ─────────────────────────────────
        if self._is_t5_model() and self.config.add_clinical_prompt:
            cleaned = f"summarize the following clinical note: {cleaned}"

        # ── Tokenize ──────────────────────────────────────────────────────────
        t0 = time.time()

        inputs = self._tokenizer(
            cleaned,
            max_length=self.config.max_input_len,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        ).to(self.device)

        # ── Generate summary ──────────────────────────────────────────────────
        gen_kwargs = {
            "max_length": max_length or self.config.max_output_len,
            "min_length": min_length or self.config.min_output_len,
            "num_beams": num_beams or self.config.num_beams,
            "length_penalty": self.config.length_penalty,
            "no_repeat_ngram_size": self.config.no_repeat_ngram,
            "early_stopping": self.config.early_stopping,
        }

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

        # ── Decode ────────────────────────────────────────────────────────────
        summary = self._tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # ── Post-process ──────────────────────────────────────────────────────
        summary = self._postprocess(summary)

        elapsed = time.time() - t0
        self._inference_times.append(elapsed)
        log.debug(f"Summarization took {elapsed:.2f}s | {len(summary)} chars")

        return summary

    def summarize_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Summarize multiple clinical notes efficiently.

        Processes texts in batches for better throughput. Handles variable-
        length inputs with dynamic padding.

        Args:
            texts         : List of clinical notes.
            max_length    : Override max output length.
            min_length    : Override min output length.
            show_progress : Log progress during processing.

        Returns:
            List of summaries, one per input text.

        Example:
            >>> notes = [
            ...     "Patient 1: presents with chest pain...",
            ...     "Patient 2: 72 y/o female with SOB...",
            ... ]
            >>> summaries = summarizer.summarize_batch(notes)
        """
        self._load_model()

        if not texts:
            return []

        summaries = []
        n_texts = len(texts)
        batch_size = self.config.batch_size

        t0 = time.time()

        for batch_start in range(0, n_texts, batch_size):
            batch_end = min(batch_start + batch_size, n_texts)
            batch_texts = texts[batch_start:batch_end]

            if show_progress:
                log.info(
                    f"Processing batch [{batch_start + 1}–{batch_end}] "
                    f"of {n_texts} notes…"
                )

            # ── Preprocess batch ──────────────────────────────────────────────
            cleaned_batch = []
            for text in batch_texts:
                cleaned = self._preprocessor.preprocess(text)
                if self._is_t5_model() and self.config.add_clinical_prompt:
                    cleaned = f"summarize the following clinical note: {cleaned}"
                cleaned_batch.append(cleaned if cleaned else "empty note")

            # ── Tokenize batch ────────────────────────────────────────────────
            inputs = self._tokenizer(
                cleaned_batch,
                max_length=self.config.max_input_len,
                truncation=True,
                padding="longest",
                return_tensors="pt",
            ).to(self.device)

            # ── Generate ──────────────────────────────────────────────────────
            gen_kwargs = {
                "max_length": max_length or self.config.max_output_len,
                "min_length": min_length or self.config.min_output_len,
                "num_beams": self.config.num_beams,
                "length_penalty": self.config.length_penalty,
                "no_repeat_ngram_size": self.config.no_repeat_ngram,
                "early_stopping": self.config.early_stopping,
            }

            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs,
                )

            # ── Decode batch ──────────────────────────────────────────────────
            batch_summaries = self._tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # ── Post-process ──────────────────────────────────────────────────
            batch_summaries = [self._postprocess(s) for s in batch_summaries]
            summaries.extend(batch_summaries)

        total_time = time.time() - t0
        avg_time = total_time / n_texts if n_texts > 0 else 0

        log.info(
            f"Batch summarization complete: {n_texts} notes in {total_time:.1f}s "
            f"(avg {avg_time:.2f}s/note)"
        )

        return summaries

    # ── Post-processing ───────────────────────────────────────────────────────

    @staticmethod
    def _postprocess(summary: str) -> str:
        """
        Clean up model output:
          - Fix spacing around punctuation
          - Capitalize first letter
          - Ensure ends with period
          - Remove any leftover artifacts
        """
        if not summary:
            return ""

        # Strip leading/trailing whitespace first
        summary = summary.strip()

        if not summary:
            return ""

        # Fix double spaces
        summary = re.sub(r"\s{2,}", " ", summary)

        # Fix spacing before punctuation
        summary = re.sub(r"\s+([.,;:!?])", r"\1", summary)

        # Ensure starts with uppercase
        summary = summary[0].upper() + summary[1:]

        # Ensure ends with period
        summary = summary.strip()
        if summary and summary[-1] not in ".!?":
            summary += "."

        return summary

    # ── Performance stats ─────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return inference performance statistics."""
        if not self._inference_times:
            return {"total_inferences": 0}

        times = self._inference_times
        return {
            "total_inferences": len(times),
            "avg_time_sec": sum(times) / len(times),
            "min_time_sec": min(times),
            "max_time_sec": max(times),
            "total_time_sec": sum(times),
            "model": self.config.model_name,
            "device": str(self.device),
        }


# ==============================================================================
# SECTION 4 — SIMPLE API: summarize_text()
# ==============================================================================

# Module-level singleton for convenience API
_summarizer: Optional[ClinicalSummarizer] = None


def summarize_text(
    text: str,
    model_name: Optional[str] = None,
    preset: Optional[str] = None,
    max_length: int = 150,
    min_length: int = 30,
) -> str:
    """
    Summarize a clinical note into a concise 2-3 sentence summary.

    This is the primary convenience API. The model is loaded lazily on first
    call and cached for subsequent calls.

    Args:
        text       : Clinical note text to summarize.
        model_name : HuggingFace model ID (default: distilbart-cnn-12-6).
        preset     : One of "fast", "quality", "balanced", "t5".
        max_length : Maximum summary length in tokens (~150 ≈ 2-3 sentences).
        min_length : Minimum summary length in tokens.

    Returns:
        Concise medical summary as a string.

    Example:
        >>> from clinical_summarizer import summarize_text
        >>> summary = summarize_text(
        ...     "65 y/o male presents with SOB and productive cough x3d. "
        ...     "PMH: HTN, COPD. CXR: bilateral infiltrates c/w pneumonia. "
        ...     "Started levofloxacin 750mg IV. O2 sat 89% on RA."
        ... )
        >>> print(summary)
    """
    global _summarizer

    # Re-create if model changed
    if _summarizer is None or (model_name and _summarizer.config.model_name != model_name):
        config = SummarizerConfig()
        if model_name:
            config.model_name = model_name
        _summarizer = ClinicalSummarizer(config=config, preset=preset)

    return _summarizer.summarize(text, max_length=max_length, min_length=min_length)


def summarize_batch(
    texts: List[str],
    model_name: Optional[str] = None,
    preset: Optional[str] = None,
) -> List[str]:
    """
    Summarize multiple clinical notes.

    Args:
        texts      : List of clinical note strings.
        model_name : HuggingFace model ID.
        preset     : Model preset name.

    Returns:
        List of summary strings.
    """
    global _summarizer

    if _summarizer is None or (model_name and _summarizer.config.model_name != model_name):
        config = SummarizerConfig()
        if model_name:
            config.model_name = model_name
        _summarizer = ClinicalSummarizer(config=config, preset=preset)

    return _summarizer.summarize_batch(texts)


# ==============================================================================
# SECTION 5 — EXAMPLE CLINICAL NOTES
# ==============================================================================

EXAMPLE_NOTES = [
    # ── Example 1: ED admission note ──────────────────────────────────────────
    {
        "title": "Emergency Department — Pneumonia",
        "text": (
            "CHIEF COMPLAINT: Shortness of breath and productive cough.\n\n"
            "HISTORY OF PRESENT ILLNESS:\n"
            "65-year-old male presents to the emergency department with a 3-day "
            "history of progressively worsening shortness of breath and productive "
            "cough with yellowish sputum. Patient reports associated fever measured "
            "at 101.2°F at home, chills, and night sweats. He denies hemoptysis, "
            "chest pain, or recent travel. Symptoms started gradually and have been "
            "getting worse despite over-the-counter cough suppressants.\n\n"
            "PAST MEDICAL HISTORY:\n"
            "1. Hypertension — diagnosed 2015, on lisinopril\n"
            "2. Type 2 Diabetes Mellitus — on metformin, HbA1c 7.2%\n"
            "3. COPD — moderate, on tiotropium inhaler\n"
            "4. Former smoker — quit 2019, 30 pack-year history\n\n"
            "PHYSICAL EXAM:\n"
            "Vitals: T 101.8°F, HR 98, BP 142/88, RR 24, O2 sat 89% on RA\n"
            "General: Appears ill, mild respiratory distress\n"
            "Lungs: Decreased breath sounds right lower lobe, crackles bilateral bases\n"
            "Heart: Tachycardic, regular rhythm, no murmurs\n\n"
            "IMAGING:\n"
            "Chest X-ray: Bilateral infiltrates, right lower lobe consolidation.\n"
            "No pleural effusion or pneumothorax.\n\n"
            "ASSESSMENT AND PLAN:\n"
            "1. Community-acquired pneumonia — start levofloxacin 750mg IV daily\n"
            "2. Acute on chronic respiratory failure — supplemental O2 via nasal "
            "cannula, target SpO2 92-96%\n"
            "3. COPD exacerbation — add albuterol/ipratropium nebulizer q4h\n"
            "4. Continue home medications\n"
            "5. Admit to medical floor, monitor respiratory status"
        ),
    },

    # ── Example 2: Radiology report ───────────────────────────────────────────
    {
        "title": "Radiology Report — Chest CT",
        "text": (
            "CLINICAL HISTORY: 58-year-old female with chronic cough and "
            "weight loss. Rule out malignancy.\n\n"
            "TECHNIQUE: CT chest with IV contrast, 1.25mm axial reconstructions.\n\n"
            "COMPARISON: Chest X-ray from 2 weeks prior.\n\n"
            "FINDINGS:\n"
            "Lungs: A 3.2 x 2.8 cm spiculated mass is identified in the right "
            "upper lobe, suspicious for primary lung malignancy. Multiple sub-"
            "centimeter pulmonary nodules are seen bilaterally (largest 8mm in "
            "left lower lobe), concerning for metastatic disease. Mild "
            "centrilobular emphysema is noted bilaterally.\n\n"
            "Mediastinum: Enlarged right paratracheal (1.8cm) and subcarinal "
            "(2.2cm) lymph nodes, likely pathologic. Small pericardial effusion.\n\n"
            "Pleura: Small right-sided pleural effusion. No pneumothorax.\n\n"
            "Bones: No suspicious osseous lesions.\n\n"
            "IMPRESSION:\n"
            "1. Right upper lobe spiculated mass highly suspicious for primary "
            "lung carcinoma.\n"
            "2. Bilateral pulmonary nodules and mediastinal lymphadenopathy "
            "concerning for metastatic disease.\n"
            "3. Small right pleural effusion and pericardial effusion.\n"
            "RECOMMENDATION: Tissue sampling recommended. Correlate with PET-CT "
            "for staging."
        ),
    },

    # ── Example 3: Discharge summary ──────────────────────────────────────────
    {
        "title": "Discharge Summary — Heart Failure",
        "text": (
            "DISCHARGE DIAGNOSIS: Acute decompensated heart failure (CHF "
            "exacerbation), atrial fibrillation with rapid ventricular response.\n\n"
            "HOSPITAL COURSE:\n"
            "72-year-old female admitted with progressive dyspnea, orthopnea, "
            "and bilateral lower extremity edema over 2 weeks. Initial BNP was "
            "2,340 pg/mL. Chest X-ray showed cardiomegaly and bilateral pleural "
            "effusions. Echocardiogram revealed ejection fraction of 25% (prior "
            "was 35% six months ago), moderate mitral regurgitation, and dilated "
            "left ventricle.\n\n"
            "Patient was started on IV furosemide with good diuretic response, "
            "losing 8 kg of fluid weight over 5 days. Atrial fibrillation was "
            "rate-controlled with IV diltiazem, then transitioned to oral "
            "metoprolol 50mg BID. Anticoagulation with apixaban 5mg BID was "
            "initiated.\n\n"
            "DISCHARGE MEDICATIONS:\n"
            "1. Furosemide 40mg PO BID\n"
            "2. Metoprolol succinate 50mg PO daily\n"
            "3. Lisinopril 10mg PO daily\n"
            "4. Apixaban 5mg PO BID\n"
            "5. Spironolactone 25mg PO daily\n\n"
            "FOLLOW UP: Cardiology clinic in 1 week, telemetry monitoring at home."
        ),
    },
]


# ==============================================================================
# SECTION 6 — DEMO & SMOKE TEST
# ==============================================================================

def run_demo(preset: str = "fast") -> None:
    """
    Run summarization on example clinical notes.
    Downloads and uses the model — requires network on first run.
    """
    log.info("=" * 70)
    log.info("  CLINICAL SUMMARIZATION DEMO")
    log.info("=" * 70)

    summarizer = ClinicalSummarizer(preset=preset)

    for idx, example in enumerate(EXAMPLE_NOTES, 1):
        title = example["title"]
        text = example["text"]

        log.info(f"\n{'─' * 70}")
        log.info(f"  EXAMPLE {idx}: {title}")
        log.info(f"{'─' * 70}")
        log.info(f"  Input length: {len(text)} chars")
        log.info(f"  Input preview: {text[:100]}…")

        summary = summarizer.summarize(text)

        print(f"\n📝  Summary:")
        print(f"   {summary}")
        print(f"   ({len(summary)} chars)")

    # Show performance stats
    stats = summarizer.get_stats()
    log.info(f"\n📊  Performance:")
    log.info(f"  Avg inference time: {stats.get('avg_time_sec', 0):.2f}s")
    log.info(f"  Model: {stats.get('model', 'N/A')}")
    log.info(f"  Device: {stats.get('device', 'N/A')}")


def _run_smoke_test() -> None:
    """
    Test pipeline components WITHOUT downloading any model.
    Validates preprocessing, postprocessing, config, and structure.
    """
    log.info("=" * 60)
    log.info("  CLINICAL SUMMARIZER SMOKE TEST")
    log.info("=" * 60)

    # ── Test 1: ClinicalPreprocessor ──────────────────────────────────────────
    log.info("  Test 1: ClinicalPreprocessor…")
    preprocessor = ClinicalPreprocessor()

    raw_note = (
        "***CONFIDENTIAL***\n"
        "Signed by Dr. Smith 01/15/2024\n"
        "==============================\n"
        "CHIEF COMPLAINT\n"
        "Shortness of breath and productive cough.\n"
        "\n\n\n\n"
        "HISTORY OF PRESENT ILLNESS\n"
        "65 y/o male with 3 day history of SOB.  Multiple   spaces  here.\n"
        "_______________\n"
        "ASSESSMENT AND PLAN\n"
        "1. Community-acquired pneumonia - start levofloxacin.\n"
    )
    cleaned = preprocessor.preprocess(raw_note)
    assert len(cleaned) > 0
    assert "***" not in cleaned
    assert "=====" not in cleaned
    assert "_____" not in cleaned
    assert "Signed by" not in cleaned
    assert "  " not in cleaned  # No double spaces
    log.info(f"  ✓ Cleaned {len(raw_note)} → {len(cleaned)} chars")
    log.info(f"    Preview: {cleaned[:80]}…")

    # ── Test 2: Truncation ────────────────────────────────────────────────────
    log.info("  Test 2: Text truncation…")
    long_text = "This is a sentence. " * 500  # ~10000 chars
    truncated = preprocessor.preprocess(long_text, max_chars=500)
    assert len(truncated) <= 520  # Small buffer for sentence boundary
    assert truncated.endswith(".")
    log.info(f"  ✓ Truncated {len(long_text)} → {len(truncated)} chars")

    # ── Test 3: SummarizerConfig ──────────────────────────────────────────────
    log.info("  Test 3: SummarizerConfig…")
    config = SummarizerConfig()
    assert config.max_input_len == 1024
    assert config.min_output_len == 30
    assert config.max_output_len == 150
    assert config.num_beams == 4

    # Test preset override
    for preset_name, preset_cfg in MODEL_PRESETS.items():
        assert "model_name" in preset_cfg
        assert "description" in preset_cfg
    log.info(f"  ✓ Config OK | {len(MODEL_PRESETS)} presets available")

    # ── Test 4: Postprocessing ────────────────────────────────────────────────
    log.info("  Test 4: Postprocessing…")
    post = ClinicalSummarizer._postprocess

    assert post("hello world") == "Hello world."
    assert post("Hello world.") == "Hello world."
    assert post("  spaces   everywhere  ") == "Spaces everywhere."
    assert post("fix  punctuation .") == "Fix punctuation."
    assert post("") == ""
    log.info("  ✓ Postprocessing rules validated")

    # ── Test 5: Device resolution ─────────────────────────────────────────────
    log.info("  Test 5: Device resolution…")
    device = ClinicalSummarizer._resolve_device("cpu")
    assert device == torch.device("cpu")
    device_auto = ClinicalSummarizer._resolve_device("auto")
    assert device_auto.type in ("cpu", "cuda", "mps")
    log.info(f"  ✓ Auto device: {device_auto}")

    # ── Test 6: Summarizer initialization (without loading model) ─────────────
    log.info("  Test 6: Summarizer initialization…")
    summarizer = ClinicalSummarizer(preset="fast")
    assert summarizer.config.model_name == "sshleifer/distilbart-cnn-12-6"
    assert summarizer._model is None  # Not loaded yet (lazy)
    stats = summarizer.get_stats()
    assert stats["total_inferences"] == 0
    log.info(f"  ✓ Lazy initialization OK (model not loaded)")

    # ── Test 7: Example notes structure ───────────────────────────────────────
    log.info("  Test 7: Example notes…")
    assert len(EXAMPLE_NOTES) >= 3
    for ex in EXAMPLE_NOTES:
        assert "title" in ex
        assert "text" in ex
        assert len(ex["text"]) > 100
    log.info(f"  ✓ {len(EXAMPLE_NOTES)} example notes validated")

    # ── Test 8: API function existence ────────────────────────────────────────
    log.info("  Test 8: API surface…")
    assert callable(summarize_text)
    assert callable(summarize_batch)
    assert callable(run_demo)
    log.info("  ✓ All API functions available")

    log.info("\n  ALL SUMMARIZER SMOKE TESTS PASSED ✓")


# ==============================================================================
# SECTION 7 — CLI ENTRY POINT
# ==============================================================================

def main():
    """
    Command-line interface for clinical text summarization.

    Usage:
        # Summarize text directly
        python clinical_summarizer.py --text "Patient presents with fever..."

        # Summarize from file
        python clinical_summarizer.py --file discharge_note.txt

        # Use a specific model preset
        python clinical_summarizer.py --text "..." --preset quality

        # Batch summarize multiple files
        python clinical_summarizer.py --files note1.txt note2.txt --output-dir ./summaries

        # Run demo with examples
        python clinical_summarizer.py --demo

        # Smoke test (no download needed)
        python clinical_summarizer.py --smoke-test
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Clinical Text Summarization — Concise Medical Summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=main.__doc__,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", type=str, help="Clinical text to summarize.")
    group.add_argument("--file", type=str, help="Path to text file with clinical note.")
    group.add_argument("--files", type=str, nargs="+", help="Multiple text files (batch mode).")
    group.add_argument("--demo", action="store_true", help="Run demo with example notes.")
    group.add_argument("--smoke-test", action="store_true", help="Run smoke tests (no model download).")

    parser.add_argument("--preset", type=str, default="fast",
                        choices=list(MODEL_PRESETS.keys()),
                        help="Model preset: fast, quality, balanced, t5. Default: fast")
    parser.add_argument("--model", type=str, default=None,
                        help="Override HuggingFace model name.")
    parser.add_argument("--max-length", type=int, default=150,
                        help="Max summary length in tokens. Default: 150")
    parser.add_argument("--min-length", type=int, default=30,
                        help="Min summary length in tokens. Default: 30")
    parser.add_argument("--output", type=str, default=None,
                        help="Save summary to file.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for batch mode.")

    args = parser.parse_args()

    # ── Smoke test ────────────────────────────────────────────────────────────
    if args.smoke_test:
        _run_smoke_test()
        return

    # ── Demo ──────────────────────────────────────────────────────────────────
    if args.demo:
        run_demo(preset=args.preset)
        return

    # ── Create summarizer ─────────────────────────────────────────────────────
    config = SummarizerConfig()
    if args.model:
        config.model_name = args.model
    summarizer = ClinicalSummarizer(config=config, preset=args.preset)

    # ── Single text ───────────────────────────────────────────────────────────
    if args.text:
        summary = summarizer.summarize(
            args.text,
            max_length=args.max_length,
            min_length=args.min_length,
        )
        print(f"\n📝  Summary:\n   {summary}\n")

        if args.output:
            Path(args.output).write_text(summary)
            log.info(f"Summary saved → {args.output}")

    # ── Single file ───────────────────────────────────────────────────────────
    elif args.file:
        text = Path(args.file).read_text()
        summary = summarizer.summarize(
            text,
            max_length=args.max_length,
            min_length=args.min_length,
        )
        print(f"\n📝  Summary of {args.file}:\n   {summary}\n")

        if args.output:
            Path(args.output).write_text(summary)
            log.info(f"Summary saved → {args.output}")

    # ── Batch files ───────────────────────────────────────────────────────────
    elif args.files:
        texts = [Path(f).read_text() for f in args.files]
        summaries = summarizer.summarize_batch(texts)

        for filepath, summary in zip(args.files, summaries):
            print(f"\n📝  {Path(filepath).name}:")
            print(f"   {summary}")

            if args.output_dir:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"summary_{Path(filepath).stem}.txt"
                out_path.write_text(summary)
                log.info(f"Saved → {out_path}")

        stats = summarizer.get_stats()
        print(f"\n📊  Avg time: {stats.get('avg_time_sec', 0):.2f}s/note")

    else:
        parser.error("Provide --text, --file, --files, --demo, or --smoke-test.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
