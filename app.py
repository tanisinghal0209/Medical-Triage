"""
==============================================================================
 FLASK API — Multi-Modal Clinical Decision Support Pipeline
==============================================================================

 Integrates all five ML components into a single REST API:
   1. chest_xray_model.py  → Disease probability predictions
   2. gradcam_xray.py      → Grad-CAM heatmap overlays
   3. clinical_ner.py       → Named Entity Recognition
   4. clinical_summarizer.py → Clinical text summarization
   5. triage_fusion.py      → Multi-modal triage scoring

 Endpoints:
   POST /predict           → Full pipeline (image + text)
   POST /predict/image     → Image-only (disease probs + heatmap)
   POST /predict/text      → Text-only (NER + summary)
   GET  /health            → Health check

 Usage:
   python app.py
   curl -X POST -F "image=@xray.jpg" -F "text=Patient has fever" http://localhost:5000/predict

==============================================================================
"""

import os
import io
import sys
import json
import time
import uuid
import base64
import logging
import traceback
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")

# ── Constants ─────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("./uploads")
HEATMAP_DIR = Path("./heatmaps")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "dcm"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# SECTION 1 — SERVICE LAYER (wraps each ML module)
# ==============================================================================

class ImageService:
    """Wraps chest_xray_model.py for disease prediction."""

    def __init__(self):
        self.model = None
        self.processor = None          # HF AutoImageProcessor (ConvNeXtV2 only)
        self._is_hf_model = False
        self._disease_classes = []     # populated during _load()
        self._num_classes = 0
        self._available = False
        self._load()

    def _load(self):
        try:
            import torch
            from torchvision import models
            
            # ConvNeXtV2 paths
            convnext_path = Path("./convnextv2-nih-results/checkpoint-2128")
            if not convnext_path.exists():
                # Fallback to the latest checkpoint if the expected one is absent
                checkpoints = sorted(Path("./convnextv2-nih-results").glob("checkpoint-*"))
                if checkpoints:
                    convnext_path = checkpoints[-1]

            if convnext_path.exists() and (convnext_path / "config.json").exists():
                log.info(f"🚀 Detected ConvNeXt V2 HF model at {convnext_path}")
                from transformers import AutoModelForImageClassification, AutoImageProcessor

                self.processor = AutoImageProcessor.from_pretrained(str(convnext_path))
                self.model = AutoModelForImageClassification.from_pretrained(str(convnext_path))

                # Store labels from the HF config — id2label is {"0": "Atelectasis", ...}
                id2label = self.model.config.id2label
                self._disease_classes = [
                    id2label[k] for k in sorted(id2label.keys(), key=lambda x: int(x))
                ]
                self._num_classes = len(self._disease_classes)

                device = torch.device(
                    "mps" if torch.backends.mps.is_available()
                    else "cuda" if torch.cuda.is_available()
                    else "cpu"
                )
                self.model.to(device)
                self.model.eval()
                self._is_hf_model = True
                log.info(f"✓ ConvNeXt V2 ({self._num_classes} classes) loaded → {device}")
                log.info(f"  Classes: {self._disease_classes}")
            else:
                log.error("✗ ConvNeXt V2 checkpoint not found!")
                self._available = False
                return

            self._available = True
        except Exception as e:
            log.error(f"✗ Image model unavailable: {e}")
            log.error(traceback.format_exc())
            self._available = False

    def predict(self, image_path: str) -> Dict[str, float]:
        if not self._available:
            raise RuntimeError("Image model not available")

        import torch

        if self._is_hf_model:
            # ── ConvNeXtV2 (Hugging Face) sigmoid multi-label inference ───────
            from PIL import Image as PILImage
            img = PILImage.open(image_path).convert("RGB")

            # Resolve device from model parameters
            device = next(self.model.parameters()).device
            inputs = self.processor(images=img, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

            results = {
                self._disease_classes[i]: float(probs[i])
                for i in range(len(self._disease_classes))
            }
            return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        else:
            raise RuntimeError("Model is available but not recognized as an HF model.")

    @property
    def available(self) -> bool:
        return self._available


class HeatmapService:
    """Wraps gradcam_xray.py for Grad-CAM heatmap generation."""

    def __init__(self, image_service: ImageService):
        self.image_service = image_service
        self._available = False
        self._load()

    def _load(self):
        try:
            from gradcam_xray import generate_heatmap
            self._generate_fn = generate_heatmap
            self._available = self.image_service.available
            if self._available:
                log.info("✓ Heatmap service ready")
            else:
                log.warning("⚠ Heatmap service: waiting for image model")
        except Exception as e:
            log.error(f"✗ Heatmap service unavailable: {e}")

    def generate(self, image_path: str, target_class: Optional[str] = None) -> str:
        """Generate heatmap overlay, return base64-encoded PNG string."""
        if not self._available:
            raise RuntimeError("Heatmap service not available")

        # ── Auto-detect Grad-CAM target layer based on model architecture ────
        model = self.image_service.model
        target_layer = None

        if hasattr(model, "convnextv2"):  # ConvNeXtV2ForImageClassification (HF)
            # HF ConvNeXtV2: model.convnextv2.encoder.stages is the correct path
            if hasattr(model.convnextv2, "encoder") and hasattr(model.convnextv2.encoder, "stages"):
                target_layer = model.convnextv2.encoder.stages[-1]
                log.info("Grad-CAM target: convnextv2.encoder.stages[-1]")
            else:
                # Older HF versions may not have encoder wrapper
                target_layer = model.convnextv2.stages[-1]
                log.info("Grad-CAM target: convnextv2.stages[-1] (fallback)")
        elif hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):  # ResNet50
            target_layer = model.backbone.layer4
        elif hasattr(model, "features"):  # MobileNetV2
            target_layer = model.features[-1]

        import torch
        # Grad-CAM REQUIRES gradients — disable torch.no_grad context here
        with torch.set_grad_enabled(True):
            overlay_image = self._generate_fn(
                image_path,
                model=model,
                target_class=target_class,
                target_layer=target_layer,
                class_names=self.image_service._disease_classes,
            )

        # Convert PIL Image → base64 PNG
        buf = io.BytesIO()
        overlay_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @property
    def available(self) -> bool:
        return self._available and self.image_service.available


class NERService:
    """Wraps clinical_ner.py for entity extraction."""

    def __init__(self):
        self._available = False
        self._load()

    def _load(self):
        try:
            from clinical_ner import extract_entities
            self._extract_fn = extract_entities
            self._available = True
            log.info("✓ NER service ready")
        except Exception as e:
            log.error(f"✗ NER service unavailable: {e}")

    def extract(self, text: str) -> Dict:
        if not self._available:
            raise RuntimeError("NER service not available")
        import torch
        with torch.inference_mode():
            return self._extract_fn(text)

    @property
    def available(self) -> bool:
        return self._available


class SummarizerService:
    """Wraps clinical_summarizer.py for text summarization."""

    def __init__(self):
        self._available = False
        self._load()

    def _load(self):
        try:
            from clinical_summarizer import summarize_text
            self._summarize_fn = summarize_text
            self._available = True
            log.info("✓ Summarizer service ready")
        except Exception as e:
            log.error(f"✗ Summarizer service unavailable: {e}")

    def summarize(self, text: str) -> str:
        if not self._available:
            raise RuntimeError("Summarizer service not available")
        import torch
        with torch.inference_mode():
            return self._summarize_fn(text)

    @property
    def available(self) -> bool:
        return self._available


class TriageService:
    """Wraps triage_fusion.py for multi-modal triage scoring."""

    def __init__(self):
        self._available = False
        self._load()

    def _load(self):
        try:
            from triage_fusion import predict_triage, estimate_urgency_rule_based
            self._predict_fn = predict_triage
            self._estimate_fn = estimate_urgency_rule_based
            self._available = True
            log.info("✓ Triage service ready")
        except Exception as e:
            log.error(f"✗ Triage service unavailable: {e}")

    def predict(
        self,
        image_probs: Dict[str, float],
        ner_entities: Dict,
    ) -> Dict:
        if not self._available:
            raise RuntimeError("Triage service not available")
        return self._predict_fn(image_probs, ner_entities)

    @property
    def available(self) -> bool:
        return self._available


# ==============================================================================
# SECTION 2 — FLASK APP FACTORY
# ==============================================================================

def create_app() -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    # ── Initialize services ───────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  Initializing ML Pipeline Services…")
    log.info("=" * 60)

    services = {}
    services["image"] = ImageService()
    services["heatmap"] = HeatmapService(services["image"])
    services["ner"] = NERService()
    services["summarizer"] = SummarizerService()
    services["triage"] = TriageService()

    log.info("─" * 60)
    for name, svc in services.items():
        status = "✓ READY" if svc.available else "✗ UNAVAILABLE"
        log.info(f"  {name:<15s}: {status}")
    log.info("=" * 60)

    # ── Helper functions ──────────────────────────────────────────────────────

    def _allowed_file(filename: str) -> bool:
        return "." in filename and \
               filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    def _save_upload(file) -> str:
        """Save uploaded file, return path."""
        ext = file.filename.rsplit(".", 1)[1].lower() if "." in file.filename else "png"
        filename = f"{uuid.uuid4().hex[:12]}.{ext}"
        path = str(UPLOAD_DIR / filename)
        file.save(path)
        return path

    def _make_error(message: str, status: int = 400) -> tuple:
        return jsonify({"success": False, "error": message}), status

    def _run_step(name: str, fn, *args, **kwargs):
        """Run a pipeline step with timing and error capture."""
        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
            elapsed = time.time() - t0
            log.info(f"  ✓ {name} ({elapsed:.2f}s)")
            return result, elapsed, None
        except Exception as e:
            elapsed = time.time() - t0
            error_msg = str(e)
            log.warning(f"  ✗ {name} failed ({elapsed:.2f}s): {error_msg}")
            return None, elapsed, error_msg

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        """Serve the frontend UI."""
        return render_template("index.html")

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "services": {
                name: svc.available for name, svc in services.items()
            },
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Full multi-modal prediction pipeline.
        """
        t_start = time.time()
        log.info("─" * 40)
        log.info("POST /predict")

        # ── Parse inputs ──────────────────────────────────────────────────────
        image_file = request.files.get("image")
        clinical_text = (
            request.form.get("text", "")
            or request.form.get("clinical_text", "")
        )
        target_class = request.form.get("target_class")

        if not image_file and not clinical_text:
            return _make_error(
                "At least one input required: 'image' file or 'text' field."
            )

        response = {
            "success": True,
            "pipeline_steps": {},
            "timings": {},
        }

        image_path = None
        image_probs = None
        ner_entities = None
        summary_text = None

        # ── Step 1: Image prediction ──────────────────────────────────────────
        if image_file:
            if not _allowed_file(image_file.filename):
                return _make_error(
                    f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"
                )
            image_path = _save_upload(image_file)

            if services["image"].available:
                image_probs, dt, err = _run_step(
                    "Disease Prediction", services["image"].predict, image_path,
                )
                response["timings"]["disease_prediction"] = round(dt, 3)
                if image_probs:
                    response["disease_predictions"] = image_probs
                else:
                    response["pipeline_steps"]["disease_prediction"] = {
                        "status": "error", "message": err,
                    }
            else:
                response["pipeline_steps"]["disease_prediction"] = {
                    "status": "skipped", "reason": "Image model unavailable",
                }

        # ── Step 2: Grad-CAM heatmap ──────────────────────────────────────────
        if image_path and services["heatmap"].available:
            heatmap_b64, dt, err = _run_step(
                "Grad-CAM Heatmap",
                services["heatmap"].generate,
                image_path,
                target_class,
            )
            response["timings"]["heatmap"] = round(dt, 3)
            if heatmap_b64:
                response["heatmap_base64"] = heatmap_b64
            else:
                response["pipeline_steps"]["heatmap"] = {
                    "status": "error", "message": err,
                }
        elif image_path:
            response["pipeline_steps"]["heatmap"] = {
                "status": "skipped", "reason": "Heatmap service unavailable",
            }

        # ── Step 3: NER extraction ────────────────────────────────────────────
        if clinical_text and services["ner"].available:
            ner_entities, dt, err = _run_step(
                "NER Extraction", services["ner"].extract, clinical_text,
            )
            response["timings"]["ner_extraction"] = round(dt, 3)
            if ner_entities:
                response["entities"] = ner_entities
            else:
                response["pipeline_steps"]["ner_extraction"] = {
                    "status": "error", "message": err,
                }

        # ── Step 4: Summarization ─────────────────────────────────────────────
        if clinical_text and services["summarizer"].available:
            summary_text, dt, err = _run_step(
                "Summarization", services["summarizer"].summarize, clinical_text,
            )
            response["timings"]["summarization"] = round(dt, 3)
            if summary_text:
                response["summary"] = summary_text
            else:
                response["pipeline_steps"]["summarization"] = {
                    "status": "error", "message": err,
                }

        # ── Step 5: Triage fusion ─────────────────────────────────────────────
        if (image_probs or ner_entities) and services["triage"].available:
            # Partial triage with defaults
            _img = image_probs or {}
            _ner = ner_entities or {
                "symptoms": [], "diseases": [],
                "medications": [], "risk_factors": [],
            }
            triage_result, dt, err = _run_step(
                "Triage Scoring",
                services["triage"].predict, _img, _ner,
            )
            response["timings"]["triage"] = round(dt, 3)
            if triage_result:
                response["triage"] = _sanitize(triage_result)
            else:
                response["pipeline_steps"]["triage"] = {
                    "status": "error", "message": err,
                }

        # ── Finalize ──────────────────────────────────────────────────────────
        total_time = time.time() - t_start
        response["total_time_seconds"] = round(total_time, 3)

        # Cleanup uploaded file
        if image_path:
            try:
                os.remove(image_path)
            except OSError:
                pass

        log.info(f"  → Response ready ({total_time:.2f}s)")
        return jsonify(response)

    return app


def _sanitize(obj: Any) -> Any:
    """Make an object JSON-serializable (handle numpy types)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clinical Decision Support API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=5050, help="Bind port")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)
