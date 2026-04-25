"""
================================================================================
  STEP 3 — GRAD-CAM INTERPRETABILITY MODULE
  Chest X-Ray Multi-Label Classifier
  Gradient-weighted Class Activation Mapping (Grad-CAM)

  Compatible with: train_convnextv2.py (Hugging Face)
  Backbone target: ConvNeXtV2 → stages[-1] (last stage)
================================================================================

IMPLEMENTS:
  1. GradCAM class           — hooks into any target layer, computes heatmaps
  2. generate_heatmap()      — single image → overlay with Grad-CAM heatmap
  3. generate_heatmaps_batch() — batch of images → list of overlay images
  4. visualize_samples()     — multi-panel figure for N samples
  5. CLI entry point         — python gradcam_xray.py --image <path> [options]

CONSTRAINTS:
  - Pure PyTorch — no external Grad-CAM libraries (captum, pytorch-grad-cam, etc.)
  - Uses forward/backward hooks on target conv layer

QUICK START (Colab):
  from gradcam_xray import generate_heatmap, visualize_samples
  overlay = generate_heatmap("xray.jpg", model)
  overlay.save("gradcam_output.png")

  # Batch
  from gradcam_xray import generate_heatmaps_batch
  overlays = generate_heatmaps_batch(["img1.jpg", "img2.jpg"], model)

  # Multi-sample visualization
  visualize_samples(["img1.jpg", "img2.jpg"], model, save_path="panel.png")
================================================================================
"""

# ==============================================================================
# SECTION 0 — IMPORTS
# ==============================================================================

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1 — GRAD-CAM CORE ENGINE
# ==============================================================================

class GradCAM:
    """
    Manual Grad-CAM implementation using PyTorch forward/backward hooks.

    How it works:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  1. Register a forward hook on target_layer to capture activations │
    │  2. Register a backward hook on target_layer to capture gradients  │
    │  3. Forward pass the image through the full model                  │
    │  4. Backward pass from the target class logit                      │
    │  5. Pool gradients (GAP) → per-channel importance weights          │
    │  6. Weighted combination of activation maps → raw heatmap          │
    │  7. ReLU (keep only positive influence) + normalize to [0, 1]      │
    │  8. Resize heatmap to input image dimensions                       │
    └─────────────────────────────────────────────────────────────────────┘

    Args:
        model       : Trained ConvNeXtV2 (or any nn.Module)
        target_layer: The conv layer to visualize (e.g., model.convnextv2.stages[-1])
                      Defaults to the last stage for ConvNeXtV2

    Usage:
        cam = GradCAM(model, model.backbone.layer4)
        heatmap = cam(input_tensor, class_idx=6)  # class 6 = Pneumonia
        cam.remove_hooks()  # clean up when done
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ):
        self.model = model
        self.target_layer = target_layer

        # Auto-detect target layer if not provided
        if self.target_layer is None:
            if hasattr(model, "convnextv2"): # ConvNeXt V2 (Hugging Face)
                if hasattr(model.convnextv2, "encoder"):
                    self.target_layer = model.convnextv2.encoder.stages[-1]
                else:
                    self.target_layer = model.convnextv2.stages[-1]
                log.info("GradCAM: Auto-detected ConvNeXt V2 target layer")
            elif hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
                self.target_layer = model.backbone.layer4
                log.info("GradCAM: Auto-detected ResNet50 (legacy) target layer (layer4)")
            elif hasattr(model, "features"):
                self.target_layer = model.features[-1]
                log.info("GradCAM: Auto-detected MobileNetV2 target layer (features[-1])")
            else:
                # Fallback: find the last conv layer
                for module in reversed(list(model.modules())):
                    if isinstance(module, nn.Conv2d):
                        self.target_layer = module
                        log.info(f"GradCAM: Using last detected Conv2d layer: {module}")
                        break
        
        if self.target_layer is None:
            raise ValueError("Could not auto-detect target layer. Please provide one explicitly.")

        # Storage for hooked tensors
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = self.target_layer.register_forward_hook(self._forward_hook)
        self._bwd_hook = self.target_layer.register_full_backward_hook(self._backward_hook)

        log.debug(f"GradCAM hooks registered on: {self.target_layer.__class__.__name__}")

    # ── Hook callbacks ────────────────────────────────────────────────────────

    def _forward_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        """Captures the feature map activations from the target layer."""
        self._activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor],
        grad_output: Tuple[torch.Tensor],
    ) -> None:
        """Captures the gradients flowing back through the target layer."""
        self._gradients = grad_output[0].detach()

    # ── Main computation ──────────────────────────────────────────────────────

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for a given input.

        Args:
            input_tensor: Preprocessed image tensor [1, 3, H, W] or [B, 3, H, W]
            class_idx   : Target class index. If None, uses the class with
                          the highest predicted score (argmax of logits).

        Returns:
            heatmap: numpy array of shape [H_input, W_input] with values in [0, 1]
                     where H_input, W_input are the spatial dims of input_tensor.
        """
        # Ensure model is in eval mode but with gradients enabled
        self.model.eval()

        # Clear any old stored tensors
        self._activations = None
        self._gradients = None

        # ── Step 1: Forward pass ──────────────────────────────────────────────
        # We need gradients for the activations, so we enable grad
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        # Handle Hugging Face output objects (ImageClassifierOutput)
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output  # Standard tensor

        # ── Step 2: Select target class ───────────────────────────────────────
        if class_idx is None:
            # Use predicted class (highest logit)
            class_idx = logits.argmax(dim=1).item()

        # ── Step 3: Backward pass from target class logit ─────────────────────
        # Zero all existing gradients
        self.model.zero_grad()

        # Create a one-hot target for the selected class
        target_logit = logits[0, class_idx]
        target_logit.backward(retain_graph=True)

        # ── Step 4: Compute Grad-CAM heatmap ──────────────────────────────────
        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Hooks failed to capture activations/gradients. "
                "Ensure target_layer is part of the forward computation graph."
            )

        # Gradients: [1, C, h, w] → global average pool → [1, C, 1, 1]
        # These are the "importance weights" α_k for each channel k
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination: Σ_k (α_k × A_k)
        # weights: [1, C, 1, 1] × activations: [1, C, h, w] → sum over C → [1, 1, h, w]
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]

        # ── Step 5: ReLU — only keep features that positively contribute ──────
        cam = F.relu(cam)

        # ── Step 6: Normalize to [0, 1] ───────────────────────────────────────
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # ── Step 7: Resize to input spatial dimensions with Smoothing ─────────
        _, _, H_in, W_in = input_tensor.shape
        cam = F.interpolate(
            cam,
            size=(H_in, W_in),
            mode="bilinear",
            align_corners=False,
        )

        # Convert to numpy for final smoothing
        cam_np = cam.squeeze().detach().cpu().numpy()
        
        # Optional: Add a light Gaussian blur to make it "lovely" (Smooth)
        from scipy.ndimage import gaussian_filter
        cam_np = gaussian_filter(cam_np, sigma=H_in/50) # Scale sigma with resolution
        
        # Final Min-Max normalization
        if cam_np.max() > 0:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            
        return cam_np

        # Convert to numpy [H, W]
        heatmap = cam.squeeze().cpu().numpy()

        return heatmap

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def remove_hooks(self) -> None:
        """Remove registered hooks to prevent memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()
        self._activations = None
        self._gradients = None
        log.debug("GradCAM hooks removed.")

    def __del__(self):
        """Safety net: remove hooks on garbage collection."""
        try:
            self.remove_hooks()
        except Exception:
            pass


# ==============================================================================
# SECTION 2 — IMAGE PREPROCESSING
# ==============================================================================

def get_inference_transform(image_size: int = 224) -> T.Compose:
    """
    Returns the standard inference transform matching the training pipeline.
    Must be identical to the val/test transform used in the training script.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_and_preprocess(
    image_path: str,
    transform: Optional[T.Compose] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Load an image from disk, apply preprocessing, return both the
    tensor (for the model) and the original PIL image (for overlay).

    Args:
        image_path : Path to the X-ray image (.jpg, .png, .dcm)
        transform  : Torchvision transform pipeline. If None, uses default.
        device     : Target device for the tensor.

    Returns:
        (input_tensor, original_pil_image)
        - input_tensor: [1, 3, 224, 224] float tensor on device
        - original_pil_image: RGB PIL Image at original resolution
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and convert to RGB (X-rays may be grayscale)
    original = Image.open(image_path).convert("RGB")

    if transform is None:
        transform = get_inference_transform()

    tensor = transform(original).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    return tensor, original


# ==============================================================================
# SECTION 3 — HEATMAP OVERLAY GENERATION
# ==============================================================================

def apply_colormap(
    heatmap: np.ndarray,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Apply a matplotlib colormap to a grayscale heatmap.

    Args:
        heatmap  : 2D numpy array with values in [0, 1]
        colormap : Matplotlib colormap name (default: 'jet')

    Returns:
        colored_heatmap: [H, W, 3] uint8 numpy array (RGB)
    """
    cmap = plt.get_cmap(colormap)
    colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel
    colored = (colored * 255).astype(np.uint8)
    return colored


def overlay_heatmap_on_image(
    original: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """
    Overlay a Grad-CAM heatmap on the original image.

    The heatmap is resized to match the original image dimensions,
    colorized with the specified colormap, and alpha-blended.

    Args:
        original : Original PIL Image (any size)
        heatmap  : 2D numpy array [H, W] with values in [0, 1]
        alpha    : Blending factor (0 = original only, 1 = heatmap only)
        colormap : Matplotlib colormap name

    Returns:
        overlay: PIL Image with heatmap overlaid
    """
    # Resize heatmap to match original image
    W, H = original.size
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (W, H), resample=Image.BILINEAR
        )
    ) / 255.0

    # Apply colormap
    heatmap_colored = apply_colormap(heatmap_resized, colormap)

    # Convert original to numpy
    original_np = np.array(original)

    # Alpha blend: overlay = α × heatmap + (1 - α) × original
    blended = (
        alpha * heatmap_colored.astype(np.float32)
        + (1 - alpha) * original_np.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    return Image.fromarray(blended)


# ==============================================================================
# SECTION 4 — MAIN API: generate_heatmap (SINGLE IMAGE)
# ==============================================================================

# Default disease class names — can be overridden at runtime
DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

def generate_heatmap(
    image_path: str,
    model: nn.Module,
    target_class: Optional[Union[int, str]] = None,
    target_layer: Optional[nn.Module] = None,
    alpha: float = 0.5,
    colormap: str = "jet",
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> Image.Image:
    """
    Generate a Grad-CAM heatmap overlay.
    
    Args:
        ...
        class_names: Optional list of class names to use for lookup. 
                     If None, defaults to the global DISEASE_CLASSES.
    """
    classes = class_names or DISEASE_CLASSES
    """
    Generate a Grad-CAM heatmap overlay for a single chest X-ray image.

    This is the primary API function. It:
      1. Loads and preprocesses the image
      2. Runs Grad-CAM on the specified (or auto-detected) target layer
      3. Overlays the heatmap on the original image
      4. Optionally saves the result to disk

    Args:
        image_path   : Path to the input X-ray image.
        model        : Trained ConvNeXtV2 model.
        target_class : Which disease class to visualize.
                       - int: class index (0–13)
                       - str: disease name (e.g., "Pneumonia")
                       - None: auto-select the class with highest prediction
        target_layer : Conv layer to hook into. Defaults to model.backbone.layer4
                       (the last residual block, producing 7×7 feature maps).
        alpha        : Heatmap transparency (0.0–1.0). Default 0.5.
        colormap     : Matplotlib colormap for heatmap. Default "jet".
        device       : Compute device. Auto-detected if None.
        save_path    : If provided, saves the overlay image to this path.

    Returns:
        overlay_image: PIL Image with Grad-CAM heatmap overlaid on the original.

    Example:
        >>> from gradcam_xray import generate_heatmap
        >>> overlay = generate_heatmap("chest_xray.jpg", model)
        >>> overlay.save("gradcam_result.png")

        >>> # Target a specific disease
        >>> overlay = generate_heatmap("xray.jpg", model, target_class="Pneumonia")
    """
    # ── Resolve device ────────────────────────────────────────────────────────
    if device is None:
        device = next(model.parameters()).device

    # ── Resolve target layer ──────────────────────────────────────────────────
    if target_layer is None:
        # Detect backbone type and pick appropriate target layer
        if hasattr(model, "convnextv2"): # ConvNeXt V2 (Hugging Face)
            if hasattr(model.convnextv2, "encoder"):
                target_layer = model.convnextv2.encoder.stages[-1]
            else:
                target_layer = model.convnextv2.stages[-1]
        elif hasattr(model, "backbone"):
            bb = model.backbone
            if hasattr(bb, "layer4"):  # ResNet50
                target_layer = bb.layer4
            elif hasattr(bb, "features"):  # MobileNetV2
                target_layer = bb.features[-1]
            else:
                # Fallback: try to find the last Conv2d layer
                conv_layers = [m for m in bb.modules() if isinstance(m, nn.Conv2d)]
                if conv_layers:
                    target_layer = conv_layers[-1]
        
        if target_layer is None:
            raise ValueError(
                "Cannot auto-detect target layer for Grad-CAM. "
                "Ensure model.backbone is available and contains features or layer4."
            )

    # ── Resolve class index ───────────────────────────────────────────────────
    class_idx = None
    if isinstance(target_class, str):
        # Look up by disease name
        try:
            class_idx = classes.index(target_class)
        except ValueError:
            raise ValueError(
                f"Unknown disease class: '{target_class}'. "
                f"Valid classes: {classes}"
            )
    elif isinstance(target_class, int):
        if not (0 <= target_class < len(classes)):
            raise ValueError(
                f"Class index {target_class} out of range [0, {len(classes)-1}]."
            )
        class_idx = target_class
    # else: None → GradCAM will auto-select the top predicted class

    # ── Load and preprocess ───────────────────────────────────────────────────
    input_tensor, original_image = load_and_preprocess(
        image_path, device=device
    )

    # ── Compute Grad-CAM ──────────────────────────────────────────────────────
    cam = GradCAM(model, target_layer)
    try:
        heatmap = cam(input_tensor, class_idx=class_idx)
    finally:
        cam.remove_hooks()  # Always clean up hooks

    # ── Generate overlay ──────────────────────────────────────────────────────
    overlay = overlay_heatmap_on_image(
        original_image, heatmap, alpha=alpha, colormap=colormap
    )

    # ── Optionally save ───────────────────────────────────────────────────────
    if save_path is not None:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        overlay.save(save_path)
        log.info(f"Grad-CAM overlay saved → {save_path}")

    # ── Log prediction info ───────────────────────────────────────────────────
    class_name = classes[class_idx] if class_idx is not None else "auto (top-1)"
    log.info(
        f"Grad-CAM generated for '{Path(image_path).name}' | "
        f"target class: {class_name}"
    )

    return overlay


# ==============================================================================
# SECTION 5 — BATCH API: generate_heatmaps_batch
# ==============================================================================

def generate_heatmaps_batch(
    image_paths: List[str],
    model: nn.Module,
    target_class: Optional[Union[int, str]] = None,
    target_layer: Optional[nn.Module] = None,
    alpha: float = 0.5,
    colormap: str = "jet",
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
) -> List[Image.Image]:
    """
    Generate Grad-CAM heatmap overlays for a batch of images.

    Processes each image sequentially (Grad-CAM requires per-image backward
    passes, so true batching doesn't provide speedup here).

    Args:
        image_paths  : List of paths to X-ray images.
        model        : Trained ConvNeXtV2 model.
        target_class : Disease class to visualize (shared across batch).
                       Same format as generate_heatmap().
        target_layer : Conv layer to hook. Defaults to model.convnextv2.stages[-1].
        alpha        : Heatmap transparency.
        colormap     : Colormap for heatmap.
        device       : Compute device.
        save_dir     : If provided, saves each overlay as
                       <save_dir>/gradcam_<original_filename>.png

    Returns:
        List of PIL Images with Grad-CAM overlays.

    Example:
        >>> paths = ["xray1.jpg", "xray2.jpg", "xray3.jpg"]
        >>> overlays = generate_heatmaps_batch(paths, model, target_class="Effusion")
        >>> for i, ov in enumerate(overlays):
        ...     ov.save(f"overlay_{i}.png")
    """
    if not image_paths:
        log.warning("Empty image list provided to generate_heatmaps_batch.")
        return []

    overlays = []
    n_images = len(image_paths)

    log.info(f"Generating Grad-CAM heatmaps for {n_images} images…")

    for idx, img_path in enumerate(image_paths):
        # ── Determine save path for this image ────────────────────────────────
        if save_dir is not None:
            fname = Path(img_path).stem
            save_path = str(Path(save_dir) / f"gradcam_{fname}.png")
        else:
            save_path = None

        try:
            overlay = generate_heatmap(
                image_path=img_path,
                model=model,
                target_class=target_class,
                target_layer=target_layer,
                alpha=alpha,
                colormap=colormap,
                device=device,
                save_path=save_path,
            )
            overlays.append(overlay)
            log.info(f"  [{idx + 1}/{n_images}] ✓ {Path(img_path).name}")

        except Exception as e:
            log.error(f"  [{idx + 1}/{n_images}] ✗ {Path(img_path).name}: {e}")
            overlays.append(None)  # Placeholder for failed images

    n_success = sum(1 for o in overlays if o is not None)
    log.info(f"Batch complete: {n_success}/{n_images} successful.")

    return overlays


# ==============================================================================
# SECTION 6 — MULTI-CLASS HEATMAP (ALL DISEASES FOR ONE IMAGE)
# ==============================================================================

def generate_all_class_heatmaps(
    image_path: str,
    model: nn.Module,
    target_layer: Optional[nn.Module] = None,
    top_k: int = 5,
    alpha: float = 0.5,
    colormap: str = "jet",
    device: Optional[torch.device] = None,
) -> Dict[str, Tuple[Image.Image, float]]:
    """
    Generate Grad-CAM heatmaps for the top-K predicted disease classes.

    Useful for understanding what the model focuses on for each diagnosis.

    Args:
        image_path   : Path to X-ray image.
        model        : Trained model.
        target_layer : Conv layer (default: model.backbone.layer4).
        top_k        : Number of top predicted classes to visualize.
        alpha        : Heatmap transparency.
        colormap     : Colormap name.
        device       : Compute device.

    Returns:
        Dict mapping disease_name → (overlay_image, probability)
        Ordered by descending probability.

    Example:
        >>> results = generate_all_class_heatmaps("xray.jpg", model, top_k=3)
        >>> for disease, (overlay, prob) in results.items():
        ...     print(f"{disease}: {prob:.2%}")
        ...     overlay.save(f"gradcam_{disease}.png")
    """
    if device is None:
        device = next(model.parameters()).device

    # ── Get model predictions ─────────────────────────────────────────────────
    input_tensor, _ = load_and_preprocess(image_path, device=device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        logits = output.logits if hasattr(output, "logits") else output
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    # ── Select top-K classes ──────────────────────────────────────────────────
    top_indices = np.argsort(probs)[::-1][:top_k]

    # ── Generate heatmap for each ─────────────────────────────────────────────
    results = {}
    for class_idx in top_indices:
        class_name = DISEASE_CLASSES[class_idx]
        probability = float(probs[class_idx])

        overlay = generate_heatmap(
            image_path=image_path,
            model=model,
            target_class=int(class_idx),
            target_layer=target_layer,
            alpha=alpha,
            colormap=colormap,
            device=device,
        )
        results[class_name] = (overlay, probability)

    return results


# ==============================================================================
# SECTION 7 — VISUALIZATION: MULTI-SAMPLE PANEL
# ==============================================================================

def visualize_samples(
    image_paths: List[str],
    model: nn.Module,
    target_class: Optional[Union[int, str]] = None,
    target_layer: Optional[nn.Module] = None,
    alpha: float = 0.5,
    colormap: str = "jet",
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    figsize_per_image: Tuple[float, float] = (5.0, 4.0),
    show_predictions: bool = True,
    max_cols: int = 4,
) -> plt.Figure:
    """
    Create a multi-panel visualization showing original images alongside
    their Grad-CAM overlays for multiple samples.

    Layout per image:
    ┌──────────────┬──────────────┐
    │   Original   │  Grad-CAM    │
    │   X-Ray      │  Overlay     │
    │              │              │
    │ ─── predictions below ───── │
    └──────────────┴──────────────┘

    Args:
        image_paths      : List of image file paths.
        model            : Trained ConvNeXtV2 model.
        target_class     : Disease class to highlight (None = auto top-1).
        target_layer     : Target conv layer for Grad-CAM hooks.
        alpha            : Heatmap overlay transparency.
        colormap         : Heatmap colormap.
        device           : Compute device.
        save_path        : Path to save the figure. If None, figure is returned.
        figsize_per_image: (width, height) in inches per image pair.
        show_predictions : If True, annotate each panel with prediction info.
        max_cols         : Maximum number of image pairs per row.

    Returns:
        matplotlib Figure object.

    Example:
        >>> paths = ["xray1.jpg", "xray2.jpg", "xray3.jpg", "xray4.jpg"]
        >>> fig = visualize_samples(paths, model, save_path="gradcam_panel.png")
    """
    if device is None:
        device = next(model.parameters()).device

    n_images = len(image_paths)
    n_cols = min(n_images, max_cols)
    n_rows = (n_images + n_cols - 1) // n_cols

    # Each image gets 2 sub-columns (original + overlay)
    fig_w = figsize_per_image[0] * 2 * n_cols
    fig_h = figsize_per_image[1] * n_rows

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    fig.suptitle(
        "Grad-CAM Interpretability — Chest X-Ray Model",
        fontsize=16,
        fontweight="bold",
        color="#1a1a2e",
        y=1.02,
    )

    # Build a grid: n_rows × (2 * n_cols) to pair original + overlay
    gs = gridspec.GridSpec(
        n_rows, 2 * n_cols,
        figure=fig,
        wspace=0.05,
        hspace=0.35,
    )

    for idx, img_path in enumerate(image_paths):
        row = idx // n_cols
        col = idx % n_cols

        try:
            # ── Load image and get predictions ────────────────────────────────
            input_tensor, original = load_and_preprocess(img_path, device=device)

            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                logits = output.logits if hasattr(output, "logits") else output
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            # ── Determine target class ────────────────────────────────────────
            if target_class is not None:
                if isinstance(target_class, str):
                    cls_idx = DISEASE_CLASSES.index(target_class)
                else:
                    cls_idx = target_class
            else:
                cls_idx = int(np.argmax(probs))

            cls_name = DISEASE_CLASSES[cls_idx]
            cls_prob = probs[cls_idx]

            # ── Generate overlay ──────────────────────────────────────────────
            overlay = generate_heatmap(
                image_path=img_path,
                model=model,
                target_class=cls_idx,
                target_layer=target_layer,
                alpha=alpha,
                colormap=colormap,
                device=device,
            )

            # ── Plot original ─────────────────────────────────────────────────
            ax_orig = fig.add_subplot(gs[row, 2 * col])
            ax_orig.imshow(original)
            ax_orig.set_title(
                f"Original\n{Path(img_path).name}",
                fontsize=9,
                fontweight="bold",
                color="#333",
            )
            ax_orig.axis("off")

            # ── Plot Grad-CAM overlay ─────────────────────────────────────────
            ax_cam = fig.add_subplot(gs[row, 2 * col + 1])
            ax_cam.imshow(overlay)

            title_text = f"Grad-CAM: {cls_name}\n"
            if show_predictions:
                # Show top 3 predictions
                top3_idx = np.argsort(probs)[::-1][:3]
                pred_lines = [
                    f"{DISEASE_CLASSES[i]}: {probs[i]:.1%}"
                    for i in top3_idx
                ]
                title_text += " | ".join(pred_lines)

            ax_cam.set_title(
                title_text,
                fontsize=8,
                fontweight="bold",
                color="#c0392b" if cls_prob >= 0.5 else "#27ae60",
            )
            ax_cam.axis("off")

            # ── Confidence bar below ──────────────────────────────────────────
            if show_predictions:
                # Add a thin colored bar indicating confidence
                bar_color = plt.cm.RdYlGn_r(cls_prob)  # Red=high, green=low
                ax_cam.add_patch(plt.Rectangle(
                    (0, 0), 1, 0.02,
                    transform=ax_cam.transAxes,
                    color=bar_color,
                    clip_on=False,
                    zorder=10,
                ))

        except Exception as e:
            # ── Handle errors gracefully ──────────────────────────────────────
            ax = fig.add_subplot(gs[row, 2 * col: 2 * col + 2])
            ax.text(
                0.5, 0.5,
                f"Error processing:\n{Path(img_path).name}\n{str(e)[:60]}",
                ha="center", va="center",
                fontsize=10, color="red",
                transform=ax.transAxes,
            )
            ax.axis("off")
            log.error(f"Failed to process {img_path}: {e}")

    plt.tight_layout()

    # ── Save figure ───────────────────────────────────────────────────────────
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        log.info(f"Visualization panel saved → {save_path}")
        plt.close(fig)

    return fig


# ==============================================================================
# SECTION 8 — VISUALIZATION: SINGLE IMAGE DETAILED VIEW
# ==============================================================================

def visualize_single_detailed(
    image_path: str,
    model: nn.Module,
    target_layer: Optional[nn.Module] = None,
    top_k: int = 4,
    alpha: float = 0.5,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Detailed Grad-CAM visualization for a single image showing the
    original image alongside heatmaps for the top-K predicted classes.

    Layout:
    ┌────────────┬────────────┬────────────┬────────────┬────────────┐
    │  Original  │ Top-1 CAM  │ Top-2 CAM  │ Top-3 CAM  │ Top-4 CAM  │
    │  + preds   │ Pneumonia  │ Effusion   │ Atel…      │ Mass       │
    │            │  87.2%     │  61.4%     │  45.1%     │  23.0%     │
    └────────────┴────────────┴────────────┴────────────┴────────────┘

    Args:
        image_path   : Path to X-ray image.
        model        : Trained model.
        target_layer : Target conv layer.
        top_k        : Number of top disease classes to show.
        alpha        : Heatmap transparency.
        device       : Compute device.
        save_path    : Output path for the figure.

    Returns:
        matplotlib Figure.
    """
    if device is None:
        device = next(model.parameters()).device

    # ── Get predictions ───────────────────────────────────────────────────────
    input_tensor, original = load_and_preprocess(image_path, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]

    # ── Create figure ─────────────────────────────────────────────────────────
    n_panels = 1 + top_k  # Original + K heatmaps
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))
    fig.suptitle(
        f"Grad-CAM Analysis — {Path(image_path).name}",
        fontsize=14,
        fontweight="bold",
        color="#1a1a2e",
    )

    # ── Original image panel ──────────────────────────────────────────────────
    axes[0].imshow(original)
    axes[0].set_title("Original X-Ray", fontsize=11, fontweight="bold")

    # Add prediction text below
    pred_text = "Predictions:\n"
    for i in range(min(5, len(DISEASE_CLASSES))):
        ci = np.argsort(probs)[::-1][i]
        marker = "●" if probs[ci] >= 0.5 else "○"
        pred_text += f" {marker} {DISEASE_CLASSES[ci]}: {probs[ci]:.1%}\n"
    axes[0].text(
        0.02, -0.02, pred_text,
        transform=axes[0].transAxes,
        fontsize=7, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8),
    )
    axes[0].axis("off")

    # ── Grad-CAM panels ──────────────────────────────────────────────────────
    for panel_idx, class_idx in enumerate(top_indices, start=1):
        class_name = DISEASE_CLASSES[class_idx]
        probability = probs[class_idx]

        overlay = generate_heatmap(
            image_path=image_path,
            model=model,
            target_class=int(class_idx),
            target_layer=target_layer,
            alpha=alpha,
            device=device,
        )

        axes[panel_idx].imshow(overlay)

        # Color the title by confidence level
        if probability >= 0.7:
            title_color = "#c0392b"  # Red — high confidence
        elif probability >= 0.4:
            title_color = "#e67e22"  # Orange — moderate
        else:
            title_color = "#27ae60"  # Green — low

        axes[panel_idx].set_title(
            f"{class_name}\n{probability:.1%}",
            fontsize=11,
            fontweight="bold",
            color=title_color,
        )
        axes[panel_idx].axis("off")

        # Add thin confidence bar
        bar = plt.Rectangle(
            (0, -0.01), probability, 0.015,
            transform=axes[panel_idx].transAxes,
            color=title_color, alpha=0.8,
            clip_on=False, zorder=10,
        )
        axes[panel_idx].add_patch(bar)

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        log.info(f"Detailed visualization saved → {save_path}")
        plt.close(fig)

    return fig


# ==============================================================================
# SECTION 9 — UTILITY: RAW HEATMAP EXTRACTION
# ==============================================================================

def get_raw_heatmap(
    image_path: str,
    model: nn.Module,
    target_class: Optional[Union[int, str]] = None,
    target_layer: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Returns the raw Grad-CAM heatmap along with predictions.
    Useful for custom post-processing or integration with other tools.

    Args:
        image_path   : Path to X-ray image.
        model        : Trained model.
        target_class : Target disease class (int, str, or None).
        target_layer : Target conv layer.
        device       : Compute device.

    Returns:
        (heatmap, original_np, predictions)
        - heatmap     : [224, 224] float array in [0, 1]
        - original_np : [H, W, 3] uint8 array (original resolution)
        - predictions : dict {disease_name: probability}
    """
    if device is None:
        device = next(model.parameters()).device

    if target_layer is None:
        target_layer = model.backbone.layer4

    # Resolve class index
    class_idx = None
    if isinstance(target_class, str):
        class_idx = DISEASE_CLASSES.index(target_class)
    elif isinstance(target_class, int):
        class_idx = target_class

    # Load and preprocess
    input_tensor, original = load_and_preprocess(image_path, device=device)
    original_np = np.array(original)

    # Get predictions
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    predictions = {
        cls: float(prob)
        for cls, prob in zip(DISEASE_CLASSES, probs)
    }
    predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))

    # Compute heatmap
    cam = GradCAM(model, target_layer)
    try:
        heatmap = cam(input_tensor, class_idx=class_idx)
    finally:
        cam.remove_hooks()

    return heatmap, original_np, predictions


# ==============================================================================
# SECTION 10 — SMOKE TEST
# ==============================================================================

def _run_smoke_test() -> None:
    """
    Validates the Grad-CAM pipeline end-to-end with synthetic data.
    No trained model or real images needed — creates everything from scratch.
    """
    import tempfile

    log.info("=" * 60)
    log.info("  GRAD-CAM SMOKE TEST")
    log.info("=" * 60)

    # ── Create a dummy ConvNeXtV2 model for testing ───────────────────────────
    log.info("  Building dummy ConvNeXtV2 for smoke test…")
    import torchvision.models as tv_models
    
    # We use a standard torchvision convnext_tiny as a proxy for the HF structure
    base_model = tv_models.convnext_tiny(weights=None)
    
    # Wrap it to mimic the Hugging Face structure used in the main app
    class HFConvNeXtWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.convnextv2 = base # Mimic the .convnextv2 attribute
            self.logits = nn.Linear(768, 14)
            # Ensure it has the correct stage structure for the hook
            self.convnextv2.encoder = base.features
            
        def forward(self, x):
            # Simplified forward pass for testing
            features = self.convnextv2.features(x)
            pooled = self.convnextv2.avgpool(features)
            logits = self.logits(pooled.flatten(1))
            
            # Mimic HF output object
            from collections import namedtuple
            Output = namedtuple("Output", ["logits"])
            return Output(logits=logits)

    model = HFConvNeXtWrapper(base_model)
    model.eval()

    device = torch.device("cpu")
    model = model.to(device).eval()

    # ── Create a synthetic test image ─────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(arr).save(f.name)
        tmp_img = f.name

    # ── Test 1: GradCAM core ──────────────────────────────────────────────────
    log.info("  Test 1: GradCAM core computation…")
    cam = GradCAM(model)
    input_tensor, _ = load_and_preprocess(tmp_img, device=device)
    heatmap = cam(input_tensor, class_idx=0)
    assert heatmap.shape == (224, 224), f"Bad shape: {heatmap.shape}"
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0 + 1e-6
    cam.remove_hooks()
    log.info(f"  ✓ Heatmap shape: {heatmap.shape} | range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    # ── Test 2: generate_heatmap (single image) ──────────────────────────────
    log.info("  Test 2: generate_heatmap (single)…")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        save_path = f.name
    overlay = generate_heatmap(tmp_img, model, target_class=0, device=device, save_path=save_path)
    assert isinstance(overlay, Image.Image)
    assert Path(save_path).exists()
    log.info(f"  ✓ Overlay size: {overlay.size} | saved to {save_path}")

    # ── Test 3: generate_heatmaps_batch ───────────────────────────────────────
    log.info("  Test 3: generate_heatmaps_batch…")
    overlays = generate_heatmaps_batch(
        [tmp_img, tmp_img], model, target_class=0, device=device
    )
    assert len(overlays) == 2
    assert all(isinstance(o, Image.Image) for o in overlays)
    log.info(f"  ✓ Batch returned {len(overlays)} overlays")

    # ── Test 4: get_raw_heatmap ───────────────────────────────────────────────
    log.info("  Test 4: get_raw_heatmap…")
    hmap, orig_np, preds = get_raw_heatmap(tmp_img, model, target_class=0, device=device)
    assert len(preds) == 14
    log.info(f"  ✓ Raw heatmap shape: {hmap.shape} | {len(preds)} predictions")

    # ── Test 5: visualize_samples ─────────────────────────────────────────────
    log.info("  Test 5: visualize_samples…")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        panel_path = f.name
    fig = visualize_samples(
        [tmp_img, tmp_img], model, target_class=0,
        device=device, save_path=panel_path,
    )
    assert Path(panel_path).exists()
    log.info(f"  ✓ Panel saved to {panel_path}")

    # ── Test 6: visualize_single_detailed ─────────────────────────────────────
    log.info("  Test 6: visualize_single_detailed…")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        detail_path = f.name
    fig = visualize_single_detailed(
        tmp_img, model, top_k=3, device=device, save_path=detail_path,
    )
    assert Path(detail_path).exists()
    log.info(f"  ✓ Detailed view saved to {detail_path}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    os.unlink(tmp_img)
    log.info("  ALL GRAD-CAM SMOKE TESTS PASSED ✓")


# ==============================================================================
# SECTION 11 — CLI ENTRY POINT
# ==============================================================================

def main():
    """
    Command-line interface for Grad-CAM visualization.

    Usage:
        # Single image
        python gradcam_xray.py --image xray.jpg --checkpoint best_model.pth

        # Batch of images
        python gradcam_xray.py --images xray1.jpg xray2.jpg --save-dir ./output

        # Target a specific disease
        python gradcam_xray.py --image xray.jpg --checkpoint best_model.pth \
               --target-class Pneumonia --alpha 0.6

        # Detailed single-image view with top-5 diseases
        python gradcam_xray.py --image xray.jpg --checkpoint best_model.pth \
               --detailed --top-k 5

        # Smoke test (no model/images needed)
        python gradcam_xray.py --smoke-test
    """
    parser = argparse.ArgumentParser(
        description="Grad-CAM for Chest X-Ray ConvNeXtV2 Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=main.__doc__,
    )

    # ── Input options ─────────────────────────────────────────────────────────
    group_in = parser.add_mutually_exclusive_group()
    group_in.add_argument(
        "--image", type=str,
        help="Path to a single X-ray image.",
    )
    group_in.add_argument(
        "--images", type=str, nargs="+",
        help="Paths to multiple X-ray images (batch mode).",
    )

    # ── Model options ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint", type=str,
        default=None,
        help="Path to model checkpoint (.pth). Required unless --smoke-test.",
    )

    # ── Grad-CAM options ──────────────────────────────────────────────────────
    parser.add_argument(
        "--target-class", type=str, default=None,
        help="Disease class to visualize (name or index). Default: auto (top-1).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Heatmap overlay transparency (0.0–1.0). Default: 0.5.",
    )
    parser.add_argument(
        "--colormap", type=str, default="jet",
        help="Colormap for heatmap. Default: 'jet'.",
    )

    # ── Output options ────────────────────────────────────────────────────────
    parser.add_argument(
        "--save-path", type=str, default=None,
        help="Output path for single image overlay or panel figure.",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Output directory for batch mode.",
    )

    # ── Visualization options ─────────────────────────────────────────────────
    parser.add_argument(
        "--detailed", action="store_true",
        help="Show detailed single-image view with multiple disease heatmaps.",
    )
    parser.add_argument(
        "--top-k", type=int, default=4,
        help="Number of top classes in detailed view. Default: 4.",
    )

    # ── Test ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run smoke test with synthetic data (no model/images needed).",
    )

    args = parser.parse_args()

    # ── Smoke test ────────────────────────────────────────────────────────────
    if args.smoke_test:
        _run_smoke_test()
        return

    # ── Validate inputs ───────────────────────────────────────────────────────
    if args.image is None and args.images is None:
        parser.error("Provide --image or --images (or --smoke-test).")

    if args.checkpoint is None:
        parser.error("--checkpoint is required for inference.")

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Legacy model loading disabled to support project cleanup
    # from chest_xray_model import load_model
    # model = load_model(args.checkpoint, device=device)
    log.warning("CLI model loading currently disabled. Use the web interface (app.py) for Grad-CAM.")
    sys.exit(0)

    # ── Resolve target class ──────────────────────────────────────────────────
    target_class = args.target_class
    if target_class is not None:
        try:
            target_class = int(target_class)
        except ValueError:
            pass  # Keep as string — generate_heatmap handles name lookup

    # ── Single image mode ─────────────────────────────────────────────────────
    if args.image:
        if args.detailed:
            save_path = args.save_path or f"gradcam_detailed_{Path(args.image).stem}.png"
            visualize_single_detailed(
                args.image, model,
                top_k=args.top_k,
                alpha=args.alpha,
                device=device,
                save_path=save_path,
            )
        else:
            save_path = args.save_path or f"gradcam_{Path(args.image).stem}.png"
            overlay = generate_heatmap(
                args.image, model,
                target_class=target_class,
                alpha=args.alpha,
                colormap=args.colormap,
                device=device,
                save_path=save_path,
            )
        log.info("Done.")

    # ── Batch mode ────────────────────────────────────────────────────────────
    elif args.images:
        save_dir = args.save_dir or "./gradcam_outputs"

        # Generate individual overlays
        overlays = generate_heatmaps_batch(
            args.images, model,
            target_class=target_class,
            alpha=args.alpha,
            colormap=args.colormap,
            device=device,
            save_dir=save_dir,
        )

        # Also generate a panel view
        panel_path = str(Path(save_dir) / "gradcam_panel.png")
        visualize_samples(
            args.images, model,
            target_class=target_class,
            alpha=args.alpha,
            colormap=args.colormap,
            device=device,
            save_path=panel_path,
        )
        log.info("Done.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
