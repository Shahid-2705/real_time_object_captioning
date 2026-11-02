"""Standalone BLIP image captioning script with GPU/CPU acceleration."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "Salesforce/blip-image-captioning-base"

# Global model state
_processor: Optional[BlipProcessor] = None
_model: Optional[BlipForConditionalGeneration] = None
_device: str = "cpu"
_use_half: bool = False


def load_model(
    model_name: str = MODEL_NAME,
    device: Optional[str] = None,
    use_half: Optional[bool] = None,
) -> Tuple[BlipProcessor, BlipForConditionalGeneration, str, bool]:
    """Initialize and cache the BLIP processor and model with optimal device settings."""
    global _processor, _model, _device, _use_half

    if _processor is not None and _model is not None:
        return _processor, _model, _device, _use_half

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if target_device == "cuda" and not torch.cuda.is_available():
        target_device = "cpu"

    half_precision = (target_device == "cuda") if use_half is None else (use_half and target_device == "cuda")

    if target_device == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass

    print(f"Loading BLIP captioning model on {target_device.upper()} (FP16={half_precision})...")

    # Use use_fast=False to avoid transformers deprecation warning
    _processor = BlipProcessor.from_pretrained(model_name, use_fast=False)

    dtype = torch.float16 if half_precision else torch.float32
    try:
        _model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
        )
    except TypeError:
        # Fallback for older transformers versions expecting torch_dtype
        _model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )

    _model.eval()
    _model.to(target_device)

    _device = target_device
    _use_half = half_precision
    print(f"Model loaded successfully on {_device.upper()}!")

    return _processor, _model, _device, _use_half


def generate_caption(image_path: str | Path, max_new_tokens: int = 30) -> str:
    """Generate a descriptive caption for the given image path.

    Args:
        image_path: Path to the image file.
        max_new_tokens: Maximum tokens in generated description.

    Returns:
        Formatted caption string with latency measurement.
    """
    path = Path(image_path).resolve()
    if not path.is_file():
        return f"Error: Image file not found at '{path}'"

    processor, model, device, use_half = load_model()

    try:
        with Image.open(path) as img:
            raw_image = img.convert("RGB")

        t0 = time.perf_counter()
        with torch.inference_mode():
            inputs = processor(raw_image, return_tensors="pt").to(device)
            if use_half and device == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
            else:
                out = model.generate(**inputs, max_new_tokens=max_new_tokens)

            caption = processor.decode(out[0], skip_special_tokens=True).strip()

        lat_ms = (time.perf_counter() - t0) * 1000.0
        return f"{caption} (Latency: {lat_ms:.1f}ms)"

    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM encountered during captioning. Falling back to CPU...")
        torch.cuda.empty_cache()
        global _device, _use_half
        _device = "cpu"
        _use_half = False
        if _model is not None:
            _model.to("cpu")
        return generate_caption(image_path, max_new_tokens=max_new_tokens)
    except Exception as e:
        return f"Error: {e}"


def main() -> None:
    # Allow specifying an image from CLI: python src/captioning.py [path_to_image]
    if len(sys.argv) > 1:
        target_img = Path(sys.argv[1]).resolve()
    else:
        # Defaults
        candidates = [
            PROJECT_ROOT / "assets" / "bus.jpg",
            PROJECT_ROOT / "static" / "bus.jpg",
            PROJECT_ROOT / "assets" / "test.jpg",
            PROJECT_ROOT / "static" / "test.jpg",
        ]
        target_img = next((p for p in candidates if p.is_file()), candidates[0])

    print("Target image:", target_img)
    if not target_img.is_file():
        print(f"Error: Target image '{target_img}' does not exist.")
        sys.exit(1)

    result = generate_caption(target_img)
    print("Result:", result)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
