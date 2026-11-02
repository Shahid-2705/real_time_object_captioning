"""Standalone BLIP image captioning test script with GPU acceleration."""

import sys
import time
from pathlib import Path
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "Salesforce/blip-image-captioning-base"

device = "cuda" if torch.cuda.is_available() else "cpu"
use_half = (device == "cuda")

if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

print(f"Loading BLIP captioning model on {device.upper()} (FP16={use_half})...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if use_half else torch.float32,
)
model.eval()
model.to(device)
print(f"Model loaded successfully on {device.upper()}!")


def generate_caption(image_path: str | Path) -> str:
    """Generates a caption for the given image path."""
    try:
        raw_image = Image.open(image_path).convert("RGB")
        t0 = time.perf_counter()
        with torch.inference_mode():
            inputs = processor(raw_image, return_tensors="pt").to(device)
            if use_half:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model.generate(**inputs, max_new_tokens=30)
            else:
                out = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(out[0], skip_special_tokens=True)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return f"{caption} (Latency: {lat_ms:.1f}ms)"
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    test_img = PROJECT_ROOT / "assets" / "bus.jpg"
    if not test_img.is_file():
        test_img = PROJECT_ROOT / "static" / "bus.jpg"

    print("Generating caption for:", test_img)
    if test_img.is_file():
        caption = generate_caption(test_img)
        print("Caption:", caption)
    else:
        print("Test image not found.")
