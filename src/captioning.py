"""Standalone BLIP image captioning test script."""

import sys
from pathlib import Path
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "Salesforce/blip-image-captioning-base"

print("Loading BLIP captioning model...")
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded successfully on {device.upper()}!")


def generate_caption(image_path: str | Path) -> str:
    """Generates a caption for the given image path."""
    try:
        raw_image = Image.open(image_path).convert("RGB")
        with torch.inference_mode():
            inputs = processor(raw_image, return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=30)
            caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    # Test with static/bus.jpg or assets/bus.jpg
    test_img = PROJECT_ROOT / "assets" / "bus.jpg"
    if not test_img.is_file():
        test_img = PROJECT_ROOT / "static" / "bus.jpg"

    print("Generating caption for:", test_img)
    if test_img.is_file():
        caption = generate_caption(test_img)
        print("Caption:", caption)
    else:
        print("Test image not found.")
