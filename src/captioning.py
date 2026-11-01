from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# -------------------------------
# Load BLIP model and processor
# -------------------------------
print("🔄 Loading BLIP captioning model... (may take a minute)")

MODEL_NAME = "Salesforce/blip-image-captioning-base"

processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"✅ Model loaded successfully on {device.upper()}!")

# -------------------------------
# Function: generate_caption
# -------------------------------
def generate_caption(image_path: str) -> str:
    """
    Generates a caption for the given image path.
    """
    try:
        raw_image = Image.open(image_path).convert('RGB')

        # Preprocess the image
        inputs = processor(raw_image, return_tensors="pt").to(device)

        # Generate caption
        out = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

    except Exception as e:
        return f"❌ Error: {str(e)}"

# -------------------------------
# Optional test
# -------------------------------
if __name__ == "__main__":
    test_img = "static/test.jpg"  # Change path if needed
    print("🖼️ Generating caption for:", test_img)
    caption = generate_caption(test_img)
    print("🗣️ Caption:", caption)
