import cv2
import torch
import pyttsx3
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import threading
import queue

# -------------------------------
# Configurations
# -------------------------------
CAMERA_INDEX = 0
YOLO_MODEL_PATH = "../models/yolov8n.pt"
CAPTION_FREQUENCY = 30  # Generate caption every N frames

# -------------------------------
# Load YOLOv8 model
# -------------------------------
print("🔄 Loading YOLOv8 model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("✅ YOLOv8 loaded!")

# -------------------------------
# Load BLIP captioning model
# -------------------------------
print("🔄 Loading BLIP captioning model...")
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model.to(device)

print(f"✅ BLIP loaded on {device.upper()}!")

# -------------------------------
# Text-to-Speech (Stable Queue-Based Design)
# -------------------------------
speech_queue = queue.Queue()

def tts_worker():
    """
    Dedicated TTS worker thread.
    A fresh engine is created per utterance to avoid pyttsx3 deadlocks.
    """
    while True:
        text = speech_queue.get()
        if text is None:
            break

        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 1.0)

            engine.say(text)
            engine.runAndWait()
            engine.stop()

        except Exception as e:
            print("🔊 TTS Error:", e)

        speech_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text: str):
    """Queue text for speech output"""
    speech_queue.put(text)

# -------------------------------
# Caption Generation
# -------------------------------
def generate_caption(frame) -> str:
    try:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(image, return_tensors="pt").to(device)
        output = blip_model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Caption error: {e}"

# -------------------------------
# Open Camera
# -------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"❌ Could not open camera at index {CAMERA_INDEX}")
    exit()

print("✅ Camera opened. Press 'q' to quit.\n")

frame_count = 0
last_caption = ""

# -------------------------------
# Real-Time Processing Loop
# -------------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break

        frame_count += 1

        # YOLO Detection
        results = yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()

        # Caption generation (every N frames)
        if frame_count % CAPTION_FREQUENCY == 0:
            new_caption = generate_caption(frame)

            # Speak only if caption changed
            if new_caption and new_caption != last_caption:
                last_caption = new_caption
                print("🗣️ Caption:", last_caption)
                speak(last_caption)

        # Overlay caption on video
        cv2.putText(
            annotated_frame,
            f"Caption: {last_caption}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Real-Time Object Captioning Assist", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean shutdown
    cap.release()
    cv2.destroyAllWindows()
    speech_queue.put(None)  # Stop TTS thread
    tts_thread.join()
    print("✅ Exited cleanly.")
