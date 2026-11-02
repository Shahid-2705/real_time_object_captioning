"""Standalone real-time object detection using YOLOv8 with GPU acceleration."""

import sys
import time
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
CAMERA_INDEX = 0

if not MODEL_PATH.is_file():
    MODEL_PATH = "yolov8n.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
use_half = (device == "cuda")

if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

print(f"Loading YOLOv8 model on {device.upper()} (FP16={use_half})...")
model = YOLO(str(MODEL_PATH))
if hasattr(model, "to"):
    model.to(device)
print(f"Model loaded successfully on {device.upper()}!")

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Could not open camera at index {CAMERA_INDEX}.")
    sys.exit(1)

print(f"Camera {CAMERA_INDEX} opened successfully. Press 'q' to quit.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        start_time = time.perf_counter()
        with torch.inference_mode():
            results = model.predict(frame, conf=0.35, device=device, half=use_half, verbose=False)
        end_time = time.perf_counter()

        annotated_frame = results[0].plot()
        lat_ms = (end_time - start_time) * 1000.0
        fps = 1.0 / max(end_time - start_time, 1e-5)

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f} | Det: {lat_ms:.1f}ms [{device.upper()}]",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLOv8 Detection (GPU Accelerated)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Exited cleanly.")
