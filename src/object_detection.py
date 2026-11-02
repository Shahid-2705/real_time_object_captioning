"""Standalone real-time object detection using YOLOv8."""

import sys
import time
from pathlib import Path
import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
CAMERA_INDEX = 0

if not MODEL_PATH.is_file():
    MODEL_PATH = "yolov8n.pt"

print("Loading YOLOv8 model...")
model = YOLO(str(MODEL_PATH))
print("Model loaded successfully!")

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

        start_time = time.time()
        results = model(frame, verbose=False)
        end_time = time.time()

        annotated_frame = results[0].plot()
        fps = 1.0 / max(end_time - start_time, 1e-5)
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLOv8 Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Exited cleanly.")
