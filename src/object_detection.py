import cv2
from ultralytics import YOLO
import time

# -------------------------------
# Configuration
# -------------------------------
CAMERA_INDEX = 1 # DroidCam is at index 1
MODEL_PATH = "../models/yolov8n.pt"

# -------------------------------
# Load YOLO model
# -------------------------------
print("🔄 Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")

# -------------------------------
# Open the camera
# -------------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"❌ Could not open camera at index {CAMERA_INDEX}.")
    exit()

print("✅ Camera opened successfully.")
print("🎥 Press 'q' to quit.\n")

# -------------------------------
# Real-time detection loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame.")
        break

    # Run detection
    start_time = time.time()
    results = model(frame, verbose=False)
    end_time = time.time()

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Calculate FPS
    fps = 1 / (end_time - start_time)
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 - Real-Time Object Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
