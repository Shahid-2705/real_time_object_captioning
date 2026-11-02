"""Quick camera feed test script."""

import sys
import cv2

camera_index = 0
if len(sys.argv) > 1:
    try:
        camera_index = int(sys.argv[1])
    except ValueError:
        pass

print(f"Opening camera index {camera_index}...")
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Could not open camera {camera_index}. Run 'python src/list_cameras.py' to check available devices.")
    sys.exit(1)

print(f"Camera {camera_index} opened. Press 'q' in the window to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Camera Feed Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released.")
