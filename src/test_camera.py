import cv2

# Open default webcam (0) — for DroidCam or Iriun, this should work.
# If it doesn’t, try 1 or 2 instead of 0.
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Could not open camera. Try changing index to 1 or 2.")
    exit()

print("✅ Camera opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    cv2.imshow("Camera Feed", frame)

    # Press 'q' to exit the window
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
