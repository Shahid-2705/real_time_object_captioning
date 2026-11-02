"""Utility to detect and list functional camera devices."""

import sys
import cv2

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print(f"[OK] Camera index {i} is available")
        else:
            print(f"[WARN] Camera index {i} opened but cannot read frames")
        cap.release()
    else:
        print(f"[--] Camera index {i} not available")
