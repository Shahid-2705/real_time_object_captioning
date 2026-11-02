"""Camera capture and management utilities."""

from __future__ import annotations

import logging
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)


def list_available_cameras(max_index: int = 5) -> List[int]:
    """Scan and return indices of functional video capture devices."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


class CameraManager:
    """Manages OpenCV VideoCapture with error recovery and FPS monitoring."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 30,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        self.cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._last_frame_time: float = time.time()
        self._frame_count: int = 0
        self._fps_calc_interval: float = 0.5
        self._last_fps_update: float = time.time()
        self._frames_since_last_fps: int = 0

    def open(self) -> bool:
        """Initialize and open the video capture device."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        logger.info(f"Opening camera at index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            logger.error(f"Failed to open camera index {self.camera_index}.")
            return False

        # Configure properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Warm up by reading test frame
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning(f"Camera {self.camera_index} opened but could not read test frame.")
            return False

        self._last_frame_time = time.time()
        self._last_fps_update = self._last_frame_time
        self._frames_since_last_fps = 0
        logger.info(f"Camera {self.camera_index} initialized successfully ({frame.shape[1]}x{frame.shape[0]}).")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame, computing running FPS. Attempts reconnection on transient failures."""
        if self.cap is None or not self.cap.isOpened():
            if not self._reconnect():
                return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Frame read failed, attempting reconnection...")
            if self._reconnect():
                ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, None

        # Update FPS
        now = time.time()
        self._frame_count += 1
        self._frames_since_last_fps += 1
        elapsed = now - self._last_fps_update
        if elapsed >= self._fps_calc_interval:
            self._fps = self._frames_since_last_fps / elapsed
            self._frames_since_last_fps = 0
            self._last_fps_update = now

        self._last_frame_time = now
        return True, frame

    def _reconnect(self) -> bool:
        """Attempt to reopen the camera device."""
        for attempt in range(1, self.reconnect_attempts + 1):
            logger.warning(f"Reconnection attempt {attempt}/{self.reconnect_attempts}...")
            time.sleep(self.reconnect_delay)
            if self.open():
                return True
        logger.error(f"Exhausted all {self.reconnect_attempts} reconnection attempts.")
        return False

    @property
    def fps(self) -> float:
        """Current calculated FPS."""
        return self._fps

    @property
    def is_opened(self) -> bool:
        """Check if camera is currently opened."""
        return self.cap is not None and self.cap.isOpened()

    def release(self) -> None:
        """Release camera resource cleanly."""
        if self.cap is not None:
            logger.info("Releasing camera...")
            self.cap.release()
            self.cap = None

    def __enter__(self) -> CameraManager:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
