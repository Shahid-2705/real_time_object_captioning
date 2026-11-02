"""YOLOv8 real-time object detection module."""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Structured detection item."""
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


@dataclass
class DetectionResult:
    """Aggregated detection result for a frame."""
    detections: List[Detection]
    annotated_frame: np.ndarray
    latency_ms: float
    summary: str


class ObjectDetector:
    """Wraps Ultralytics YOLOv8 with robust resource and device management."""

    def __init__(
        self,
        model_path: Path | str,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: str = "auto",
        enable_tracking: bool = False,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.enable_tracking = enable_tracking
        self.device = self._resolve_device(device)

        self.model: Optional[YOLO] = None
        self._load_model()

    def _resolve_device(self, requested: str) -> str:
        """Resolve target device with fallback."""
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return requested

    def _load_model(self) -> None:
        """Load YOLO model weights with clear diagnostics."""
        if not self.model_path.is_file():
            # If default yolov8n.pt doesn't exist locally, YOLO can download or search
            logger.warning(
                f"YOLO weights not found at {self.model_path}. "
                "Attempting to download/load standard YOLOv8 nano model..."
            )
            model_target = "yolov8n.pt"
        else:
            model_target = str(self.model_path)

        try:
            logger.info(f"Loading YOLO model from {model_target} on {self.device}...")
            self.model = YOLO(model_target)
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            logger.info(f"YOLO model loaded successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Could not initialize YOLO detector: {e}")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run object detection on an input frame."""
        if self.model is None:
            raise RuntimeError("Detector model is not initialized.")

        start_time = time.perf_counter()
        detections: List[Detection] = []

        try:
            with torch.inference_mode():
                if self.enable_tracking:
                    results = self.model.track(
                        frame,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        imgsz=self.img_size,
                        device=self.device,
                        persist=True,
                        verbose=False,
                    )
                else:
                    results = self.model.predict(
                        frame,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        imgsz=self.img_size,
                        device=self.device,
                        verbose=False,
                    )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000.0

            result = results[0]
            annotated_frame = result.plot()

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                track_ids = None
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)

                names = result.names
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    cls_id = classes[i]
                    cls_name = names.get(cls_id, f"obj_{cls_id}")
                    conf = float(confs[i])
                    t_id = int(track_ids[i]) if track_ids is not None else None

                    detections.append(
                        Detection(
                            box=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name,
                            track_id=t_id,
                        )
                    )

            summary = self.build_detection_summary(detections)
            return DetectionResult(
                detections=detections,
                annotated_frame=annotated_frame,
                latency_ms=latency_ms,
                summary=summary,
            )

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA Out of Memory during detection! Switching to CPU fallback...")
            torch.cuda.empty_cache()
            self.device = "cpu"
            if hasattr(self.model, "to"):
                self.model.to("cpu")
            # Retry on CPU
            return self.detect(frame)
        except Exception as e:
            logger.error(f"Detection inference error: {e}")
            end_time = time.perf_counter()
            return DetectionResult(
                detections=[],
                annotated_frame=frame.copy(),
                latency_ms=(end_time - start_time) * 1000.0,
                summary="",
            )

    @staticmethod
    def build_detection_summary(detections: List[Detection]) -> str:
        """Create a factual, natural summary of detected objects without visual hallucinations."""
        if not detections:
            return ""

        counts = Counter(d.class_name for d in detections)
        items = []
        for name, count in sorted(counts.items(), key=lambda x: -x[1]):
            if count == 1:
                items.append(f"1 {name}")
            else:
                # Pluralize simple names
                plural = name + "s" if not name.endswith("s") else name
                items.append(f"{count} {plural}")

        return ", ".join(items)
