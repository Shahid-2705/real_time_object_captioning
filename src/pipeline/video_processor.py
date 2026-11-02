"""Real-time video processing pipeline orchestrator optimized for GPU and CPU."""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np
import torch

from src.audio.speech_engine import SpeechEngine
from src.config import AppConfig
from src.utils.camera import CameraManager
from src.vision.captioner import AsyncCaptionWorker, ImageCaptioner
from src.vision.detector import ObjectDetector

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Orchestrates camera capture, YOLO detection, async BLIP captioning, and speech."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

        # Subsystems
        self.camera = CameraManager(
            camera_index=config.camera.index,
            width=config.camera.width,
            height=config.camera.height,
            target_fps=config.camera.fps,
            reconnect_attempts=config.camera.reconnect_attempts,
            reconnect_delay=config.camera.reconnect_delay_seconds,
        )

        self.detector = ObjectDetector(
            model_path=config.vision.yolo_model_path,
            confidence_threshold=config.vision.confidence_threshold,
            iou_threshold=config.vision.iou_threshold,
            img_size=config.vision.img_size,
            device=config.vision.device,
            use_half=config.vision.use_half,
            enable_tracking=config.vision.enable_tracking,
        )

        self.captioner_core = ImageCaptioner(
            model_name=config.caption.model_name,
            device=config.caption.device,
            use_half=config.caption.use_half,
            max_new_tokens=config.caption.max_new_tokens,
        )

        self.speech = SpeechEngine(
            enabled=config.audio.enable_speech,
            rate=config.audio.rate,
            volume=config.audio.volume,
            min_speech_interval_seconds=config.audio.min_speech_interval_seconds,
            queue_size=config.audio.queue_size,
        )

        self.caption_worker = AsyncCaptionWorker(
            captioner=self.captioner_core,
            interval_seconds=config.caption.interval_seconds,
            scene_change_threshold=config.caption.scene_change_threshold,
            similarity_threshold=config.caption.similarity_threshold,
            on_new_caption=self._on_new_caption,
        )

        # Runtime state
        self._is_running = False
        self._last_detection_summary: str = ""
        self._last_e2e_latency_ms: float = 0.0

    def _on_new_caption(self, caption: str, latency_ms: float) -> None:
        """Callback invoked when async captioning generates a new novel caption."""
        announcement = self.compose_announcement(self._last_detection_summary, caption)
        if announcement:
            self.speech.speak(announcement)

    @staticmethod
    def compose_announcement(detection_summary: str, caption: str) -> str:
        """Compose a factual auditory announcement combining YOLO detections and BLIP caption."""
        parts = []
        if detection_summary:
            parts.append(f"Detected {detection_summary}.")
        if caption and not caption.startswith("Error"):
            parts.append(f"Scene shows {caption}.")
        return " ".join(parts)

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        det_latency: float,
        cap_latency: float,
        e2e_latency: float,
        caption: str,
        detection_summary: str,
        is_captioning: bool,
    ) -> np.ndarray:
        """Render a clean, responsive assistive HUD onto the video frame with GPU telemetry."""
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # Top status bar background
        cv2.rectangle(canvas, (0, 0), (w, 42), (20, 20, 20), -1)

        # Performance Metrics: FPS & Latencies
        fps_text = f"FPS: {fps:.1f}"
        lat_text = f"Det: {det_latency:.0f}ms | BLIP: {cap_latency:.0f}ms | E2E: {e2e_latency:.0f}ms"

        # Device and VRAM Telemetry
        is_cuda = self.detector.device == "cuda" and torch.cuda.is_available()
        if is_cuda:
            alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            dev_name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
            device_text = f"[{dev_name} FP16: {alloc_mb:.0f}MB]"
        else:
            device_text = "[CPU]"

        cv2.putText(canvas, fps_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(canvas, lat_text, (130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(canvas, device_text, (max(w - 260, 360), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1)

        # Bottom banner for Caption and Detected objects
        banner_h = 70
        cv2.rectangle(canvas, (0, h - banner_h), (w, h), (15, 15, 15), -1)

        cap_display = caption if caption else "Analyzing scene..."
        if is_captioning:
            cap_display += " (Updating...)"

        speech_status = "VOICE: ON" if self.speech.enabled else "VOICE: OFF"
        cv2.putText(
            canvas,
            f"Caption: {cap_display}",
            (10, h - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        sub_info = f"Objects: {detection_summary or 'None'}  |  {speech_status} [S:Toggle, C:Now, Q:Quit]"
        cv2.putText(
            canvas,
            sub_info,
            (10, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (160, 230, 160),
            1,
        )

        return canvas

    def start(self) -> None:
        """Start workers and begin real-time processing loop."""
        logger.info("Starting visual assistance subsystems...")
        self.speech.start()
        self.caption_worker.start()

        if not self.camera.open():
            logger.error(f"Cannot proceed: unable to open camera index {self.config.camera.index}")
            self.stop()
            return

        self._is_running = True
        logger.info("Pipeline running. Focus camera window and press 'q' to quit.")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received.")
        finally:
            self.stop()

    def _main_loop(self) -> None:
        """Main camera acquisition and real-time inference loop."""
        while self._is_running:
            t0 = time.perf_counter()

            ret, frame = self.camera.read()
            if not ret or frame is None:
                logger.warning("Frame read failure in main loop.")
                time.sleep(0.01)
                continue

            # Submit frame to background caption worker (throttled by interval & scene change)
            self.caption_worker.submit_frame(frame)

            # Synchronous YOLO detection (GPU/Tensor Cores accelerated)
            det_result = self.detector.detect(frame)
            self._last_detection_summary = det_result.summary

            # Measure end-to-end latency for this frame
            t1 = time.perf_counter()
            self._last_e2e_latency_ms = (t1 - t0) * 1000.0

            # Render HUD overlay
            annotated = self.draw_hud(
                frame=det_result.annotated_frame,
                fps=self.camera.fps,
                det_latency=det_result.latency_ms,
                cap_latency=self.caption_worker.latest_latency_ms,
                e2e_latency=self._last_e2e_latency_ms,
                caption=self.caption_worker.latest_caption,
                detection_summary=det_result.summary,
                is_captioning=self.caption_worker.is_busy,
            )

            cv2.imshow(self.config.display.window_title, annotated)

            # Process keyboard events
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                logger.info("Quit command triggered by user.")
                break
            elif key in (ord("s"), ord("S")):
                new_state = self.speech.toggle_speech()
                logger.info(f"Speech toggled: {new_state}")
            elif key in (ord("c"), ord("C")):
                logger.info("Manual caption generation requested.")
                self.caption_worker.submit_frame(frame, force=True)
            elif key in (ord("t"), ord("T")):
                self.detector.enable_tracking = not self.detector.enable_tracking
                logger.info(f"Tracking toggled: {self.detector.enable_tracking}")

    def stop(self) -> None:
        """Cleanly terminate all pipeline workers, resources, and free GPU caches."""
        self._is_running = False
        logger.info("Shutting down visual assistance pipeline...")

        try:
            self.camera.release()
        except Exception as e:
            logger.error(f"Error releasing camera: {e}")

        try:
            self.caption_worker.stop()
        except Exception as e:
            logger.error(f"Error stopping caption worker: {e}")

        try:
            self.speech.stop()
        except Exception as e:
            logger.error(f"Error stopping speech engine: {e}")

        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Error destroying OpenCV windows: {e}")

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("GPU VRAM cache cleared.")
            except Exception:
                pass

        logger.info("All pipeline resources released cleanly.")
