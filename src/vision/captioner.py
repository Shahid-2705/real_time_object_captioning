"""Asynchronous BLIP image captioning with scene-change detection and deduplication."""

from __future__ import annotations

import difflib
import logging
import queue
import re
import threading
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

logger = logging.getLogger(__name__)


def calculate_scene_change(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    """Compute normalized visual difference between two frames (0.0 to 1.0).

    Uses downsampled grayscale difference for high-speed, lightweight comparison.
    """
    if prev_frame is None or curr_frame is None:
        return 1.0

    try:
        # Resize to small dimension for fast computation
        h, w = 64, 64
        gray1 = cv2.cvtColor(cv2.resize(prev_frame, (w, h)), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv2.resize(curr_frame, (w, h)), cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        score = float(np.mean(diff)) / 255.0
        return score
    except Exception as e:
        logger.debug(f"Scene change calculation error: {e}")
        return 1.0


def normalize_caption_text(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and compressing whitespace."""
    clean = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(clean.split())


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate SequenceMatcher similarity ratio between normalized strings (0.0 to 1.0)."""
    norm1 = normalize_caption_text(text1)
    norm2 = normalize_caption_text(text2)
    if not norm1 and not norm2:
        return 1.0
    if not norm1 or not norm2:
        return 0.0
    return difflib.SequenceMatcher(None, norm1, norm2).ratio()


class ImageCaptioner:
    """Synchronous core BLIP caption generator."""

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = "auto",
        max_new_tokens: int = 30,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = self._resolve_device(device)

        self.processor: Optional[BlipProcessor] = None
        self.model: Optional[BlipForConditionalGeneration] = None
        self._load_model()

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for BLIP but unavailable. Using CPU.")
            return "cpu"
        return requested

    def _load_model(self) -> None:
        logger.info(f"Loading BLIP captioning model '{self.model_name}' on {self.device}...")
        try:
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
            self.model.eval()
            self.model.to(self.device)
            logger.info(f"BLIP model loaded successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            raise RuntimeError(f"Could not load BLIP captioner: {e}")

    def generate(self, frame: np.ndarray) -> Tuple[str, float]:
        """Generate caption from an OpenCV BGR frame.

        Returns:
            Tuple[str, float]: (generated caption, latency in milliseconds)
        """
        if self.model is None or self.processor is None:
            return "Captioner unavailable", 0.0

        start_time = time.perf_counter()
        try:
            # Convert BGR OpenCV image to PIL RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            with torch.inference_mode():
                inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                caption = self.processor.decode(outputs[0], skip_special_tokens=True).strip()

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return caption, latency_ms

        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA OOM during BLIP generation! Falling back to CPU...")
            torch.cuda.empty_cache()
            self.device = "cpu"
            self.model.to("cpu")
            return self.generate(frame)
        except Exception as e:
            logger.error(f"BLIP caption generation error: {e}")
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return f"Error: {e}", latency_ms


class AsyncCaptionWorker:
    """Manages asynchronous BLIP captioning in a dedicated background worker thread.

    Provides non-blocking submission, bounded frame queuing, scene change gating,
    text deduplication, and safe background inference.
    """

    def __init__(
        self,
        captioner: ImageCaptioner,
        interval_seconds: float = 3.0,
        scene_change_threshold: float = 0.20,
        similarity_threshold: float = 0.70,
        on_new_caption: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self.captioner = captioner
        self.interval_seconds = interval_seconds
        self.scene_change_threshold = scene_change_threshold
        self.similarity_threshold = similarity_threshold
        self.on_new_caption = on_new_caption

        # Single-element bounded queue to avoid stale frame accumulation
        self._input_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._latest_caption: str = ""
        self._latest_latency_ms: float = 0.0
        self._last_caption_time: float = 0.0
        self._last_captioned_frame: Optional[np.ndarray] = None
        self._is_busy = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background caption worker."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._run, name="CaptionWorker", daemon=True)
        self._worker_thread.start()
        logger.info("Async caption worker thread started.")

    def _run(self) -> None:
        """Worker loop processing submitted frames."""
        while not self._stop_event.is_set():
            try:
                item = self._input_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None or self._stop_event.is_set():
                break

            frame, is_forced = item
            with self._lock:
                self._is_busy = True

            try:
                caption, latency = self.captioner.generate(frame)
                now = time.time()

                with self._lock:
                    self._latest_latency_ms = latency
                    self._last_caption_time = now
                    self._last_captioned_frame = frame.copy()

                    # Check text similarity for deduplication
                    sim = calculate_text_similarity(self._latest_caption, caption)
                    is_novel = sim < self.similarity_threshold or is_forced or not self._latest_caption

                    if is_novel and not caption.startswith("Error"):
                        self._latest_caption = caption
                        logger.info(f"New caption [{latency:.1f}ms]: {caption}")
                        if self.on_new_caption:
                            try:
                                self.on_new_caption(caption, latency)
                            except Exception as cb_err:
                                logger.error(f"Callback on_new_caption error: {cb_err}")
                    else:
                        logger.debug(f"Caption duplicate/similar ({sim:.2f}): '{caption}' vs '{self._latest_caption}'")

            except Exception as e:
                logger.error(f"Async caption inference failure: {e}")
            finally:
                with self._lock:
                    self._is_busy = False
                self._input_queue.task_done()

        logger.info("Async caption worker terminated.")

    def submit_frame(self, frame: np.ndarray, force: bool = False) -> bool:
        """Submit a frame for asynchronous captioning if eligible."""
        now = time.time()

        with self._lock:
            if self._is_busy:
                return False

            # Check time interval
            if not force and (now - self._last_caption_time) < self.interval_seconds:
                return False

            # Check scene change
            if not force and self._last_captioned_frame is not None:
                scene_diff = calculate_scene_change(self._last_captioned_frame, frame)
                if scene_diff < self.scene_change_threshold:
                    logger.debug(f"Scene change too low ({scene_diff:.2f} < {self.scene_change_threshold}); skipping caption.")
                    return False

        # Attempt to put in single-element queue, discard stale if needed
        try:
            self._input_queue.put_nowait((frame.copy(), force))
            return True
        except queue.Full:
            try:
                _ = self._input_queue.get_nowait()
                self._input_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._input_queue.put_nowait((frame.copy(), force))
                return True
            except queue.Full:
                return False

    @property
    def latest_caption(self) -> str:
        with self._lock:
            return self._latest_caption

    @property
    def latest_latency_ms(self) -> float:
        with self._lock:
            return self._latest_latency_ms

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._is_busy

    def stop(self, timeout: float = 2.0) -> None:
        """Cleanly shutdown the worker thread."""
        self._stop_event.set()
        try:
            self._input_queue.put_nowait(None)
        except Exception:
            pass

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            self._worker_thread = None
