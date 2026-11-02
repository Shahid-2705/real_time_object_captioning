"""Robust, non-blocking text-to-speech engine using pyttsx3."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class SpeechEngine:
    """Dedicated background speech worker for real-time auditory assistance."""

    def __init__(
        self,
        enabled: bool = True,
        rate: int = 160,
        volume: float = 1.0,
        min_speech_interval_seconds: float = 3.0,
        queue_size: int = 3,
    ) -> None:
        self.enabled = enabled
        self.rate = rate
        self.volume = volume
        self.min_speech_interval = min_speech_interval_seconds
        self.queue_size = queue_size

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._last_spoken_text: str = ""
        self._last_spoken_time: float = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background speech thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning("Speech engine already running.")
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, name="SpeechWorker", daemon=True)
        self._worker_thread.start()
        logger.info("Speech engine worker started.")

    def _run_loop(self) -> None:
        """Background thread loop: initializes TTS once and processes requests."""
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            logger.info("pyttsx3 engine initialized successfully in worker thread.")
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 engine: {e}")
            engine = None

        while not self._stop_event.is_set():
            try:
                # Poll with timeout to allow checking stop_event
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None or self._stop_event.is_set():
                break

            text = item
            if not self.enabled or engine is None or not text.strip():
                self._queue.task_done()
                continue

            try:
                # Apply runtime volume/rate updates if changed
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS utterance error for '{text}': {e}")

            self._queue.task_done()

        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        logger.info("Speech engine worker stopped cleanly.")

    def speak(self, text: str, force: bool = False) -> bool:
        """Queue text for announcement.

        Applies deduplication and cooldown checks unless force=True.
        Drops oldest message if queue is full.
        """
        if not self.enabled:
            return False

        clean_text = text.strip()
        if not clean_text:
            return False

        now = time.time()
        with self._lock:
            # Check cooldown and repetition
            if not force:
                if clean_text == self._last_spoken_text and (now - self._last_spoken_time) < (self.min_speech_interval * 3):
                    logger.debug(f"Speech skipped (duplicate within cooldown): {clean_text}")
                    return False
                if (now - self._last_spoken_time) < self.min_speech_interval:
                    logger.debug(f"Speech skipped (cooldown active): {clean_text}")
                    return False

            self._last_spoken_text = clean_text
            self._last_spoken_time = now

        # Enqueue text (bounded queue: discard oldest if full)
        try:
            self._queue.put_nowait(clean_text)
            return True
        except queue.Full:
            try:
                # Discard oldest and push new
                dropped = self._queue.get_nowait()
                self._queue.task_done()
                logger.debug(f"Speech queue full; dropped oldest: '{dropped}'")
            except queue.Empty:
                pass

            try:
                self._queue.put_nowait(clean_text)
                return True
            except queue.Full:
                return False

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable spoken feedback."""
        self.enabled = enabled
        logger.info(f"Speech enabled set to: {self.enabled}")

    def toggle_speech(self) -> bool:
        """Toggle speech enabled state and return new state."""
        self.enabled = not self.enabled
        logger.info(f"Speech toggled: {'ENABLED' if self.enabled else 'DISABLED'}")
        return self.enabled

    def set_rate(self, rate: int) -> None:
        """Set speech rate in words per minute."""
        if rate > 0:
            self.rate = rate

    def set_volume(self, volume: float) -> None:
        """Set speech volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, volume))

    def stop(self, timeout: float = 2.0) -> None:
        """Cleanly terminate the speech worker."""
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except (queue.Full, Exception):
            pass

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
            self._worker_thread = None
