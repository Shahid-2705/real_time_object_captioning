"""Unit tests for the speech engine worker and queue management."""

import time
import unittest
from unittest.mock import patch, MagicMock

from src.audio.speech_engine import SpeechEngine


class TestSpeechEngine(unittest.TestCase):
    """Test SpeechEngine thread lifecycle, queuing, and deduplication."""

    @patch("pyttsx3.init")
    def test_speech_queue_and_deduplication(self, mock_init):
        mock_engine = MagicMock()
        mock_init.return_value = mock_engine

        engine = SpeechEngine(
            enabled=True,
            rate=150,
            volume=0.9,
            min_speech_interval_seconds=0.5,
            queue_size=2,
        )

        engine.start()

        # Enqueue first message
        res1 = engine.speak("Test announcement")
        self.assertTrue(res1)

        # Immediate identical message should be rejected due to cooldown/dedup
        res2 = engine.speak("Test announcement")
        self.assertFalse(res2)

        # Wait past cooldown
        time.sleep(0.6)
        res3 = engine.speak("Second announcement")
        self.assertTrue(res3)

        # Test queue bounding
        for i in range(5):
            engine.speak(f"Alert {i}", force=True)

        # Verify queue did not exceed bounded limit
        self.assertLessEqual(engine._queue.qsize(), 2)

        # Test toggle
        engine.toggle_speech()
        self.assertFalse(engine.enabled)
        self.assertFalse(engine.speak("Should not speak"))

        # Clean shutdown
        engine.stop()
        self.assertFalse(engine._worker_thread.is_alive() if engine._worker_thread else False)


if __name__ == "__main__":
    unittest.main()
