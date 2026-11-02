"""Unit tests for captioning logic, scene-change detection, and async worker."""

import time
import unittest
from unittest.mock import MagicMock
import numpy as np

from src.vision.captioner import (
    AsyncCaptionWorker,
    calculate_scene_change,
    calculate_text_similarity,
    normalize_caption_text,
)


class TestCaptioner(unittest.TestCase):
    """Test caption analysis and async worker queue."""

    def test_text_normalization(self):
        self.assertEqual(normalize_caption_text("A Person Walking!"), "a person walking")
        self.assertEqual(normalize_caption_text("  two   cats   on a mat. "), "two cats on a mat")

    def test_similarity_deduplication(self):
        # Identical text
        self.assertAlmostEqual(calculate_text_similarity("a dog on grass", "A dog on grass!"), 1.0)
        # Minor variant (should have high similarity)
        sim = calculate_text_similarity("a dog sitting on grass", "a dog standing on grass")
        self.assertGreater(sim, 0.70)
        # Completely different text
        diff_sim = calculate_text_similarity("a laptop on a table", "a car driving on the road")
        self.assertLess(diff_sim, 0.40)

    def test_scene_change_detection(self):
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame3 = np.ones((100, 100, 3), dtype=np.uint8) * 255

        # Identical frames -> 0.0 change
        self.assertAlmostEqual(calculate_scene_change(frame1, frame2), 0.0, places=2)
        # Black vs White -> 1.0 change
        self.assertAlmostEqual(calculate_scene_change(frame1, frame3), 1.0, places=2)

    def test_async_caption_worker_flow_and_shutdown(self):
        """Test async worker with mocked ImageCaptioner to avoid downloading weights."""
        mock_captioner = MagicMock()
        mock_captioner.generate.return_value = ("a person smiling in front of a camera", 45.0)

        captions_received = []

        def callback(caption, lat):
            captions_received.append((caption, lat))

        worker = AsyncCaptionWorker(
            captioner=mock_captioner,
            interval_seconds=0.05,
            scene_change_threshold=0.1,
            similarity_threshold=0.8,
            on_new_caption=callback,
        )

        worker.start()
        dummy_frame = np.zeros((64, 64, 3), dtype=np.uint8)

        # First frame should submit
        accepted = worker.submit_frame(dummy_frame, force=True)
        self.assertTrue(accepted)

        # Wait briefly for worker to process
        time.sleep(0.3)
        self.assertEqual(len(captions_received), 1)
        self.assertEqual(worker.latest_caption, "a person smiling in front of a camera")
        self.assertAlmostEqual(worker.latest_latency_ms, 45.0)

        # Clean shutdown
        worker.stop()
        self.assertFalse(worker._worker_thread.is_alive() if worker._worker_thread else False)


if __name__ == "__main__":
    unittest.main()
