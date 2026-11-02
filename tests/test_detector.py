"""Unit tests for object detector interfaces and tracking."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.vision.detector import Detection, ObjectDetector
from src.vision.tracker import CentroidTracker


class TestDetectorAndTracker(unittest.TestCase):
    """Test object detector logic and centroid tracker."""

    def test_build_detection_summary(self):
        detections = [
            Detection(box=(0, 0, 10, 10), confidence=0.9, class_id=0, class_name="person"),
            Detection(box=(10, 10, 20, 20), confidence=0.85, class_id=0, class_name="person"),
            Detection(box=(20, 20, 30, 30), confidence=0.8, class_id=56, class_name="chair"),
        ]
        summary = ObjectDetector.build_detection_summary(detections)
        self.assertEqual(summary, "2 persons, 1 chair")

    def test_empty_detection_summary(self):
        summary = ObjectDetector.build_detection_summary([])
        self.assertEqual(summary, "")

    @patch("src.vision.detector.YOLO")
    def test_detector_inference_mocked(self, mock_yolo_class):
        mock_yolo_instance = MagicMock()
        mock_yolo_class.return_value = mock_yolo_instance

        # Mock YOLO prediction result
        mock_result = MagicMock()
        mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_result.boxes = None  # No boxes
        mock_yolo_instance.predict.return_value = [mock_result]

        # Use an existing path so it doesn't log a warning
        import tempfile
        from pathlib import Path
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            detector = ObjectDetector(model_path=tmp.name, device="cpu")
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            res = detector.detect(frame)

        self.assertEqual(len(res.detections), 0)
        self.assertGreaterEqual(res.latency_ms, 0.0)
        self.assertEqual(res.summary, "")

    def test_centroid_tracker(self):
        tracker = CentroidTracker(max_disappeared=5, max_distance=50.0)
        # First frame: one object
        rects = [(10, 10, 30, 30)]  # centroid at (20, 20)
        objects = tracker.update(rects)
        self.assertEqual(len(objects), 1)
        self.assertIn(0, objects)
        self.assertTrue(np.array_equal(objects[0], [20, 20]))

        # Second frame: object slightly moved
        rects2 = [(12, 12, 32, 32)]  # centroid at (22, 22)
        objects2 = tracker.update(rects2)
        self.assertEqual(len(objects2), 1)
        self.assertIn(0, objects2)
        self.assertTrue(np.array_equal(objects2[0], [22, 22]))


if __name__ == "__main__":
    unittest.main()
