"""Unit tests for the integrated video processor pipeline."""

import unittest
from unittest.mock import MagicMock, patch

from src.config import AppConfig
from src.pipeline.video_processor import VideoProcessor


class TestPipeline(unittest.TestCase):
    """Test video processor pipeline functions."""

    def test_compose_announcement(self):
        # Both present
        msg1 = VideoProcessor.compose_announcement("2 persons, 1 dog", "a park with green grass")
        self.assertEqual(msg1, "Detected 2 persons, 1 dog. Scene shows a park with green grass.")

        # Only detections
        msg2 = VideoProcessor.compose_announcement("1 chair", "")
        self.assertEqual(msg2, "Detected 1 chair.")

        # Only caption
        msg3 = VideoProcessor.compose_announcement("", "a busy city street")
        self.assertEqual(msg3, "Scene shows a busy city street.")

        # Neither
        msg4 = VideoProcessor.compose_announcement("", "")
        self.assertEqual(msg4, "")

    @patch("src.pipeline.video_processor.ObjectDetector")
    @patch("src.pipeline.video_processor.ImageCaptioner")
    @patch("src.pipeline.video_processor.SpeechEngine")
    @patch("src.pipeline.video_processor.CameraManager")
    def test_pipeline_stop_lifecycle(
        self,
        mock_camera,
        mock_speech,
        mock_captioner,
        mock_detector,
    ):
        config = AppConfig()
        processor = VideoProcessor(config)

        # Call stop and ensure all subcomponents are stopped/released cleanly
        processor.stop()
        processor.camera.release.assert_called()
        processor.speech.stop.assert_called()


if __name__ == "__main__":
    unittest.main()
