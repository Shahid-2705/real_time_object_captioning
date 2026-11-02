"""Unit tests for configuration loading, validation, and serialization."""

import json
import tempfile
import unittest
from pathlib import Path

from src.config import AppConfig, AudioConfig, CameraConfig, CaptionConfig, VisionConfig


class TestConfig(unittest.TestCase):
    """Test configuration module."""

    def test_default_config_valid(self):
        """Ensure default configuration passes validation."""
        config = AppConfig()
        config.validate()
        self.assertEqual(config.camera.index, 0)
        self.assertGreater(config.caption.interval_seconds, 0)
        self.assertTrue(config.audio.enable_speech)
        self.assertTrue(config.vision.use_half)
        self.assertTrue(config.caption.use_half)

    def test_path_resolution(self):
        """Verify relative paths resolve properly against project root."""
        config = AppConfig()
        config.resolve_paths()
        self.assertTrue(config.vision.yolo_model_path.is_absolute())
        self.assertEqual(config.vision.yolo_model_path.name, "yolov8n.pt")

    def test_invalid_camera_config(self):
        """Check validation errors on invalid camera params."""
        with self.assertRaises(ValueError):
            CameraConfig(index=-1).validate()

        with self.assertRaises(ValueError):
            CameraConfig(width=0).validate()

    def test_invalid_vision_config(self):
        """Check validation on invalid detection thresholds."""
        with self.assertRaises(ValueError):
            VisionConfig(confidence_threshold=0.0).validate()

        with self.assertRaises(ValueError):
            VisionConfig(confidence_threshold=1.5).validate()

        with self.assertRaises(ValueError):
            VisionConfig(device="tpu").validate()

    def test_invalid_audio_config(self):
        """Check validation on audio parameters."""
        with self.assertRaises(ValueError):
            AudioConfig(rate=0).validate()

        with self.assertRaises(ValueError):
            AudioConfig(volume=-0.1).validate()

    def test_json_serialization_and_deserialization(self):
        """Test round-trip JSON configuration save and load."""
        config = AppConfig()
        config.camera.index = 2
        config.audio.rate = 180
        config.vision.confidence_threshold = 0.5
        config.vision.use_half = True

        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            json.dump(config.to_dict(), tmp)

        try:
            loaded = AppConfig.from_json(tmp_path)
            self.assertEqual(loaded.camera.index, 2)
            self.assertEqual(loaded.audio.rate, 180)
            self.assertAlmostEqual(loaded.vision.confidence_threshold, 0.5)
            self.assertTrue(loaded.vision.use_half)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_cli_parsing(self):
        """Test CLI arguments override defaults properly."""
        args = [
            "--camera-index", "1",
            "--confidence", "0.42",
            "--no-speech",
            "--speech-rate", "140",
            "--device", "cpu",
            "--no-half",
        ]
        config = AppConfig.from_cli_args(args)
        self.assertEqual(config.camera.index, 1)
        self.assertAlmostEqual(config.vision.confidence_threshold, 0.42)
        self.assertFalse(config.audio.enable_speech)
        self.assertEqual(config.audio.rate, 140)
        self.assertEqual(config.vision.device, "cpu")
        self.assertFalse(config.vision.use_half)
        self.assertFalse(config.caption.use_half)


if __name__ == "__main__":
    unittest.main()
