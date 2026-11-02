"""Unit tests for camera manager and error recovery."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.utils.camera import CameraManager, list_available_cameras


class TestCameraManager(unittest.TestCase):
    """Test CameraManager with simulated OpenCV capture device."""

    @patch("cv2.VideoCapture")
    def test_camera_open_success(self, mock_cv_capture):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv_capture.return_value = mock_cap

        camera = CameraManager(camera_index=0, width=640, height=480)
        opened = camera.open()
        self.assertTrue(opened)
        self.assertTrue(camera.is_opened)

        ret, frame = camera.read()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)

        camera.release()
        self.assertFalse(camera.is_opened)
        mock_cap.release.assert_called()

    @patch("cv2.VideoCapture")
    def test_camera_failure_and_reconnection_exhaustion(self, mock_cv_capture):
        mock_cap = MagicMock()
        # Fails to open
        mock_cap.isOpened.return_value = False
        mock_cv_capture.return_value = mock_cap

        camera = CameraManager(camera_index=99, reconnect_attempts=2, reconnect_delay=0.01)
        opened = camera.open()
        self.assertFalse(opened)

        ret, frame = camera.read()
        self.assertFalse(ret)
        self.assertIsNone(frame)

    @patch("cv2.VideoCapture")
    def test_list_available_cameras(self, mock_cv_capture):
        def side_effect(idx):
            cap = MagicMock()
            if idx in (0, 1):
                cap.isOpened.return_value = True
                cap.read.return_value = (True, np.zeros((10, 10, 3)))
            else:
                cap.isOpened.return_value = False
            return cap

        mock_cv_capture.side_effect = side_effect
        available = list_available_cameras(max_index=4)
        self.assertEqual(available, [0, 1])


if __name__ == "__main__":
    unittest.main()
