"""Centralized configuration system for the real-time visual assistance application.

Supports command-line arguments, JSON configuration files, and programmatic customization
with robust path resolution and validation.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default project root: parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CameraConfig:
    """Camera hardware and capture configuration."""
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    reconnect_attempts: int = 3
    reconnect_delay_seconds: float = 1.0

    def validate(self) -> None:
        if self.index < 0:
            raise ValueError(f"Camera index must be >= 0, got {self.index}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Frame dimensions must be > 0, got {self.width}x{self.height}")
        if self.fps <= 0:
            raise ValueError(f"Target FPS must be > 0, got {self.fps}")


@dataclass
class VisionConfig:
    """Object detection and tracking configuration."""
    yolo_model_path: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "yolov8n.pt")
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    img_size: int = 640
    device: str = "auto"  # 'auto', 'cuda', 'cpu'
    enable_tracking: bool = False
    tracker_type: str = "bytetrack"

    def validate(self) -> None:
        if not (0.0 < self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in (0, 1], got {self.confidence_threshold}")
        if not (0.0 < self.iou_threshold <= 1.0):
            raise ValueError(f"iou_threshold must be in (0, 1], got {self.iou_threshold}")
        if self.img_size <= 0:
            raise ValueError(f"img_size must be positive, got {self.img_size}")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"device must be 'auto', 'cuda', or 'cpu', got {self.device}")


@dataclass
class CaptionConfig:
    """Image captioning and scene analysis configuration."""
    model_name: str = "Salesforce/blip-image-captioning-base"
    interval_seconds: float = 3.0
    scene_change_threshold: float = 0.20  # Normalized difference (0.0 - 1.0)
    similarity_threshold: float = 0.70    # String similarity ratio for deduplication
    max_new_tokens: int = 30
    device: str = "auto"                  # 'auto', 'cuda', 'cpu'

    def validate(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be > 0, got {self.interval_seconds}")
        if not (0.0 <= self.scene_change_threshold <= 1.0):
            raise ValueError(f"scene_change_threshold must be in [0, 1], got {self.scene_change_threshold}")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError(f"similarity_threshold must be in [0, 1], got {self.similarity_threshold}")
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {self.max_new_tokens}")


@dataclass
class AudioConfig:
    """Text-to-speech feedback configuration."""
    enable_speech: bool = True
    rate: int = 160
    volume: float = 1.0
    min_speech_interval_seconds: float = 3.0
    queue_size: int = 3

    def validate(self) -> None:
        if self.rate <= 0:
            raise ValueError(f"speech rate must be positive, got {self.rate}")
        if not (0.0 <= self.volume <= 1.0):
            raise ValueError(f"volume must be between 0.0 and 1.0, got {self.volume}")
        if self.min_speech_interval_seconds < 0:
            raise ValueError(f"min_speech_interval_seconds must be >= 0, got {self.min_speech_interval_seconds}")
        if self.queue_size <= 0:
            raise ValueError(f"queue_size must be positive, got {self.queue_size}")


@dataclass
class DisplayConfig:
    """Visualization and HUD configuration."""
    window_title: str = "AI Visual Assistant"
    show_hud: bool = True
    show_fps: bool = True
    debug_mode: bool = False
    overlay_font_scale: float = 0.65
    overlay_thickness: int = 2

    def validate(self) -> None:
        if self.overlay_font_scale <= 0:
            raise ValueError("overlay_font_scale must be positive")


@dataclass
class AppConfig:
    """Master application configuration coordinating all subsystems."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)

    def validate(self) -> None:
        """Validate all sub-configurations."""
        self.camera.validate()
        self.vision.validate()
        self.caption.validate()
        self.audio.validate()
        self.display.validate()

    def resolve_paths(self, base_path: Optional[Path] = None) -> None:
        """Resolve all relative file paths against the project root or specified base path."""
        root = Path(base_path or self.project_root).resolve()
        self.project_root = root

        yolo_path = Path(self.vision.yolo_model_path)
        if not yolo_path.is_absolute():
            self.vision.yolo_model_path = (root / yolo_path).resolve()
        else:
            self.vision.yolo_model_path = yolo_path.resolve()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary, serializing Paths as strings."""
        data = asdict(self)
        data["vision"]["yolo_model_path"] = str(self.vision.yolo_model_path)
        data["project_root"] = str(self.project_root)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppConfig:
        """Construct AppConfig from a dictionary."""
        config = cls()
        if "camera" in data:
            for k, v in data["camera"].items():
                if hasattr(config.camera, k):
                    setattr(config.camera, k, v)
        if "vision" in data:
            for k, v in data["vision"].items():
                if k == "yolo_model_path" and v:
                    config.vision.yolo_model_path = Path(v)
                elif hasattr(config.vision, k):
                    setattr(config.vision, k, v)
        if "caption" in data:
            for k, v in data["caption"].items():
                if hasattr(config.caption, k):
                    setattr(config.caption, k, v)
        if "audio" in data:
            for k, v in data["audio"].items():
                if hasattr(config.audio, k):
                    setattr(config.audio, k, v)
        if "display" in data:
            for k, v in data["display"].items():
                if hasattr(config.display, k):
                    setattr(config.display, k, v)
        if "project_root" in data and data["project_root"]:
            config.project_root = Path(data["project_root"])

        config.resolve_paths()
        config.validate()
        return config

    @classmethod
    def from_json(cls, json_path: Path | str) -> AppConfig:
        """Load configuration from a JSON file."""
        path = Path(json_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_cli_args(cls, args: Optional[list[str]] = None) -> AppConfig:
        """Parse command-line arguments and construct a validated AppConfig."""
        parser = argparse.ArgumentParser(
            description="AI-Powered Real-Time Visual Assistance System",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument("--config", type=str, default=None, help="Path to JSON configuration file")
        parser.add_argument("--camera-index", type=int, default=None, help="Webcam/video device index")
        parser.add_argument("--resolution", type=str, default=None, help="Camera resolution as WIDTHxHEIGHT (e.g. 640x480)")
        parser.add_argument("--yolo-model", type=str, default=None, help="Path to YOLOv8 weights (.pt)")
        parser.add_argument("--blip-model", type=str, default=None, help="HuggingFace model ID or path for BLIP")
        parser.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold (0.0 - 1.0)")
        parser.add_argument("--caption-interval", type=float, default=None, help="Interval between caption runs (seconds)")
        parser.add_argument("--min-speech-interval", type=float, default=None, help="Cooldown between speech announcements (seconds)")
        parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default=None, help="Compute device for inference")
        parser.add_argument("--enable-tracking", action="store_true", default=None, help="Enable object tracking across frames")
        parser.add_argument("--no-speech", action="store_true", help="Disable text-to-speech output")
        parser.add_argument("--speech-rate", type=int, default=None, help="Text-to-speech words per minute")
        parser.add_argument("--speech-volume", type=float, default=None, help="Text-to-speech volume (0.0 - 1.0)")
        parser.add_argument("--debug", action="store_true", help="Enable debug overlay and detailed logging")

        parsed = parser.parse_args(args)

        # Base configuration (from file if specified, else defaults)
        if parsed.config:
            config = cls.from_json(parsed.config)
        else:
            config = cls()
            config.resolve_paths()

        # Overwrite with explicit CLI flags
        if parsed.camera_index is not None:
            config.camera.index = parsed.camera_index
        if parsed.resolution:
            try:
                w, h = parsed.resolution.lower().split("x")
                config.camera.width = int(w)
                config.camera.height = int(h)
            except ValueError:
                raise ValueError(f"Invalid resolution format '{parsed.resolution}'. Expected WIDTHxHEIGHT e.g. 640x480")
        if parsed.yolo_model:
            config.vision.yolo_model_path = Path(parsed.yolo_model)
        if parsed.blip_model:
            config.caption.model_name = parsed.blip_model
        if parsed.confidence is not None:
            config.vision.confidence_threshold = parsed.confidence
        if parsed.caption_interval is not None:
            config.caption.interval_seconds = parsed.caption_interval
        if parsed.min_speech_interval is not None:
            config.audio.min_speech_interval_seconds = parsed.min_speech_interval
        if parsed.device:
            config.vision.device = parsed.device
            config.caption.device = parsed.device
        if parsed.enable_tracking is not None:
            config.vision.enable_tracking = parsed.enable_tracking
        if parsed.no_speech:
            config.audio.enable_speech = False
        if parsed.speech_rate is not None:
            config.audio.rate = parsed.speech_rate
        if parsed.speech_volume is not None:
            config.audio.volume = parsed.speech_volume
        if parsed.debug:
            config.display.debug_mode = True

        config.resolve_paths()
        config.validate()
        return config
