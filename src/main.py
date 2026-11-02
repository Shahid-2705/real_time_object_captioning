"""Main entry point for the AI Visual Assistance application."""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path

# Add project root to sys.path to ensure module imports succeed from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig
from src.pipeline.video_processor import VideoProcessor


def setup_logging(debug: bool = False) -> None:
    """Configure unified application logging with safe UTF-8 output."""
    level = logging.DEBUG if debug else logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format = "%H:%M:%S"

    # Configure stdout encoding on Windows
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> int:
    """Parse configuration and run the visual assistance pipeline."""
    try:
        config = AppConfig.from_cli_args()
    except Exception as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    setup_logging(debug=config.display.debug_mode)
    logger = logging.getLogger("main")
    logger.info("Initializing AI Visual Assistant...")

    processor = VideoProcessor(config)

    # Signal handlers for clean interruption
    def handle_signal(sig, frame):
        logger.info(f"Signal {sig} received. Initiating graceful shutdown...")
        processor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        processor.start()
        return 0
    except Exception as e:
        logger.critical(f"Unhandled exception in video processor: {e}", exc_info=True)
        processor.stop()
        return 1


if __name__ == "__main__":
    sys.exit(main())
