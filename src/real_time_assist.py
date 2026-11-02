"""Backwards-compatible entry point for Real-Time Assist.

Delegates execution to the modular VideoProcessor architecture while supporting
legacy command-line arguments and defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import main

if __name__ == "__main__":
    sys.exit(main())
