"""Pytest configuration: make the repository importable without installing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
