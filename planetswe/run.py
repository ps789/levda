"""Run with python -m planetswe.run, or directly from any directory."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levda_core.runner import run

if __name__ == "__main__":
    run("planetswe")
