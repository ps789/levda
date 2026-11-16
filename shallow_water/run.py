"""Run with python -m shallow_water.run, or directly from any directory."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levda_core.runner import run
from shallow_water.problem import PROBLEM


def main(argv=None):
    return run(PROBLEM, argv)


if __name__ == "__main__":
    main()
