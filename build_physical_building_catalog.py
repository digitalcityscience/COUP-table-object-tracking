"""Compatibility launcher for the relocated physical-building catalog tool."""

from pathlib import Path
import runpy


TOOL = Path(__file__).resolve().parent / "building_catalog" / "build.py"


if __name__ == "__main__":
    runpy.run_path(str(TOOL), run_name="__main__")
