"""Pytest configuration.

The checks in `script_checks/` are standalone scripts (each runs with plain `python` and exits
non-zero on failure). They are run through `test_script_checks.py`, not collected directly.
"""
import sys
from pathlib import Path

# Make the package importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

collect_ignore_glob = ["script_checks/*"]
