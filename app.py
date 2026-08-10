#!/usr/bin/env python3
"""Compatibility shim: `python app.py` still works after the move to src/mpe_lkg.

The project has been a clone-and-run-app.py app since 2024 and the README said so
for two years, so that has to keep working whether or not the package is installed.
Everything real lives in src/mpe_lkg/.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from mpe_lkg.app import app, run  # noqa: E402,F401  (re-exported for old imports)

if __name__ == "__main__":
    run()
