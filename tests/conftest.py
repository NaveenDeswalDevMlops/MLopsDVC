"""Pytest configuration.

Puts ``src/`` and this directory on the import path so ``import mlops`` and
``from helpers import ...`` both work without installing the package first.
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", Path(__file__).parent):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


# The pipeline logs at INFO on every stage. During a test run that buries the
# pass/fail lines, so the tests raise the floor; a failing assertion still shows
# its full traceback.
logging.disable(logging.INFO)
