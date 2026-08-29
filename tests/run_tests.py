"""Fallback test runner for environments without pytest.

The test files are written as plain functions with bare ``assert`` statements and
no pytest fixtures, so they run identically under ``pytest tests`` and under this
script. That is not redundancy for its own sake: `make test` uses pytest as the
brief requires, but a reviewer who has only cloned the repository can still verify
the suite with `python tests/run_tests.py` and no extra installs.

    python tests/run_tests.py [pattern]
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (ROOT / "src", HERE):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


# The pipeline logs at INFO on every stage. During a test run that buries the
# pass/fail lines, so the floor is raised here; a failing assertion still shows its
# full traceback.
logging.disable(logging.INFO)


def load_module(path: Path):
    """Import a test file as a module.

    Args:
        path: The file to import.

    Returns:
        The imported module.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    """Discover and run the tests.

    Args:
        argv: Optional single substring filter on test names.

    Returns:
        Process exit status.
    """
    pattern = argv[0] if argv else ""
    passed: list[str] = []
    failed: list[tuple[str, str]] = []
    started = time.perf_counter()

    for path in sorted(HERE.glob("test_*.py")):
        module = load_module(path)
        for name, function in sorted(vars(module).items()):
            if not name.startswith("test_") or not inspect.isfunction(function):
                continue
            if function.__module__ != module.__name__:
                continue
            if pattern and pattern not in f"{path.stem}.{name}":
                continue
            label = f"{path.stem}::{name}"
            try:
                function()
            except Exception:  # noqa: BLE001 - a failing test is the point
                failed.append((label, traceback.format_exc()))
                print(f"FAIL {label}", flush=True)
            else:
                passed.append(label)
                print(f"ok   {label}", flush=True)

    elapsed = time.perf_counter() - started
    print("")
    for label, trace in failed:
        print(f"--- {label} ---\n{trace}")
    print(f"{len(passed)} passed, {len(failed)} failed in {elapsed:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

