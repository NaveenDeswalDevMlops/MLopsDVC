"""Shared fixtures for the test suite.

Every test builds its own project root under a temporary directory: its own
config, its own data, its own artifacts. Nothing reads or writes the developer's
real ``data/`` or ``artifacts/``, so running the tests can never destroy a trained
model, and tests cannot pass because of state a previous run left behind.

These are plain functions rather than pytest fixtures so the same files run under
``pytest`` and under ``python tests/run_tests.py``, which matters because the
fallback runner is what works in an environment where pytest is not installed.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mlops.config import Config  # noqa: E402


def base_config_document() -> dict[str, Any]:
    """Read the project's real configuration document.

    Returns:
        The parsed ``configs/config.yaml``.
    """
    return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))


class TempProject:
    """An isolated project root that cleans itself up.

    Attributes:
        root: The temporary project root.
        config: A :class:`~mlops.config.Config` pointing at it.
    """

    def __init__(self, **overrides: Any) -> None:
        """Create the project.

        Args:
            **overrides: Dotted config paths to override, e.g.
                ``**{"training.epochs": 3}``.
        """
        self.root = Path(tempfile.mkdtemp(prefix="mlops-test-"))
        document = base_config_document()
        # Tiny by default: these tests check behaviour, not accuracy.
        document["data"]["images_per_class"] = 12
        document["model"]["feature_size"] = 8
        document["training"]["epochs"] = 3
        document["training"]["early_stopping"]["enabled"] = False
        document["data"]["augmentation"]["multiplier"] = 1
        document["tracking"]["backend"] = "file"
        document["monitoring"]["kubernetes"]["enabled"] = False

        for dotted, value in overrides.items():
            cursor = document
            parts = dotted.split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value

        (self.root / "configs").mkdir(parents=True, exist_ok=True)
        (self.root / "configs" / "config.yaml").write_text(
            yaml.safe_dump(document), encoding="utf-8"
        )
        self.config = Config(raw=document, root=self.root)
        self.config.ensure_dirs()

    def cleanup(self) -> None:
        """Delete the temporary root."""
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> TempProject:
        """Enter the context.

        Returns:
            This project.
        """
        return self

    def __exit__(self, *_exc: Any) -> None:
        """Leave the context and clean up."""
        self.cleanup()


def write_raw_images(project: TempProject, per_class: int = 12) -> None:
    """Write a small deterministic raw dataset.

    Uses flat colour blocks rather than the real generator so the test suite stays
    fast and the class signal is unambiguous — these tests check that the plumbing
    works, not that the model is accurate.

    Args:
        project: The project to populate.
        per_class: Images per class.
    """
    raw = project.config.path("paths.raw_dir")
    for label, (name, colour) in enumerate(
        (("cat", (40, 60, 200)), ("dog", (200, 90, 40)))
    ):
        folder = raw / name
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            rng = np.random.default_rng(1000 * label + index)
            array = np.zeros((64, 64, 3), dtype=np.uint8)
            array[:, :] = colour
            array = np.clip(
                array.astype(np.int16) + rng.integers(-25, 25, array.shape), 0, 255
            ).astype(np.uint8)
            Image.fromarray(array, mode="RGB").save(folder / f"{name}_{index:03d}.jpg", quality=90)


def prepared_project(**overrides: Any) -> TempProject:
    """Build a project with raw images already preprocessed.

    Args:
        **overrides: Config overrides passed to :class:`TempProject`.

    Returns:
        The prepared project.
    """
    from mlops.data import preprocess

    project = TempProject(**overrides)
    write_raw_images(project)
    preprocess.run(project.config)
    return project


def trained_project(**overrides: Any) -> TempProject:
    """Build a project with a trained and evaluated model.

    Args:
        **overrides: Config overrides passed to :class:`TempProject`.

    Returns:
        The trained project.
    """
    from mlops.models import evaluate, train

    project = prepared_project(**overrides)
    train.run(project.config, use_mlflow=False)
    evaluate.run(project.config, use_mlflow=False)
    return project


def sample_image_bytes(colour: tuple[int, int, int] = (200, 90, 40), size: int = 64) -> bytes:
    """Encode a flat-colour JPEG.

    Args:
        colour: RGB fill colour.
        size: Side length in pixels.

    Returns:
        The encoded bytes.
    """
    import io

    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[:, :] = colour
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file.

    Args:
        path: File to read.

    Returns:
        The parsed document.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "ROOT",
    "TempProject",
    "base_config_document",
    "prepared_project",
    "read_json",
    "sample_image_bytes",
    "trained_project",
    "write_raw_images",
]
