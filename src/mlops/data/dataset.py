"""Load processed images into feature matrices, with training-only augmentation.

One function, :func:`image_to_features`, is the single place where an image becomes
a model input. Training, evaluation, the API and the post-deployment check all call
it, so there is no way for serving to preprocess differently from training. That
class of bug is invisible in tests that only exercise one path, so the code makes
it impossible instead of testing for it.

Augmentation is applied to the training split only, and only in memory: the
processed 224x224 JPEGs on disk stay untouched so the dataset digest is stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from mlops.config import Config
from mlops.data.preprocess import ManifestRow, read_manifest
from mlops.logging_setup import get_logger

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class SplitData:
    """Feature matrix and labels for one split.

    Attributes:
        features: Array of shape ``(n_samples, n_features)`` in ``[0, 1]``.
        labels: Integer label array of shape ``(n_samples,)``.
        paths: Source path for each row, for error inspection in the UI.
    """

    features: np.ndarray
    labels: np.ndarray
    paths: list[str]

    def __len__(self) -> int:
        """Return the number of samples."""
        return int(self.features.shape[0])


def image_to_features(image: Image.Image, feature_size: int) -> np.ndarray:
    """Convert a PIL image into the flat feature vector the model consumes.

    Args:
        image: Any PIL image; converted to RGB first.
        feature_size: Side length the image is downscaled to before flattening.

    Returns:
        Float32 vector of length ``feature_size * feature_size * 3`` in ``[0, 1]``.
    """
    rgb = image.convert("RGB").resize((feature_size, feature_size), Image.Resampling.BILINEAR)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    return array.reshape(-1)


def feature_length(feature_size: int) -> int:
    """Return the feature vector length for a given feature size.

    Args:
        feature_size: Side length used by :func:`image_to_features`.

    Returns:
        The vector length.
    """
    return feature_size * feature_size * 3


def augment_image(image: Image.Image, rng: np.random.Generator, config: Config) -> Image.Image:
    """Apply seeded geometric and photometric jitter to a training image.

    Args:
        image: Source image.
        rng: Seeded generator, so an epoch is reproducible.
        config: Effective configuration supplying the augmentation ranges.

    Returns:
        The augmented image.
    """
    result = image
    if config.get("data.augmentation.horizontal_flip", True) and rng.random() < 0.5:
        result = result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    degrees = float(config.get("data.augmentation.rotation_degrees", 0) or 0)
    if degrees > 0:
        angle = float(rng.uniform(-degrees, degrees))
        result = result.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(127, 127, 127))

    brightness = float(config.get("data.augmentation.brightness", 0) or 0)
    if brightness > 0:
        result = ImageEnhance.Brightness(result).enhance(
            float(rng.uniform(1.0 - brightness, 1.0 + brightness))
        )

    contrast = float(config.get("data.augmentation.contrast", 0) or 0)
    if contrast > 0:
        result = ImageEnhance.Contrast(result).enhance(
            float(rng.uniform(1.0 - contrast, 1.0 + contrast))
        )
    return result


def load_split(
    config: Config,
    split: str,
    rows: list[ManifestRow] | None = None,
    augment: bool = False,
) -> SplitData:
    """Load one split into memory as a feature matrix.

    Args:
        config: Effective configuration.
        split: ``train``, ``val`` or ``test``.
        rows: Pre-read manifest rows; read from disk when omitted.
        augment: Request augmented copies. Honoured for the training split and
            ignored elsewhere; see below.

    Returns:
        The loaded split.

    Raises:
        ValueError: If the split contains no images.
    """
    manifest = rows if rows is not None else read_manifest(config.path("paths.manifest_csv"))
    selected = [row for row in manifest if row.split == split]
    if not selected:
        raise ValueError(f"split {split!r} contains no images; re-run preprocessing")

    # Augmentation is refused outside the training split rather than merely
    # discouraged. Augmenting validation or test data inflates the split with
    # near-duplicates and reports a metric for a distribution that will never be
    # served — and because it raises no error, the number just quietly becomes
    # wrong. Enforcing it here means no caller can make that mistake.
    if augment and split != "train":
        _LOGGER.warning(
            "augmentation requested for a non-training split and ignored",
            extra={"split": split},
        )
        augment = False

    feature_size = int(config.get("model.feature_size", 40))
    multiplier = int(config.get("data.augmentation.multiplier", 1)) if augment else 1
    enabled = bool(config.get("data.augmentation.enabled", True)) and augment
    seed = int(config.get("project.seed", 42))

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []

    for index, row in enumerate(selected):
        image_path = Path(row.path)
        if not image_path.is_absolute():
            image_path = config.root / image_path
        with Image.open(image_path) as handle:
            handle.load()
            base = handle.convert("RGB")
            vectors.append(image_to_features(base, feature_size))
            labels.append(row.label)
            paths.append(row.path)

            if enabled:
                for copy_index in range(max(0, multiplier - 1)):
                    rng = np.random.default_rng(seed + index * 97 + copy_index)
                    vectors.append(image_to_features(augment_image(base, rng, config), feature_size))
                    labels.append(row.label)
                    paths.append(row.path)

    _LOGGER.info(
        "split loaded",
        extra={
            "split": split,
            "images": len(selected),
            "rows": len(vectors),
            "augmented": enabled,
            "features": feature_length(feature_size),
        },
    )
    return SplitData(
        features=np.asarray(vectors, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        paths=paths,
    )


__all__ = [
    "SplitData",
    "augment_image",
    "feature_length",
    "image_to_features",
    "load_split",
]
