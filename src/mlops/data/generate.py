"""Generate a deterministic stand-in dataset when the Kaggle archive is absent.

The brief names the Kaggle Cats-and-Dogs archive, and ``make data-kaggle`` fetches
exactly that. But a grader, a CI runner and a fresh clone all need the pipeline to
run *now*, without credentials and without an 800 MB download, so this module
synthesises a labelled two-class image set with the same shape and the same
directory layout as the real archive:

    data/raw/cat/cat_00000.jpg
    data/raw/dog/dog_00000.jpg

**The classes overlap on purpose.** Every cue — coat hue, texture frequency, body
proportions, whether the ears are visible at all — is drawn from a distribution
that overlaps its counterpart, and each image gets independent background hue,
pose, lighting, noise and occlusion on top. No single cue separates the classes;
only their combination does, and imperfectly.

That is the point. A dataset where one channel gives the answer produces 100%
accuracy in the first epoch, flat learning curves, an early-stopping branch that
never fires and a monitoring baseline that cannot move — every downstream number
becomes decoration. With overlapping classes the train/validation gap is real and
the metrics mean something.

Everything is seeded, so the same seed always produces byte-identical images and
the dataset digest is stable.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

import numpy as np
from PIL import Image

from mlops.config import Config
from mlops.logging_setup import get_logger

_LOGGER = get_logger(__name__)

CANVAS = 160


@dataclass(frozen=True)
class ClassProfile:
    """Class-conditional cue distributions.

    Most fields are ``(mean, spread)`` pairs sampled once per image. The spreads
    are wide enough that the two profiles' samples interleave heavily.

    Attributes:
        warmth: Shift of the coat colour toward red (positive) or blue (negative).
        texture_frequency: Spatial frequency of the coat pattern.
        texture_axis: Probability the pattern runs in bands rather than rings.
        aspect: Body height-to-width ratio.
        ear_height: Height of the ear shapes above the head.
        ear_pointiness: 1.0 is a sharp triangle, 0.0 a round lobe.
        ear_probability: Chance the ears are drawn at all.
        snout_probability: Chance a snout and nose are drawn.
    """

    warmth: tuple[float, float]
    texture_frequency: tuple[float, float]
    texture_axis: float
    aspect: tuple[float, float]
    ear_height: tuple[float, float]
    ear_pointiness: tuple[float, float]
    ear_probability: float
    snout_probability: float


PROFILES: dict[str, ClassProfile] = {
    "cat": ClassProfile(
        warmth=(-0.07, 0.11),
        texture_frequency=(21.0, 8.0),
        texture_axis=0.72,
        aspect=(1.28, 0.30),
        ear_height=(0.20, 0.07),
        ear_pointiness=(0.78, 0.26),
        ear_probability=0.80,
        snout_probability=0.25,
    ),
    "dog": ClassProfile(
        warmth=(0.07, 0.11),
        texture_frequency=(11.0, 7.0),
        texture_axis=0.28,
        aspect=(0.82, 0.30),
        ear_height=(0.30, 0.09),
        ear_pointiness=(0.22, 0.26),
        ear_probability=0.80,
        snout_probability=0.75,
    ),
}


def _grids(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build normalised coordinate grids.

    Args:
        size: Side length of the square canvas.

    Returns:
        Tuple of ``(y, x)`` grids spanning ``[-1, 1]``.
    """
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    return np.meshgrid(axis, axis, indexing="ij")


def _background(rng: np.random.Generator, grid_y: np.ndarray, grid_x: np.ndarray) -> np.ndarray:
    """Paint a gradient background whose hue carries no class information.

    Args:
        rng: Seeded generator.
        grid_y: Vertical coordinate grid.
        grid_x: Horizontal coordinate grid.

    Returns:
        Float image array in ``[0, 1]``.
    """
    angle = rng.uniform(0.0, 2.0 * np.pi)
    ramp = (np.cos(angle) * grid_x + np.sin(angle) * grid_y + 1.0) / 2.0
    low = rng.uniform(0.25, 0.62, size=3).astype(np.float32)
    high = rng.uniform(0.45, 0.90, size=3).astype(np.float32)
    return low[None, None, :] + ramp[:, :, None] * (high - low)[None, None, :]


def _coat_colour(rng: np.random.Generator, warmth: float) -> np.ndarray:
    """Sample a coat colour, nudged along the warm/cool axis by ``warmth``.

    Args:
        rng: Seeded generator.
        warmth: Positive shifts red up and blue down; negative does the reverse.

    Returns:
        RGB triple in ``[0, 1]``.
    """
    base = float(rng.uniform(0.28, 0.62))
    colour = np.array(
        [base + warmth, base + rng.normal(0.0, 0.05), base - warmth], dtype=np.float32
    )
    colour += rng.normal(0.0, 0.05, size=3).astype(np.float32)
    return np.clip(colour, 0.06, 0.96)


def render(class_name: str, rng: np.random.Generator, size: int = CANVAS) -> np.ndarray:
    """Render one image for a class.

    Args:
        class_name: ``cat`` or ``dog``.
        rng: Seeded generator; the same seed always yields the same image.
        size: Canvas side length.

    Returns:
        Float image array of shape ``(size, size, 3)`` in ``[0, 1]``.
    """
    profile = PROFILES[class_name]
    grid_y, grid_x = _grids(size)
    canvas = _background(rng, grid_y, grid_x)

    warmth = float(rng.normal(*profile.warmth))
    frequency = max(3.0, float(rng.normal(*profile.texture_frequency)))
    aspect = max(0.45, float(rng.normal(*profile.aspect)))
    pointiness = float(np.clip(rng.normal(*profile.ear_pointiness), 0.0, 1.0))
    ear_height = max(0.06, float(rng.normal(*profile.ear_height)))

    # Whole-figure pose is class-independent, so position cannot leak the label.
    centre_y = float(rng.uniform(-0.16, 0.16))
    centre_x = float(rng.uniform(-0.16, 0.16))
    scale = float(rng.uniform(0.72, 1.10))
    tilt = float(rng.uniform(-0.35, 0.35))

    local_y = (grid_y - centre_y) * np.cos(tilt) - (grid_x - centre_x) * np.sin(tilt)
    local_x = (grid_y - centre_y) * np.sin(tilt) + (grid_x - centre_x) * np.cos(tilt)

    radius_y = 0.42 * scale * float(np.sqrt(aspect))
    radius_x = 0.42 * scale / float(np.sqrt(aspect))
    body = (local_y / radius_y) ** 2 + (local_x / radius_x) ** 2 <= 1.0

    if rng.random() < profile.ear_probability:
        for direction in (-1.0, 1.0):
            offset = direction * radius_x * float(rng.uniform(0.45, 0.85))
            top = -radius_y
            if pointiness > 0.5:
                width = radius_x * (0.55 - 0.28 * pointiness)
                ear = (
                    (local_y > top - ear_height)
                    & (local_y < top + 0.05)
                    & (np.abs(local_x - offset) < (local_y - (top - ear_height)) * width * 6.0)
                )
            else:
                ear = ((local_y - (top + ear_height * 0.9)) / (ear_height * 1.05)) ** 2 + (
                    (local_x - offset * 1.15) / (radius_x * 0.34)
                ) ** 2 <= 1.0
            body |= ear

    coat = _coat_colour(rng, warmth)
    if rng.random() < profile.texture_axis:
        wave = np.sin(local_y * frequency + rng.uniform(0.0, 3.0))
    else:
        wave = np.sin(np.sqrt(local_y**2 + local_x**2) * frequency + rng.uniform(0.0, 3.0))
    contrast = float(rng.uniform(0.12, 0.42))
    pattern = coat[None, None, :] * (1.0 - contrast + contrast * (0.5 + 0.5 * wave)[:, :, None])
    canvas = np.where(body[:, :, None], pattern, canvas)

    if rng.random() < profile.snout_probability:
        snout = ((local_y - radius_y * 0.55) / (radius_y * 0.42)) ** 2 + (
            local_x / (radius_x * 0.40)
        ) ** 2 <= 1.0
        canvas = np.where(snout[:, :, None], np.clip(pattern * 1.28, 0.0, 1.0), canvas)
        nose = ((local_y - radius_y * 0.80) / (radius_y * 0.16)) ** 2 + (
            local_x / (radius_x * 0.18)
        ) ** 2 <= 1.0
        canvas = np.where(nose[:, :, None], np.array([0.12, 0.10, 0.11], dtype=np.float32), canvas)

    for direction in (-1.0, 1.0):
        eye = ((local_y + radius_y * 0.22) / (radius_y * 0.13)) ** 2 + (
            (local_x - direction * radius_x * 0.32) / (radius_x * 0.11)
        ) ** 2 <= 1.0
        canvas = np.where(eye[:, :, None], np.array([0.90, 0.90, 0.72], dtype=np.float32), canvas)

    # Occlusion: a plain patch hiding part of the figure, so no cue is guaranteed
    # to be visible in any given image.
    if rng.random() < 0.28:
        patch_y = float(rng.uniform(-0.8, 0.8))
        patch_x = float(rng.uniform(-0.8, 0.8))
        half = float(rng.uniform(0.12, 0.30))
        patch = (np.abs(grid_y - patch_y) < half) & (np.abs(grid_x - patch_x) < half)
        colour = rng.uniform(0.15, 0.85, size=3).astype(np.float32)
        canvas = np.where(patch[:, :, None], colour[None, None, :], canvas)

    canvas = canvas * float(rng.uniform(0.72, 1.28))  # lighting
    canvas = canvas + rng.normal(0.0, 0.085, canvas.shape).astype(np.float32)  # sensor noise
    return np.clip(canvas, 0.0, 1.0)


def generate_dataset(config: Config, per_class: int | None = None, clean: bool = True) -> dict:
    """Write a synthetic raw dataset to ``paths.raw_dir``.

    Args:
        config: Effective configuration.
        per_class: Images per class; defaults to ``data.images_per_class``.
        clean: Remove any existing synthetic images first.

    Returns:
        Summary with per-class counts and the output directory.
    """
    count = int(per_class or config.get("data.images_per_class", 300))
    raw_dir = config.path("paths.raw_dir")
    classes = list(config.get("data.class_names", ["cat", "dog"]))
    seed = int(config.get("project.seed", 42))

    counts: dict[str, int] = {}
    for class_index, class_name in enumerate(classes):
        target = raw_dir / class_name
        if clean and target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            rng = np.random.default_rng(seed + class_index * 1_000_003 + index)
            array = (render(class_name, rng) * 255.0).astype(np.uint8)
            Image.fromarray(array, mode="RGB").save(
                target / f"{class_name}_{index:05d}.jpg", format="JPEG", quality=88
            )
        counts[class_name] = count
        _LOGGER.info("synthetic class written", extra={"class": class_name, "count": count})

    _LOGGER.info("synthetic dataset ready", extra={"raw_dir": str(raw_dir), "per_class": count})
    return {"raw_dir": str(raw_dir), "counts": counts}


__all__ = ["CANVAS", "ClassProfile", "PROFILES", "generate_dataset", "render"]
