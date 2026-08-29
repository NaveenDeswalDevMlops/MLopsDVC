"""Turn the raw archive into a versioned, split, 224x224 RGB dataset.

Split assignment hashes ``salt + relative source path`` rather than shuffling a
list. That makes the split a pure function of the file name: adding images never
moves an existing image between train and test, and two machines that run the
pipeline independently produce the same split. Random shuffling cannot promise
either of those, and a test set that silently changes between runs makes every
comparison downstream meaningless.

Outputs:
    data/processed/<split>/<class>/<name>.jpg   resized RGB JPEGs
    data/processed/manifest.csv                 one row per image
    artifacts/metrics/preprocess_stats.json     counts, geometry, dataset digest
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from mlops.config import Config
from mlops.logging_setup import get_logger

_LOGGER = get_logger(__name__)

SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = ("path", "split", "label", "class_name", "source", "sha256")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_HASH_BYTES = 8


class PreprocessError(RuntimeError):
    """Raised when the raw dataset cannot be processed."""


@dataclass(frozen=True)
class ManifestRow:
    """One processed image.

    Attributes:
        path: Path to the processed JPEG, relative to the project root.
        split: ``train``, ``val`` or ``test``.
        label: Integer class index.
        class_name: Human-readable class name.
        source: Source path relative to the raw directory.
        sha256: Digest of the processed file.
    """

    path: str
    split: str
    label: int
    class_name: str
    source: str
    sha256: str


def split_bucket(key: str, salt: str) -> float:
    """Map a stable key onto ``[0, 1)``.

    Args:
        key: Usually the source path relative to the raw directory.
        salt: Salt from ``data.split.hash_salt``; changing it redraws all splits.

    Returns:
        A deterministic value in ``[0, 1)``.
    """
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()[:_HASH_BYTES]
    return int.from_bytes(digest, "big") / float(1 << (8 * _HASH_BYTES))


def assign_split(bucket: float, config: Config) -> str:
    """Convert a bucket value into a split name.

    Args:
        bucket: Value in ``[0, 1)`` from :func:`split_bucket`.
        config: Effective configuration supplying the ratios.

    Returns:
        One of :data:`SPLITS`.
    """
    train = float(config.get("data.split.train", 0.8))
    val = float(config.get("data.split.val", 0.1))
    if bucket < train:
        return "train"
    if bucket < train + val:
        return "val"
    return "test"


def _rebalance(assignments: list[tuple[Path, float, str]]) -> list[tuple[Path, float, str]]:
    """Guarantee every split receives at least one image of this class.

    A tiny class (as in CI, where a handful of fixtures are used) can hash entirely
    into one split and leave validation empty, which fails much later with a
    confusing error. Deterministically donate the lowest-bucket images from the
    largest split instead.

    This is the single exception to the "adding data never moves an existing
    image" guarantee: a donated image returns to its hashed split once the class
    is large enough that no split is empty. The trade is deliberate — the
    alternative is a pipeline that crashes on small datasets — and it only ever
    fires on classes with barely more images than there are splits, which real
    datasets are not.

    Args:
        assignments: Triples of path, bucket and assigned split for one class.

    Returns:
        The rebalanced assignments.

    Raises:
        PreprocessError: If the class has fewer images than there are splits.
    """
    if len(assignments) < len(SPLITS):
        raise PreprocessError(
            f"a class needs at least {len(SPLITS)} images to populate every split, "
            f"got {len(assignments)}"
        )
    grouped: dict[str, list[tuple[Path, float, str]]] = {split: [] for split in SPLITS}
    for item in assignments:
        grouped[item[2]].append(item)

    for split in SPLITS:
        if grouped[split]:
            continue
        donor = max(SPLITS, key=lambda name: len(grouped[name]))
        grouped[donor].sort(key=lambda item: item[1])
        moved = grouped[donor].pop(0)
        grouped[split].append((moved[0], moved[1], split))
        _LOGGER.info("rebalanced an empty split", extra={"split": split, "donor": donor})

    return [item for split in SPLITS for item in grouped[split]]


def preprocess_image(source: Path, destination: Path, size: int, quality: int) -> str:
    """Convert one image to a square RGB JPEG.

    Args:
        source: Input image path.
        destination: Output JPEG path.
        size: Target side length in pixels.
        quality: JPEG quality.

    Returns:
        SHA-256 digest of the written file.

    Raises:
        PreprocessError: If the source cannot be decoded.
    """
    try:
        with Image.open(source) as handle:
            handle.load()
            image = handle.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    except Exception as exc:  # noqa: BLE001 - Pillow raises many unrelated types
        raise PreprocessError(f"cannot decode {source}: {exc}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=quality)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def read_manifest(path: Path | str) -> list[ManifestRow]:
    """Read a manifest CSV.

    Args:
        path: Manifest location.

    Returns:
        Parsed rows.

    Raises:
        PreprocessError: If the file is missing.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise PreprocessError(f"manifest not found: {manifest_path}; run the preprocess stage")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        return [
            ManifestRow(
                path=row["path"],
                split=row["split"],
                label=int(row["label"]),
                class_name=row["class_name"],
                source=row["source"],
                sha256=row["sha256"],
            )
            for row in csv.DictReader(handle)
        ]


def split_counts(rows: list[ManifestRow]) -> dict[str, dict[str, int]]:
    """Count images per split and class.

    Args:
        rows: Manifest rows.

    Returns:
        Nested ``{split: {class_name: count, "total": n}}`` mapping.
    """
    counts: dict[str, dict[str, int]] = {split: {"total": 0} for split in SPLITS}
    for row in rows:
        bucket = counts.setdefault(row.split, {"total": 0})
        bucket[row.class_name] = bucket.get(row.class_name, 0) + 1
        bucket["total"] += 1
    return counts


def dataset_digest(rows: list[ManifestRow]) -> str:
    """Compute a single digest identifying the whole processed dataset.

    Args:
        rows: Manifest rows.

    Returns:
        A hex digest that changes if any image, label or split changes.
    """
    hasher = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item.path):
        hasher.update(f"{row.path}|{row.split}|{row.label}|{row.sha256}".encode())
    return hasher.hexdigest()


def run(config: Config, clean: bool = True) -> dict:
    """Execute the preprocessing stage.

    Args:
        config: Effective configuration.
        clean: Remove the previous processed directory first.

    Returns:
        Statistics dictionary, also written to ``paths.preprocess_stats``.

    Raises:
        PreprocessError: If the raw directory is missing or a class is empty.
    """
    raw_dir = config.path("paths.raw_dir")
    processed_dir = config.path("paths.processed_dir")
    size = int(config.get("data.image_size", 224))
    quality = int(config.get("data.jpeg_quality", 92))
    salt = str(config.get("data.split.hash_salt", "catsdogs-v1"))
    classes = list(config.get("data.class_names", ["cat", "dog"]))

    if not raw_dir.is_dir():
        raise PreprocessError(
            f"raw directory {raw_dir} does not exist; run `make data` (synthetic) "
            "or `make data-kaggle` (real archive) first"
        )

    if clean and processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    skipped: list[str] = []

    for label, class_name in enumerate(classes):
        class_dir = raw_dir / class_name
        if not class_dir.is_dir():
            raise PreprocessError(
                f"expected a class directory at {class_dir}; run `make data` "
                "(synthetic) or `make data-kaggle` (real archive) first"
            )
        sources = sorted(
            path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not sources:
            raise PreprocessError(f"no images found for class {class_name!r} in {class_dir}")

        buckets = [
            (path, split_bucket(str(path.relative_to(raw_dir)), salt)) for path in sources
        ]
        assignments = _rebalance(
            [(path, bucket, assign_split(bucket, config)) for path, bucket in buckets]
        )

        for index, (source, _bucket, split) in enumerate(assignments):
            destination = processed_dir / split / class_name / f"{class_name}_{index:05d}.jpg"
            try:
                digest = preprocess_image(source, destination, size, quality)
            except PreprocessError as exc:
                skipped.append(str(source))
                _LOGGER.warning("skipped an undecodable image", extra={"error": str(exc)})
                continue
            rows.append(
                ManifestRow(
                    path=str(destination.resolve().relative_to(Path(config.root).resolve())),
                    split=split,
                    label=label,
                    class_name=class_name,
                    source=str(source.resolve().relative_to(Path(raw_dir).resolve())),
                    sha256=digest,
                )
            )

    if not rows:
        raise PreprocessError("every image failed to decode; nothing was written")

    manifest_path = config.path("paths.manifest_csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item.path):
            writer.writerow(row.__dict__)

    stats = {
        "image_size": size,
        "channels": int(config.get("data.channels", 3)),
        "classes": classes,
        "counts": split_counts(rows),
        "total_images": len(rows),
        "skipped": skipped,
        "dataset_digest": dataset_digest(rows),
        "manifest": str(manifest_path.resolve().relative_to(Path(config.root).resolve())),
        "split_ratios": {
            "train": float(config.get("data.split.train", 0.8)),
            "val": float(config.get("data.split.val", 0.1)),
            "test": float(config.get("data.split.test", 0.1)),
        },
        "hash_salt": salt,
    }
    stats_path = config.path("paths.preprocess_stats")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    _LOGGER.info(
        "preprocessing complete",
        extra={
            "total": len(rows),
            "digest": stats["dataset_digest"][:12],
            "counts": stats["counts"],
        },
    )
    return stats


__all__ = [
    "MANIFEST_FIELDS",
    "ManifestRow",
    "PreprocessError",
    "SPLITS",
    "assign_split",
    "dataset_digest",
    "preprocess_image",
    "read_manifest",
    "run",
    "split_bucket",
    "split_counts",
]
