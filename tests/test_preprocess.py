"""Tests for the data preprocessing stage (M1.1).

The interesting property here is not that images get resized — it is that the
split is a pure function of the file name. Several of these tests exist to prove
that adding data cannot silently move an image from test into train, because that
failure is invisible at runtime and quietly invalidates every metric downstream.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from helpers import TempProject, prepared_project, write_raw_images
from mlops.data import preprocess
from mlops.data.preprocess import PreprocessError


def test_preprocess_resizes_to_the_configured_geometry() -> None:
    """Every processed image is a square RGB JPEG at the configured size."""
    with prepared_project() as project:
        size = int(project.config.get("data.image_size"))
        images = list(project.config.path("paths.processed_dir").rglob("*.jpg"))
        assert images, "preprocessing produced no images"
        for path in images:
            with Image.open(path) as handle:
                assert handle.size == (size, size), f"{path} is {handle.size}, expected {size}²"
                assert handle.mode == "RGB", f"{path} is mode {handle.mode}, expected RGB"


def test_manifest_covers_every_processed_image() -> None:
    """The manifest has exactly one row per file written."""
    with prepared_project() as project:
        rows = preprocess.read_manifest(project.config.path("paths.manifest_csv"))
        files = list(project.config.path("paths.processed_dir").rglob("*.jpg"))
        assert len(rows) == len(files)
        for row in rows:
            assert (project.root / row.path).is_file()
            assert row.split in preprocess.SPLITS
            assert row.label in (0, 1)


def test_split_assignment_is_deterministic() -> None:
    """The same path and salt always land in the same split."""
    with TempProject() as project:
        first = preprocess.split_bucket("cat/cat_001.jpg", "salt-a")
        second = preprocess.split_bucket("cat/cat_001.jpg", "salt-a")
        assert first == second
        assert 0.0 <= first < 1.0
        assert preprocess.split_bucket("cat/cat_002.jpg", "salt-a") != first
        assert preprocess.split_bucket("cat/cat_001.jpg", "salt-b") != first
        assert preprocess.assign_split(0.0, project.config) == "train"
        assert preprocess.assign_split(0.85, project.config) == "val"
        assert preprocess.assign_split(0.95, project.config) == "test"


def test_adding_images_does_not_reshuffle_existing_ones() -> None:
    """Growing the dataset leaves every existing image in its original split.

    This is the property a random shuffle cannot offer, and the reason the split is
    hashed rather than sampled. Class sizes here are large enough that the
    empty-split rebalance never fires, since that is the one documented case where
    an assignment may move.
    """
    with TempProject() as project:
        write_raw_images(project, per_class=40)
        preprocess.run(project.config)
        before = {
            row.source: row.split
            for row in preprocess.read_manifest(project.config.path("paths.manifest_csv"))
        }

        write_raw_images(project, per_class=80)  # rewrites 0-39, adds 40-79
        preprocess.run(project.config)
        after = {
            row.source: row.split
            for row in preprocess.read_manifest(project.config.path("paths.manifest_csv"))
        }

        for source, split in before.items():
            assert after.get(source) == split, f"{source} moved from {split} to {after.get(source)}"
        assert len(after) > len(before)


def test_every_split_is_populated_even_for_a_tiny_class() -> None:
    """A class too small to hash into all three splits is rebalanced, not dropped."""
    with TempProject() as project:
        write_raw_images(project, per_class=3)
        stats = preprocess.run(project.config)
        for split in preprocess.SPLITS:
            assert stats["counts"][split]["total"] > 0, f"{split} is empty"


def test_dataset_digest_reacts_to_content_changes() -> None:
    """The digest changes when the data changes and is stable when it does not."""
    with prepared_project() as project:
        rows = preprocess.read_manifest(project.config.path("paths.manifest_csv"))
        digest = preprocess.dataset_digest(rows)
        assert digest == preprocess.dataset_digest(list(reversed(rows))), "order must not matter"

        mutated = list(rows)
        mutated[0] = preprocess.ManifestRow(
            path=rows[0].path,
            split=rows[0].split,
            label=1 - rows[0].label,
            class_name=rows[0].class_name,
            source=rows[0].source,
            sha256=rows[0].sha256,
        )
        assert preprocess.dataset_digest(mutated) != digest, "a flipped label must change the digest"


def test_missing_raw_directory_is_reported_clearly() -> None:
    """A missing dataset raises an actionable error rather than an obscure one."""
    with TempProject() as project:
        try:
            preprocess.run(project.config)
        except PreprocessError as exc:
            assert "make data" in str(exc), "the error should say how to fix it"
        else:
            raise AssertionError("expected PreprocessError")


def test_undecodable_files_are_skipped_not_fatal() -> None:
    """One corrupt file does not abort the whole stage."""
    with TempProject() as project:
        write_raw_images(project, per_class=6)
        broken = project.config.path("paths.raw_dir") / "cat" / "corrupt.jpg"
        broken.write_bytes(b"this is not a JPEG")
        stats = preprocess.run(project.config)
        assert len(stats["skipped"]) == 1
        assert stats["total_images"] == 12


def test_preprocess_image_returns_a_content_digest(tmp_path: Path | None = None) -> None:
    """Preprocessing one image returns the digest of what it wrote."""
    with TempProject() as project:
        write_raw_images(project, per_class=1)
        source = next((project.config.path("paths.raw_dir") / "cat").glob("*.jpg"))
        destination = project.root / "out.jpg"
        digest = preprocess.preprocess_image(source, destination, 32, 90)
        assert destination.is_file()
        assert len(digest) == 64
        again = preprocess.preprocess_image(source, project.root / "out2.jpg", 32, 90)
        assert digest == again, "the same input must produce the same bytes"
