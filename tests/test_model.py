"""Tests for feature extraction and the model (M1.2).

The load-bearing test here is that training and serving derive features through
the same function. Everything else in the pipeline can be correct and the service
will still return nonsense if those two paths diverge, and the divergence produces
no error — only quietly wrong predictions.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from helpers import TempProject, prepared_project, sample_image_bytes, trained_project
from mlops.data.dataset import augment_image, feature_length, image_to_features, load_split
from mlops.models.model import CatsDogsModel, ModelError, build_model


def test_image_to_features_shape_and_range() -> None:
    """Features are a flat float32 vector in [0, 1] of the expected length."""
    image = Image.open(io.BytesIO(sample_image_bytes()))
    for size in (8, 16, 24):
        vector = image_to_features(image, size)
        assert vector.shape == (feature_length(size),)
        assert vector.dtype == np.float32
        assert float(vector.min()) >= 0.0 and float(vector.max()) <= 1.0


def test_image_to_features_is_deterministic() -> None:
    """The same image always produces the same vector."""
    image = Image.open(io.BytesIO(sample_image_bytes()))
    assert np.array_equal(image_to_features(image, 12), image_to_features(image, 12))


def test_image_to_features_normalises_mode_and_geometry() -> None:
    """Greyscale, alpha and odd-sized inputs all yield the same vector length."""
    expected = feature_length(10)
    for image in (
        Image.new("L", (40, 90), 128),
        Image.new("RGBA", (17, 3), (10, 20, 30, 255)),
        Image.new("RGB", (500, 120), (5, 5, 5)),
    ):
        assert image_to_features(image, 10).shape == (expected,)


def test_serving_and_training_share_one_feature_path() -> None:
    """A checkpoint scores an image identically whether reached via the loader or
    the serving helper.

    If these ever diverge the API returns confident nonsense with no error, so the
    equality is asserted directly rather than inferred.
    """
    with trained_project() as project:
        model = CatsDogsModel.load(project.config.path("paths.model_path"))
        split = load_split(project.config, "test", augment=False)
        path = project.root / split.paths[0]

        from_loader = model.predict_proba(split.features[:1])[0]
        with Image.open(path) as handle:
            handle.load()
            vector = image_to_features(handle, model.metadata.feature_size)
        from_serving = model.predict_proba(vector.reshape(1, -1))[0]

        assert np.allclose(from_loader, from_serving, atol=1e-9)


def test_augmentation_changes_pixels_but_not_shape() -> None:
    """Augmentation perturbs the image while preserving its dimensions."""
    with TempProject() as project:
        image = Image.open(io.BytesIO(sample_image_bytes(size=48)))
        rng = np.random.default_rng(3)
        augmented = augment_image(image, rng, project.config)
        assert augmented.size == image.size
        original = image_to_features(image, 16)
        changed = image_to_features(augmented, 16)
        assert not np.array_equal(original, changed), "augmentation had no effect"


def test_augmentation_is_seeded() -> None:
    """The same seed reproduces the same augmentation."""
    with TempProject() as project:
        image = Image.open(io.BytesIO(sample_image_bytes(size=48)))
        first = augment_image(image, np.random.default_rng(7), project.config)
        second = augment_image(image, np.random.default_rng(7), project.config)
        assert np.array_equal(image_to_features(first, 16), image_to_features(second, 16))


def test_augmentation_only_applies_to_the_training_split() -> None:
    """Augmented copies inflate the training split and leave the others alone."""
    with prepared_project(**{"data.augmentation.multiplier": 3}) as project:
        plain = load_split(project.config, "train", augment=False)
        augmented = load_split(project.config, "train", augment=True)
        assert len(augmented) == len(plain) * 3
        test_plain = load_split(project.config, "test", augment=False)
        test_augmented = load_split(project.config, "test", augment=True)
        assert len(test_augmented) == len(test_plain)


def test_build_model_rejects_an_unknown_type() -> None:
    """An unsupported model type fails immediately with a clear message."""
    with TempProject(**{"model.type": "transformer"}) as project:
        try:
            build_model(project.config)
        except ModelError as exc:
            assert "transformer" in str(exc)
        else:
            raise AssertionError("expected ModelError")


def test_checkpoint_round_trips_with_its_metadata() -> None:
    """Saving and loading preserves both the weights and the provenance block."""
    with trained_project() as project:
        path = project.config.path("paths.model_path")
        model = CatsDogsModel.load(path)
        assert model.metadata.n_features == feature_length(model.metadata.feature_size)
        assert model.metadata.class_names == ["cat", "dog"]
        assert model.metadata.trained_at
        assert model.metadata.epochs > 0
        assert model.metadata.versions.get("scikit-learn")

        features = np.random.default_rng(0).random((4, model.metadata.n_features), dtype=np.float32)
        reloaded = CatsDogsModel.load(path)
        assert np.allclose(model.predict_proba(features), reloaded.predict_proba(features))


def test_probabilities_are_valid_distributions() -> None:
    """Every probability row is non-negative and sums to one."""
    with trained_project() as project:
        model = CatsDogsModel.load(project.config.path("paths.model_path"))
        split = load_split(project.config, "test", augment=False)
        proba = model.predict_proba(split.features)
        assert proba.shape == (len(split), 2)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        assert np.allclose(proba.sum(axis=1), 1.0)


def test_wrong_feature_width_is_rejected() -> None:
    """Feeding the wrong number of features raises rather than guessing."""
    with trained_project() as project:
        model = CatsDogsModel.load(project.config.path("paths.model_path"))
        try:
            model.predict_proba(np.zeros((1, model.metadata.n_features + 5), dtype=np.float32))
        except ModelError as exc:
            assert str(model.metadata.n_features) in str(exc)
        else:
            raise AssertionError("expected ModelError")


def test_loading_a_missing_checkpoint_explains_the_fix() -> None:
    """A missing model file points at the command that creates one."""
    with TempProject() as project:
        try:
            CatsDogsModel.load(project.root / "nothing.pkl")
        except ModelError as exc:
            assert "make train" in str(exc)
        else:
            raise AssertionError("expected ModelError")
