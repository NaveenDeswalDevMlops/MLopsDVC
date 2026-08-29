"""The classifier, its metadata and its on-disk format.

The baseline the brief names — logistic regression on flattened pixels — is
implemented with :class:`~sklearn.linear_model.SGDClassifier` using the log loss,
because ``partial_fit`` gives genuine per-epoch loss and accuracy curves. A
multi-layer perceptron is available behind ``model.type: mlp`` on exactly the same
training harness, so switching architectures is a config change rather than a code
change.

A checkpoint is self-describing. The pickle carries the fitted scaler, the fitted
estimator and a :class:`ModelMetadata` block recording the feature geometry, class
order, threshold, dataset digest and library versions. The inference service reads
its preprocessing parameters from that block rather than from a YAML file it might
not have, which is what stops a redeploy from silently changing the input pipeline.
"""

from __future__ import annotations

import platform
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from mlops.config import Config
from mlops.data.dataset import feature_length
from mlops.logging_setup import get_logger

_LOGGER = get_logger(__name__)

SUPPORTED_TYPES = ("logreg", "mlp")


class ModelError(RuntimeError):
    """Raised when a model cannot be built, loaded or used."""


@dataclass
class ModelMetadata:
    """Everything needed to reproduce and identify a checkpoint.

    Attributes:
        model_type: ``logreg`` or ``mlp``.
        feature_size: Side length images are downscaled to before flattening.
        n_features: Length of the flattened feature vector.
        image_size: Geometry of the stored processed images.
        class_names: Class order; index equals the model's label index.
        threshold: Decision threshold applied to the positive-class probability.
        seed: Seed used for training.
        checkpoint_id: Unique id for this checkpoint, used to prove that a metric
            and a model file describe the same thing.
        trained_at: UTC timestamp.
        dataset_digest: Digest of the processed dataset used for training.
        train_samples: Rows seen during training, including augmented copies.
        epochs: Epochs actually run (may be fewer than configured if stopped early).
        params: Hyperparameters passed to the estimator.
        versions: Library and interpreter versions at training time.
        git_sha: Commit the training ran from, when available.
    """

    checkpoint_id: str = ""
    model_type: str = "logreg"
    feature_size: int = 40
    n_features: int = 4800
    image_size: int = 224
    class_names: list[str] = field(default_factory=lambda: ["cat", "dog"])
    threshold: float = 0.5
    seed: int = 42
    trained_at: str = ""
    dataset_digest: str = ""
    train_samples: int = 0
    epochs: int = 0
    params: dict[str, Any] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    git_sha: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the metadata as a plain dictionary.

        Returns:
            A JSON-serialisable mapping.
        """
        return asdict(self)


def runtime_versions() -> dict[str, str]:
    """Capture the versions that define this training environment.

    Returns:
        Mapping of component name to version string.
    """
    import numpy

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "platform": platform.platform(),
    }


def git_sha() -> str:
    """Return the current commit SHA when the code is inside a Git checkout.

    Returns:
        The short SHA, or an empty string when Git is unavailable.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


class CatsDogsModel:
    """A scaler plus an incrementally trainable classifier.

    Attributes:
        metadata: Identity and provenance of this checkpoint.
    """

    def __init__(self, metadata: ModelMetadata, scaler: StandardScaler, estimator: Any) -> None:
        """Initialise the model.

        Args:
            metadata: Model metadata block.
            scaler: Fitted or unfitted feature scaler.
            estimator: An estimator exposing ``partial_fit`` and ``predict_proba``.
        """
        self.metadata = metadata
        self.scaler = scaler
        self.estimator = estimator

    @property
    def classes(self) -> list[str]:
        """Return the class names in label-index order."""
        return list(self.metadata.class_names)

    def fit_scaler(self, features: np.ndarray) -> None:
        """Fit the feature scaler on the training matrix.

        Args:
            features: Training feature matrix.
        """
        self.scaler.fit(features)

    def partial_fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Run one incremental training update.

        Args:
            features: Feature matrix for this epoch.
            labels: Matching label vector.
        """
        scaled = self.scaler.transform(features)
        self.estimator.partial_fit(scaled, labels, classes=np.array([0, 1]))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return per-class probabilities.

        Args:
            features: Feature matrix of shape ``(n, n_features)``.

        Returns:
            Array of shape ``(n, 2)`` whose rows sum to 1.

        Raises:
            ModelError: If the feature width does not match the checkpoint.
        """
        features = np.atleast_2d(np.asarray(features, dtype=np.float32))
        if features.shape[1] != self.metadata.n_features:
            raise ModelError(
                f"expected {self.metadata.n_features} features, got {features.shape[1]}"
            )
        scaled = self.scaler.transform(features)
        proba = np.asarray(self.estimator.predict_proba(scaled), dtype=np.float64)
        # Renormalise. The estimator's rows can drift from 1.0 by a float epsilon,
        # which scikit-learn's log_loss warns about and which would make the API's
        # documented guarantee — probabilities sum to one — technically false.
        totals = proba.sum(axis=1, keepdims=True)
        return np.divide(proba, totals, out=np.full_like(proba, 0.5), where=totals > 0)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return hard labels using the configured threshold.

        Args:
            features: Feature matrix.

        Returns:
            Integer label array.
        """
        proba = self.predict_proba(features)
        return (proba[:, 1] >= self.metadata.threshold).astype(np.int64)

    def save(self, path: Path | str) -> Path:
        """Persist the checkpoint.

        Args:
            path: Destination ``.pkl`` path.

        Returns:
            The path written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format_version": 1,
                "metadata": self.metadata.to_dict(),
                "scaler": self.scaler,
                "estimator": self.estimator,
            },
            destination,
        )
        _LOGGER.info(
            "checkpoint saved",
            extra={"path": str(destination), "bytes": destination.stat().st_size},
        )
        return destination

    @classmethod
    def load(cls, path: Path | str) -> CatsDogsModel:
        """Load a checkpoint from disk.

        Args:
            path: Path to the ``.pkl`` file.

        Returns:
            The loaded model.

        Raises:
            ModelError: If the file is missing or not a checkpoint of this format.
        """
        source = Path(path)
        if not source.is_file():
            raise ModelError(f"no checkpoint at {source}; train one with `make train`")
        try:
            payload = joblib.load(source)
        except Exception as exc:  # noqa: BLE001 - joblib surfaces many types
            raise ModelError(f"cannot load checkpoint {source}: {exc}") from exc
        if not isinstance(payload, dict) or "metadata" not in payload:
            raise ModelError(f"{source} is not a mlops-catsdogs checkpoint")
        metadata = ModelMetadata(**payload["metadata"])
        return cls(metadata=metadata, scaler=payload["scaler"], estimator=payload["estimator"])


def build_model(config: Config, dataset_digest: str = "") -> CatsDogsModel:
    """Construct an untrained model from configuration.

    Args:
        config: Effective configuration.
        dataset_digest: Digest of the dataset this model will be trained on.

    Returns:
        An untrained :class:`CatsDogsModel`.

    Raises:
        ModelError: If ``model.type`` is not supported.
    """
    model_type = str(config.get("model.type", "logreg")).lower()
    if model_type not in SUPPORTED_TYPES:
        raise ModelError(f"model.type must be one of {SUPPORTED_TYPES}, got {model_type!r}")

    feature_size = int(config.get("model.feature_size", 40))
    seed = int(config.get("project.seed", 42))
    learning_rate = float(config.get("training.learning_rate", 0.02))

    if model_type == "logreg":
        # `average=True` keeps a running mean of the weight vector. Without it the
        # final epoch's weights are whatever the last mini-batch pushed them to,
        # which on this feature width swings validation accuracy by ten points
        # between epochs and produces saturated, badly calibrated probabilities.
        # A decaying (`invscaling`) rate plus averaging converges smoothly and
        # keeps the reported confidence meaningful.
        params: dict[str, Any] = {
            "loss": "log_loss",
            "alpha": float(config.get("model.logreg.alpha", 0.05)),
            "penalty": str(config.get("model.logreg.penalty", "l2")),
            "learning_rate": str(config.get("model.logreg.schedule", "invscaling")),
            "eta0": learning_rate,
            "power_t": float(config.get("model.logreg.power_t", 0.3)),
            "average": bool(config.get("model.logreg.average", True)),
            "random_state": seed,
        }
        estimator: Any = SGDClassifier(**params)
    else:
        params = {
            "hidden_layer_sizes": tuple(config.get("model.mlp.hidden_layer_sizes", [128, 64])),
            "alpha": float(config.get("model.mlp.alpha", 0.0006)),
            "learning_rate_init": learning_rate,
            "random_state": seed,
        }
        estimator = MLPClassifier(**params)

    metadata = ModelMetadata(
        # A timestamp is not an identity: two trainings inside the same second
        # produce the same `trained_at`, and the promotion gate would then accept a
        # metric from one model as evidence for another. The id is unique per
        # checkpoint regardless of clock resolution.
        checkpoint_id=uuid.uuid4().hex[:12],
        model_type=model_type,
        feature_size=feature_size,
        n_features=feature_length(feature_size),
        image_size=int(config.get("data.image_size", 224)),
        class_names=list(config.get("data.class_names", ["cat", "dog"])),
        threshold=float(config.get("evaluation.threshold", 0.5)),
        seed=seed,
        dataset_digest=dataset_digest,
        params={key: str(value) for key, value in params.items()},
        versions=runtime_versions(),
        git_sha=git_sha(),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return CatsDogsModel(metadata=metadata, scaler=StandardScaler(), estimator=estimator)


__all__ = [
    "CatsDogsModel",
    "ModelError",
    "ModelMetadata",
    "SUPPORTED_TYPES",
    "build_model",
    "git_sha",
    "runtime_versions",
]
