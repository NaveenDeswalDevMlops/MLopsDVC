"""Turn raw request bytes into a prediction.

The predictor owns exactly one responsibility: bytes in, label and probabilities
out, with every rejection expressed as a typed exception the HTTP layer maps to a
status code. Keeping the decoding rules here rather than in the route means the
post-deployment checker and the tests exercise the same validation the API does.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from mlops.config import Config
from mlops.data.dataset import image_to_features
from mlops.logging_setup import get_logger, safe_payload_meta
from mlops.models.model import CatsDogsModel

_LOGGER = get_logger(__name__)

ALLOWED_PREFIXES = ("image/",)
MAX_BATCH = 32


class PredictionError(RuntimeError):
    """Base class for prediction failures."""


class InvalidImageError(PredictionError):
    """The payload could not be decoded as an image."""


class PayloadTooLargeError(PredictionError):
    """The payload exceeds the configured size limit."""


class UnsupportedMediaTypeError(PredictionError):
    """The declared content type is not an image type."""


class BatchSizeError(PredictionError):
    """The batch is empty or too large."""


@dataclass
class PredictionResult:
    """One prediction.

    Attributes:
        label: Predicted class name.
        label_index: Predicted class index.
        confidence: Probability of the predicted class.
        probabilities: Probability per class name.
        threshold: Threshold applied to the positive class.
        latency_ms: Inference latency in milliseconds.
        image: Non-identifying metadata about the input.
    """

    label: str
    label_index: int
    confidence: float
    probabilities: dict[str, float]
    threshold: float
    latency_ms: float
    image: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of this result."""
        return {
            "label": self.label,
            "label_index": self.label_index,
            "confidence": round(self.confidence, 6),
            "probabilities": {k: round(v, 6) for k, v in self.probabilities.items()},
            "threshold": self.threshold,
            "latency_ms": round(self.latency_ms, 3),
            "image": self.image,
        }


class Predictor:
    """Loads a checkpoint once and serves predictions from it.

    Attributes:
        model: The loaded model.
        model_version: Version string reported by the API.
        max_bytes: Largest accepted payload.
    """

    def __init__(self, model: CatsDogsModel, model_version: str, max_bytes: int) -> None:
        """Initialise the predictor.

        Args:
            model: A loaded model.
            model_version: Version string reported in responses.
            max_bytes: Largest accepted payload in bytes.
        """
        self.model = model
        self.model_version = model_version
        self.max_bytes = int(max_bytes)

    @classmethod
    def from_config(cls, config: Config) -> Predictor:
        """Build a predictor from configuration.

        Args:
            config: Effective configuration.

        Returns:
            A ready predictor.

        Raises:
            ModelError: If no checkpoint exists at ``paths.model_path``.
        """
        model = CatsDogsModel.load(config.path("paths.model_path"))
        return cls(
            model=model,
            model_version=str(config.get("serving.model_version", "local")),
            max_bytes=int(config.get("serving.max_upload_bytes", 5_242_880)),
        )

    # -- decoding ----------------------------------------------------------

    def _decode(self, payload: bytes, content_type: str | None) -> Image.Image:
        """Decode raw bytes into a PIL image, enforcing the request limits.

        Args:
            payload: Raw bytes.
            content_type: Declared content type, if any.

        Returns:
            The decoded image.

        Raises:
            PayloadTooLargeError: If the payload exceeds ``max_bytes``.
            UnsupportedMediaTypeError: If a non-image type was declared.
            InvalidImageError: If the bytes are empty or undecodable.
        """
        if len(payload) > self.max_bytes:
            raise PayloadTooLargeError(
                f"payload is {len(payload)} bytes; the limit is {self.max_bytes}"
            )
        if content_type and not content_type.lower().startswith(ALLOWED_PREFIXES):
            raise UnsupportedMediaTypeError(
                f"content type {content_type!r} is not an image type"
            )
        if not payload:
            raise InvalidImageError("the request body is empty")
        try:
            image = Image.open(io.BytesIO(payload))
            image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError(f"payload is not a decodable image: {exc}") from exc
        return image

    # -- inference ---------------------------------------------------------

    def predict_bytes(self, payload: bytes, content_type: str | None = None) -> PredictionResult:
        """Classify a single image.

        Args:
            payload: Raw image bytes.
            content_type: Declared content type, if any.

        Returns:
            The prediction.
        """
        started = time.perf_counter()
        image = self._decode(payload, content_type)
        original_size = image.size
        features = image_to_features(image, self.model.metadata.feature_size)
        probabilities = self.model.predict_proba(features.reshape(1, -1))[0]
        latency_ms = (time.perf_counter() - started) * 1000.0

        threshold = float(self.model.metadata.threshold)
        index = 1 if probabilities[1] >= threshold else 0
        classes = self.model.classes
        result = PredictionResult(
            label=classes[index],
            label_index=index,
            confidence=float(probabilities[index]),
            probabilities={name: float(probabilities[i]) for i, name in enumerate(classes)},
            threshold=threshold,
            latency_ms=latency_ms,
            image={
                **safe_payload_meta(payload, content_type),
                "width": original_size[0],
                "height": original_size[1],
                "mode": image.mode,
            },
        )
        _LOGGER.info(
            "prediction served",
            extra={
                "label": result.label,
                "confidence": round(result.confidence, 4),
                "latency_ms": round(latency_ms, 2),
                "image": result.image,
            },
        )
        return result

    def predict_batch(
        self, payloads: list[bytes], content_types: list[str | None] | None = None
    ) -> list[PredictionResult]:
        """Classify several images in one call.

        Args:
            payloads: Raw image bytes per item.
            content_types: Declared content types, aligned with ``payloads``.

        Returns:
            One result per input, in order.

        Raises:
            BatchSizeError: If the batch is empty or larger than :data:`MAX_BATCH`.
        """
        if not payloads:
            raise BatchSizeError("the batch is empty")
        if len(payloads) > MAX_BATCH:
            raise BatchSizeError(f"batch of {len(payloads)} exceeds the limit of {MAX_BATCH}")
        declared = content_types or [None] * len(payloads)
        return [
            self.predict_bytes(payload, declared[index] if index < len(declared) else None)
            for index, payload in enumerate(payloads)
        ]

    def predict_file(self, path: Path | str) -> PredictionResult:
        """Classify an image already on disk.

        Args:
            path: Image path.

        Returns:
            The prediction.
        """
        data = Path(path).read_bytes()
        return self.predict_bytes(data, content_type="image/jpeg")

    def info(self) -> dict[str, Any]:
        """Describe the loaded model.

        Returns:
            Identity, input contract and provenance.
        """
        metadata = self.model.metadata
        return {
            "model_version": self.model_version,
            "model_type": metadata.model_type,
            "classes": metadata.class_names,
            "threshold": metadata.threshold,
            "input": {
                "image_size": metadata.image_size,
                "feature_size": metadata.feature_size,
                "n_features": metadata.n_features,
                "channels": 3,
                "max_upload_bytes": self.max_bytes,
            },
            "training": {
                "trained_at": metadata.trained_at,
                "epochs": metadata.epochs,
                "train_samples": metadata.train_samples,
                "seed": metadata.seed,
                "dataset_digest": metadata.dataset_digest,
                "git_sha": metadata.git_sha,
                "params": metadata.params,
            },
            "versions": metadata.versions,
        }


def probabilities_from_array(array: np.ndarray, classes: list[str]) -> dict[str, float]:
    """Zip a probability row against class names.

    Args:
        array: Probability row.
        classes: Class names in index order.

    Returns:
        Mapping of class name to probability.
    """
    return {name: float(array[index]) for index, name in enumerate(classes)}


__all__ = [
    "BatchSizeError",
    "InvalidImageError",
    "MAX_BATCH",
    "PayloadTooLargeError",
    "PredictionError",
    "PredictionResult",
    "Predictor",
    "UnsupportedMediaTypeError",
    "probabilities_from_array",
]
