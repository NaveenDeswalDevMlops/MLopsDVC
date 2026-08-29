"""Train the classifier and record everything about the run.

The loop is deliberately epoch-based even though the estimator could be fitted in
one call. Epochs give a stepped metric series, which is what makes the MLflow run
and the dashboard's learning curves informative rather than decorative, and they
are what early stopping needs to act on.

The best epoch by validation accuracy is the checkpoint that gets saved — not the
last one. Training accuracy that keeps climbing while validation accuracy turns over
is the normal shape of overfitting, and saving the final epoch would ship the worse
model of the two.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from mlops.config import Config
from mlops.data.dataset import SplitData, load_split
from mlops.data.preprocess import read_manifest
from mlops.logging_setup import get_logger
from mlops.models import plots
from mlops.models.model import CatsDogsModel, build_model
from mlops.tracking.tracker import Tracker

_LOGGER = get_logger(__name__)


@dataclass
class TrainingResult:
    """Outcome of a training run.

    Attributes:
        model_path: Where the checkpoint was written.
        run_id: Tracking run id.
        epochs_run: Number of epochs actually executed.
        best_epoch: Epoch whose weights were kept.
        history: Per-epoch metric series.
        final_metrics: Metrics for the kept epoch.
        duration_seconds: Wall-clock training time.
        plots: Paths of the generated figures.
    """

    model_path: str
    run_id: str
    epochs_run: int
    best_epoch: int
    history: dict[str, list[float]] = field(default_factory=dict)
    final_metrics: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    plots: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of this result."""
        return {
            "model_path": self.model_path,
            "run_id": self.run_id,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "final_metrics": self.final_metrics,
            "duration_seconds": round(self.duration_seconds, 3),
            "plots": self.plots,
        }


def _epoch_indices(
    n_rows: int, fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """Choose the rows seen in one epoch.

    Args:
        n_rows: Number of available training rows.
        fraction: Fraction of the training set to sample.
        rng: Seeded generator.

    Returns:
        Shuffled row indices.
    """
    take = max(2, int(round(n_rows * max(0.05, min(1.0, fraction)))))
    return rng.choice(n_rows, size=take, replace=False)


def _score(model: CatsDogsModel, data: SplitData) -> tuple[float, float]:
    """Compute loss and accuracy for a split.

    Args:
        model: The model to score.
        data: The split to score against.

    Returns:
        Tuple of ``(log_loss, accuracy)``.
    """
    proba = model.predict_proba(data.features)
    loss = float(log_loss(data.labels, proba, labels=[0, 1]))
    accuracy = float(accuracy_score(data.labels, model.predict(data.features)))
    return loss, accuracy


def run(config: Config, use_mlflow: bool | None = None) -> TrainingResult:
    """Train a model end to end.

    Args:
        config: Effective configuration.
        use_mlflow: Force MLflow tracking on or off; ``None`` follows config.

    Returns:
        The training result.

    Raises:
        ValueError: If the processed dataset is missing or a split is empty.
    """
    config.ensure_dirs()
    started = time.perf_counter()

    manifest = read_manifest(config.path("paths.manifest_csv"))
    stats_path = config.path("paths.preprocess_stats")
    dataset_digest = ""
    if stats_path.is_file():
        dataset_digest = str(json.loads(stats_path.read_text(encoding="utf-8")).get("dataset_digest", ""))

    train_data = load_split(config, "train", rows=manifest, augment=True)
    val_data = load_split(config, "val", rows=manifest, augment=False)

    model = build_model(config, dataset_digest=dataset_digest)
    model.fit_scaler(train_data.features)

    epochs = int(config.get("training.epochs", 24))
    fraction = float(config.get("training.batch_fraction", 0.35))
    patience = int(config.get("training.early_stopping.patience", 6))
    min_delta = float(config.get("training.early_stopping.min_delta", 0.001))
    stopping_enabled = bool(config.get("training.early_stopping.enabled", True))
    rng = np.random.default_rng(int(config.get("project.seed", 42)))

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_state: Any = None
    best_accuracy = -1.0
    best_epoch = 0
    stale = 0
    epochs_run = 0

    tags = {
        "stage": "train",
        "model_type": model.metadata.model_type,
        "dataset_digest": dataset_digest[:16],
        "git_sha": model.metadata.git_sha,
    }

    with Tracker(config, run_name="train", tags=tags, use_mlflow=use_mlflow) as tracker:
        tracker.log_params(config.flat_params())
        tracker.log_params(
            {
                "train_rows": len(train_data),
                "val_rows": len(val_data),
                "n_features": model.metadata.n_features,
            }
        )

        for epoch in range(1, epochs + 1):
            epochs_run = epoch
            indices = _epoch_indices(len(train_data), fraction, rng)
            model.partial_fit(train_data.features[indices], train_data.labels[indices])

            train_loss, train_accuracy = _score(model, train_data)
            val_loss, val_accuracy = _score(model, val_data)

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            tracker.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch,
            )
            _LOGGER.info(
                "epoch complete",
                extra={
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_accuracy, 4),
                },
            )

            if val_accuracy > best_accuracy + min_delta:
                best_accuracy = val_accuracy
                best_epoch = epoch
                best_state = copy.deepcopy(model.estimator)
                stale = 0
            else:
                stale += 1
                if stopping_enabled and stale >= patience:
                    _LOGGER.info(
                        "early stopping", extra={"epoch": epoch, "best_epoch": best_epoch}
                    )
                    break

        if best_state is not None:
            model.estimator = best_state

        final_val_loss, final_val_accuracy = _score(model, val_data)
        final_train_loss, final_train_accuracy = _score(model, train_data)
        model.metadata.epochs = epochs_run
        model.metadata.train_samples = len(train_data)

        model_path = model.save(config.path("paths.model_path"))

        plots_dir = config.path("paths.plots_dir")
        loss_png = plots.loss_curve(history, plots_dir / "loss_curve.png")
        accuracy_png = plots.accuracy_curve(history, plots_dir / "accuracy_curve.png")

        history_payload = {
            "history": history,
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "model_type": model.metadata.model_type,
        }
        history_path = config.path("paths.history_path")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

        final_metrics = {
            "best_epoch": float(best_epoch),
            "epochs_run": float(epochs_run),
            "train_loss": final_train_loss,
            "train_accuracy": final_train_accuracy,
            "val_loss": final_val_loss,
            "val_accuracy": final_val_accuracy,
        }
        tracker.log_metrics(final_metrics)
        tracker.set_tags(
            {
                "best_epoch": best_epoch,
                "model_path": str(model_path),
                # Identifies the exact checkpoint this run produced. The promotion
                # gate compares it against the file on disk.
                "checkpoint_id": model.metadata.checkpoint_id,
                "model_trained_at": model.metadata.trained_at,
            }
        )
        tracker.log_sklearn_model(model.scaler, model.estimator)
        for artifact in (loss_png, accuracy_png, history_path, model_path):
            tracker.log_artifact(artifact, artifact_subdir="training")

        result = TrainingResult(
            model_path=str(model_path.resolve().relative_to(Path(config.root).resolve())),
            run_id=tracker.record.run_id,
            epochs_run=epochs_run,
            best_epoch=best_epoch,
            history=history,
            final_metrics=final_metrics,
            duration_seconds=time.perf_counter() - started,
            plots=[
                str(loss_png.resolve().relative_to(Path(config.root).resolve())),
                str(accuracy_png.resolve().relative_to(Path(config.root).resolve())),
            ],
        )

    _LOGGER.info("training complete", extra=result.final_metrics)
    return result


__all__ = ["TrainingResult", "run"]
