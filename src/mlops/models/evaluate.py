"""Evaluate the saved checkpoint on the held-out test split.

This stage writes ``artifacts/metrics/baseline.json``, which is the reference the
post-deployment check compares live traffic against. That file is therefore the
contract between "the model was good when we trained it" and "the model is still
good now that it is serving", and the two stages read the same keys.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from mlops.config import Config
from mlops.data.dataset import load_split
from mlops.data.preprocess import read_manifest
from mlops.logging_setup import get_logger
from mlops.models import plots
from mlops.models.model import CatsDogsModel
from mlops.tracking.tracker import Tracker

_LOGGER = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Metrics and artifacts produced by evaluation.

    Attributes:
        split: Split that was evaluated.
        metrics: Scalar metrics.
        confusion: Confusion matrix as nested lists.
        class_names: Class order.
        report: Text classification report.
        artifacts: Paths of files written.
        run_id: Tracking run id.
    """

    split: str
    metrics: dict[str, float]
    confusion: list[list[int]]
    class_names: list[str]
    report: str = ""
    artifacts: list[str] = field(default_factory=list)
    run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of this result."""
        return {
            "split": self.split,
            "metrics": self.metrics,
            "confusion_matrix": self.confusion,
            "class_names": self.class_names,
            "report": self.report,
            "artifacts": self.artifacts,
            "run_id": self.run_id,
        }


def compute_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute the standard binary classification metrics.

    Args:
        labels: True labels.
        probabilities: Positive-class probabilities.
        threshold: Decision threshold.

    Returns:
        Mapping of metric name to value. ROC-AUC is ``0.5`` when only one class is
        present, which is the honest value for a degenerate split rather than an
        exception that kills the pipeline.
    """
    predictions = (probabilities >= threshold).astype(np.int64)
    single_class = len(np.unique(labels)) < 2
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": 0.5 if single_class else float(roc_auc_score(labels, probabilities)),
        "log_loss": float(
            log_loss(labels, np.column_stack([1 - probabilities, probabilities]), labels=[0, 1])
        ),
        "samples": float(len(labels)),
    }


def build_model_card(config: Config, model: CatsDogsModel, result: EvaluationResult) -> str:
    """Render a Markdown model card.

    Args:
        config: Effective configuration.
        model: The evaluated model.
        result: Its evaluation result.

    Returns:
        The model card text.
    """
    metadata = model.metadata
    rows = "\n".join(
        f"| {key} | {value:.4f} |"
        for key, value in result.metrics.items()
        if key != "samples"
    )
    versions = "\n".join(f"| {key} | {value} |" for key, value in metadata.versions.items())
    return f"""# Model card — {config.get('tracking.registered_model_name', 'catsdogs-classifier')}

**Task.** Binary image classification, cat vs dog, for a pet adoption platform.

## Model
| Field | Value |
| --- | --- |
| Type | {metadata.model_type} |
| Input geometry | {metadata.image_size}x{metadata.image_size} RGB, downscaled to {metadata.feature_size}x{metadata.feature_size} |
| Features | {metadata.n_features} |
| Classes | {', '.join(metadata.class_names)} (index order) |
| Decision threshold | {metadata.threshold} |
| Epochs run | {metadata.epochs} |
| Training rows | {metadata.train_samples} |
| Seed | {metadata.seed} |
| Trained at | {metadata.trained_at} |
| Dataset digest | `{metadata.dataset_digest[:16]}` |
| Git commit | `{metadata.git_sha or 'n/a'}` |

## Test-split metrics
| Metric | Value |
| --- | --- |
{rows}

Evaluated on {int(result.metrics.get('samples', 0))} held-out images.

## Confusion matrix
Rows are true classes, columns predicted: `{result.confusion}`

## Environment
| Component | Version |
| --- | --- |
{versions}

## Intended use and limits
Intended as a first-pass triage aid for adoption listings, not an authority. It was
trained on a small, evenly balanced set and has seen no photographs of animals other
than the two classes; anything else is forced into one of them. Confidence is
calibrated only against the training distribution, so a low-confidence prediction
should route to a human rather than to a default.

_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}._
"""


def run(config: Config, split: str = "test", use_mlflow: bool | None = None) -> EvaluationResult:
    """Evaluate the checkpoint and write the baseline.

    Args:
        config: Effective configuration.
        split: Split to evaluate.
        use_mlflow: Force MLflow tracking on or off.

    Returns:
        The evaluation result.
    """
    config.ensure_dirs()
    started = time.perf_counter()

    model = CatsDogsModel.load(config.path("paths.model_path"))
    manifest = read_manifest(config.path("paths.manifest_csv"))
    data = load_split(config, split, rows=manifest, augment=False)

    probabilities = model.predict_proba(data.features)[:, 1]
    threshold = float(model.metadata.threshold)
    metrics = compute_metrics(data.labels, probabilities, threshold)
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(data.labels, predictions, labels=[0, 1]).tolist()
    report = classification_report(
        data.labels,
        predictions,
        labels=[0, 1],
        target_names=model.classes,
        zero_division=0,
    )

    plots_dir = config.path("paths.plots_dir")
    confusion_png = plots.confusion_matrix_plot(matrix, model.classes, plots_dir / "confusion_matrix.png")
    if len(np.unique(data.labels)) > 1:
        fpr, tpr, _ = roc_curve(data.labels, probabilities)
    else:
        fpr, tpr = [0.0, 1.0], [0.0, 1.0]
    roc_png = plots.roc_curve_plot(fpr, tpr, metrics["roc_auc"], plots_dir / "roc_curve.png")

    metrics_dir = config.path("paths.metrics_dir")
    report_path = metrics_dir / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    result = EvaluationResult(
        split=split,
        metrics=metrics,
        confusion=matrix,
        class_names=model.classes,
        report=report,
        artifacts=[
            str(confusion_png.resolve().relative_to(Path(config.root).resolve())),
            str(roc_png.resolve().relative_to(Path(config.root).resolve())),
            str(report_path.resolve().relative_to(Path(config.root).resolve())),
        ],
    )

    card = build_model_card(config, model, result)
    card_path = config.path("paths.model_card")
    card_path.write_text(card, encoding="utf-8")
    result.artifacts.append(str(card_path.resolve().relative_to(Path(config.root).resolve())))

    baseline = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split,
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": matrix,
        "class_names": model.classes,
        "model": model.metadata.to_dict(),
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    baseline_path = config.path("paths.baseline_path")
    baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    result.artifacts.append(str(baseline_path.resolve().relative_to(Path(config.root).resolve())))

    with Tracker(
        config,
        run_name="evaluate",
        tags={
            "stage": "evaluate",
            "split": split,
            "model_type": model.metadata.model_type,
            "checkpoint_id": model.metadata.checkpoint_id,
            "model_trained_at": model.metadata.trained_at,
        },
        use_mlflow=use_mlflow,
    ) as tracker:
        tracker.log_params({"threshold": threshold, "split": split})
        tracker.log_metrics({f"test_{key}": value for key, value in metrics.items()})
        # Registration resolves runs:/<run_id>/model against *this* run, because
        # this is the run whose test_accuracy the promotion gate reads.
        tracker.log_sklearn_model(model.scaler, model.estimator)
        for artifact in (confusion_png, roc_png, report_path, card_path, baseline_path):
            tracker.log_artifact(Path(artifact), artifact_subdir="evaluation")
        result.run_id = tracker.record.run_id

    _LOGGER.info("evaluation complete", extra={"split": split, **metrics})
    return result


__all__ = ["EvaluationResult", "build_model_card", "compute_metrics", "run"]
