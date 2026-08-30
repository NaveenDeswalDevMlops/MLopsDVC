"""Experiment tracking.

Two backends, one interface:

* **MLflow** is the tracker the brief asks for. It is used whenever the package is
  importable and ``tracking.backend`` is ``auto`` or ``mlflow``.
* **A local JSON run store** is written on *every* run regardless of backend.

The mirror is not redundancy for its own sake. The dashboard has to render the run
history, and a dashboard that shows nothing when MLflow is unreachable is a
dashboard nobody trusts. Writing a small, readable run record next to the artifacts
means the experiment view works from a bare clone, in CI, and inside a Kubernetes
pod that has no route to a tracking server — while MLflow still receives everything
when it is available.

Run records live in ``artifacts/runs/<run_id>/run.json``.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlops.config import Config
from mlops.logging_setup import get_logger

_LOGGER = get_logger(__name__)


def mlflow_available() -> bool:
    """Report whether the MLflow package can be imported."""
    try:
        import mlflow  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False
    return True


@dataclass
class RunRecord:
    """One tracked run."""

    run_id: str
    run_name: str
    experiment: str
    status: str = "RUNNING"
    started_at: str = ""
    ended_at: str = ""
    duration_seconds: float = 0.0
    params: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metric_history: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    backend: str = "file"
    mlflow_run_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of this record."""
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "experiment": self.experiment,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "params": self.params,
            "tags": self.tags,
            "metrics": self.metrics,
            "metric_history": self.metric_history,
            "artifacts": self.artifacts,
            "backend": self.backend,
            "mlflow_run_id": self.mlflow_run_id,
        }


class Tracker:
    """Records params, metrics and artifacts for a single run."""

    def __init__(
        self,
        config: Config,
        run_name: str,
        tags: dict[str, str] | None = None,
        use_mlflow: bool | None = None,
    ) -> None:
        """Initialise a run."""
        self.config = config
        backend_setting = str(config.get("tracking.backend", "auto")).lower()
        wanted = backend_setting in {"auto", "mlflow"} if use_mlflow is None else use_mlflow
        self._use_mlflow = bool(wanted) and mlflow_available()
        if wanted and not self._use_mlflow:
            _LOGGER.info("mlflow not importable; using the local run store only")

        self.record = RunRecord(
            run_id=f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}",
            run_name=run_name,
            experiment=str(config.get("tracking.experiment_name", "catsdogs")),
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            tags=dict(tags or {}),
            backend="mlflow+file" if self._use_mlflow else "file",
        )
        self.run_dir = config.path("paths.runs_dir") / self.record.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._start = time.perf_counter()
        self._mlflow_run: Any = None

    def __enter__(self) -> Tracker:
        """Begin the run."""
        if self._use_mlflow:
            try:
                import mlflow

                mlflow.set_tracking_uri(
                    str(self.config.get("tracking.mlflow_tracking_uri", "file:./mlruns"))
                )
                mlflow.set_experiment(self.record.experiment)
                self._mlflow_run = mlflow.start_run(run_name=self.record.run_name)
                self.record.mlflow_run_id = self._mlflow_run.info.run_id
                mlflow.set_tags(self.record.tags)
            except Exception as exc:  # noqa: BLE001 - tracking must never break training
                _LOGGER.warning("mlflow run could not start", extra={"error": str(exc)})
                self._use_mlflow = False
                self.record.backend = "file"
        self._flush()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Finish the run, recording success or failure."""
        self.record.status = "FAILED" if exc_type else "FINISHED"
        self.record.ended_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.record.duration_seconds = time.perf_counter() - self._start
        if self._use_mlflow:
            try:
                import mlflow

                mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
            except Exception as exc_inner:  # noqa: BLE001
                _LOGGER.warning("mlflow end_run failed", extra={"error": str(exc_inner)})
        self._flush()
        _LOGGER.info(
            "run recorded",
            extra={
                "run_id": self.record.run_id,
                "status": self.record.status,
                "backend": self.record.backend,
                "metrics": self.record.metrics,
            },
        )

    def log_params(self, params: dict[str, Any]) -> None:
        """Record run parameters."""
        clean = {str(key): str(value) for key, value in params.items()}
        self.record.params.update(clean)
        if self._use_mlflow:
            try:
                import mlflow

                items = list(clean.items())
                for start in range(0, len(items), 90):
                    mlflow.log_params(dict(items[start : start + 90]))
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("mlflow log_params failed", extra={"error": str(exc)})
        self._flush()

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Record one metric value."""
        self.log_metrics({key: value}, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record several metric values at once."""
        clean = {str(key): float(value) for key, value in metrics.items()}
        self.record.metrics.update(clean)
        if step is not None:
            for key, value in clean.items():
                self.record.metric_history.setdefault(key, []).append(
                    {"step": float(step), "value": value}
                )
        if self._use_mlflow:
            try:
                import mlflow

                mlflow.log_metrics(clean, step=step)
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("mlflow log_metrics failed", extra={"error": str(exc)})
        self._flush()

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Record run tags."""
        clean = {str(key): str(value) for key, value in tags.items()}
        self.record.tags.update(clean)
        if self._use_mlflow:
            try:
                import mlflow

                mlflow.set_tags(clean)
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("mlflow set_tags failed", extra={"error": str(exc)})
        self._flush()

    def log_sklearn_model(self, scaler: Any, estimator: Any) -> bool:
        """Log the fitted model to MLflow."""
        if not self._use_mlflow:
            return False
        try:
            import mlflow.sklearn
            from sklearn.pipeline import Pipeline

            mlflow.sklearn.log_model(
                Pipeline([("scaler", scaler), ("classifier", estimator)]),
                artifact_path="model",
            )
        except Exception as exc:  # noqa: BLE001 - tracking must never break training
            _LOGGER.warning("mlflow log_model failed", extra={"error": str(exc)})
            return False
        self.record.tags["mlflow_model_logged"] = "true"
        self._flush()
        _LOGGER.info("model logged to mlflow", extra={"run_id": self.record.run_id})
        return True

    def log_artifact(self, path: Path | str, artifact_subdir: str = "") -> None:
        """Attach a file to the run."""
        source = Path(path)
        if not source.is_file():
            _LOGGER.warning("artifact missing, not logged", extra={"path": str(source)})
            return
        target_dir = self.run_dir / "artifacts" / artifact_subdir if artifact_subdir else (
            self.run_dir / "artifacts"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_dir / source.name)
        relative = str((Path(artifact_subdir) / source.name) if artifact_subdir else source.name)
        if relative not in self.record.artifacts:
            self.record.artifacts.append(relative)
        if self._use_mlflow:
            try:
                import mlflow

                mlflow.log_artifact(str(source), artifact_path=artifact_subdir or None)
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("mlflow log_artifact failed", extra={"error": str(exc)})
        self._flush()

    def _flush(self) -> None:
        """Write the run record to disk."""
        (self.run_dir / "run.json").write_text(
            json.dumps(self.record.to_dict(), indent=2), encoding="utf-8"
        )


def list_runs(config: Config, limit: int = 50) -> list[dict[str, Any]]:
    """Return recorded runs, newest first."""
    runs_dir = config.path("paths.runs_dir")
    if not runs_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for run_file in runs_dir.glob("*/run.json"):
        try:
            records.append(json.loads(run_file.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    records.sort(key=lambda item: str(item.get("started_at", "")), reverse=True)
    return records[:limit]


def get_run(config: Config, run_id: str) -> dict[str, Any] | None:
    """Return a single run record."""
    run_file = config.path("paths.runs_dir") / run_id / "run.json"
    if not run_file.is_file():
        return None
    try:
        return json.loads(run_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _checkpoint_identity(config: Config) -> str:
    """Return the unique id of the checkpoint currently on disk."""
    try:
        from mlops.models.model import CatsDogsModel

        return CatsDogsModel.load(config.path("paths.model_path")).metadata.checkpoint_id
    except Exception:  # noqa: BLE001 - a missing or unreadable model is just "none"
        return ""


def register_best_model(config: Config) -> dict[str, Any]:
    """Promote the best finished run whose weights are the ones on disk."""
    threshold = float(config.get("tracking.promote_min_accuracy", 0.75))
    candidates = [
        run
        for run in list_runs(config, limit=200)
        if run.get("status") == "FINISHED" and "test_accuracy" in run.get("metrics", {})
    ]
    checkpoint_identity = _checkpoint_identity(config)
    decision: dict[str, Any] = {
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threshold": threshold,
        "registered_model_name": str(
            config.get("tracking.registered_model_name", "catsdogs-classifier")
        ),
        "promoted": False,
        "reason": "no finished run with a test_accuracy metric",
        "run_id": "",
        "accuracy": None,
        "checkpoint_id": checkpoint_identity,
        "checkpoint_matches_run": False,
        "mlflow_registered": False,
    }
    if candidates:
        matching_candidates = [
            run
            for run in candidates
            if str(run.get("tags", {}).get("checkpoint_id", "")) == checkpoint_identity
        ]

        if not matching_candidates:
            decision["reason"] = (
                "no finished test run matches the checkpoint currently on disk; "
                "re-run training and evaluation so the artifact and score describe "
                "the same model"
            )
            return decision

        best = max(
            matching_candidates,
            key=lambda run: float(run["metrics"]["test_accuracy"]),
        )
        accuracy = float(best["metrics"]["test_accuracy"])
        run_identity = str(best.get("tags", {}).get("checkpoint_id", ""))
        matches = bool(run_identity) and run_identity == checkpoint_identity
        clears_threshold = accuracy >= threshold

        if not clears_threshold:
            reason = f"accuracy {accuracy:.4f} < threshold {threshold:.4f}"
        elif not matches:
            reason = (
                f"accuracy {accuracy:.4f} clears the threshold, but that run's checkpoint "
                f"({run_identity or 'unknown'}) is not the one on disk "
                f"({checkpoint_identity or 'none'}); re-run training and evaluation so the "
                "artifact and the score describe the same model"
            )
        else:
            reason = f"accuracy {accuracy:.4f} >= threshold {threshold:.4f}"

        decision.update(
            {
                "run_id": best["run_id"],
                "accuracy": accuracy,
                "run_checkpoint_id": run_identity,
                "checkpoint_matches_run": matches,
                "promoted": clears_threshold and matches,
                "reason": reason,
            }
        )
        if decision["promoted"] and not best.get("tags", {}).get("mlflow_model_logged"):
            decision["mlflow_note"] = (
                "the winning run has no MLflow model artifact, so there is nothing to "
                "register; this run was recorded by the local store only"
            )
        elif decision["promoted"]:
            decision["mlflow_note"] = "winning run contains an MLflow model artifact"
    return decision
