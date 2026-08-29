"""Post-deployment model performance tracking.

Offline test metrics tell you the artifact was good when it was trained. This
module answers the more useful question: is the model that is *serving traffic
right now* still that good?

It draws a stratified, seeded, labelled batch from the held-out test split, sends
every image to the live HTTP endpoint exactly as a client would, scores the
responses against the known labels, compares the result to
``artifacts/metrics/baseline.json`` and returns PASS or FAIL against two gates —
an absolute accuracy floor and a maximum drop from the baseline.

Going over HTTP rather than calling the model in-process is the whole point. It
exercises the deployed container, its preprocessing, its threshold and its
checkpoint together, so it catches the failures an offline metric cannot: a stale
image tag, a checkpoint that never got baked in, a normalisation mismatch.

The HTTP client is injected, which is what lets the test suite point the same code
at a Flask test client and get real coverage without a network.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from mlops.config import Config
from mlops.data.preprocess import ManifestRow, read_manifest
from mlops.logging_setup import get_logger
from mlops.models.evaluate import compute_metrics
from mlops.tracking.tracker import Tracker

_LOGGER = get_logger(__name__)


class HttpResponse(Protocol):
    """Minimal response shape shared by ``requests`` and Flask's test client."""

    status_code: int

    def json(self) -> Any:
        """Return the parsed JSON body."""


class PerfCheckError(RuntimeError):
    """Raised when the check cannot be carried out at all."""


def _json_body(response: Any) -> Any:
    """Read the JSON body from either client library's response object.

    ``requests`` exposes ``.json()`` as a method; Werkzeug's test client exposes
    ``.json`` as a property. Supporting both is what lets the test suite point this
    checker at a Flask test client and exercise the real code path without a
    socket.

    Args:
        response: A response from either library.

    Returns:
        The parsed body.
    """
    payload = getattr(response, "json", None)
    return payload() if callable(payload) else payload


@dataclass
class PerfCheckResult:
    """Outcome of one post-deployment check.

    Attributes:
        checked_at: ISO-8601 UTC timestamp.
        endpoint: Base URL that was exercised.
        sample_size: Requests actually completed.
        requested: Requests attempted.
        metrics: Live metrics computed from the responses.
        baseline_metrics: Metrics recorded at training time.
        deltas: ``live - baseline`` per metric.
        confusion: Live confusion matrix.
        latency_ms: Latency summary across the batch.
        failures: Requests that did not return a usable prediction.
        gates: Individual gate results.
        passed: Overall verdict.
        model_version: Version reported by the endpoint.
    """

    checked_at: str
    endpoint: str
    sample_size: int
    requested: int
    metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    deltas: dict[str, float] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    gates: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = False
    model_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of this result."""
        return {
            "checked_at": self.checked_at,
            "endpoint": self.endpoint,
            "sample_size": self.sample_size,
            "requested": self.requested,
            "metrics": self.metrics,
            "baseline_metrics": self.baseline_metrics,
            "deltas": self.deltas,
            "confusion_matrix": self.confusion,
            "latency_ms": self.latency_ms,
            "failures": self.failures,
            "gates": self.gates,
            "passed": self.passed,
            "model_version": self.model_version,
        }


def stratified_sample(
    rows: list[ManifestRow], size: int, seed: int
) -> list[ManifestRow]:
    """Draw a class-balanced sample deterministically.

    Args:
        rows: Candidate manifest rows.
        size: Desired sample size.
        seed: Seed so the same batch is drawn every run.

    Returns:
        The sampled rows, shuffled.

    Raises:
        PerfCheckError: If there are no rows to sample from.
    """
    if not rows:
        raise PerfCheckError("no test images available; run preprocessing first")
    by_class: dict[str, list[ManifestRow]] = {}
    for row in rows:
        by_class.setdefault(row.class_name, []).append(row)

    rng = random.Random(seed)
    per_class = max(1, size // max(1, len(by_class)))
    sample: list[ManifestRow] = []
    for class_name in sorted(by_class):
        candidates = sorted(by_class[class_name], key=lambda item: item.path)
        rng.shuffle(candidates)
        sample.extend(candidates[:per_class])
    rng.shuffle(sample)
    return sample[:size]


def default_client(base_url: str, timeout: float = 15.0) -> Callable[[bytes, str], HttpResponse]:
    """Build an HTTP client that posts an image to ``/predict``.

    Args:
        base_url: Base URL of the running service.
        timeout: Per-request timeout in seconds.

    Returns:
        A callable taking image bytes and a filename, returning a response.
    """
    import requests

    def send(payload: bytes, filename: str) -> HttpResponse:
        return requests.post(
            f"{base_url.rstrip('/')}/predict",
            files={"file": (filename, payload, "image/jpeg")},
            timeout=timeout,
        )

    return send


def load_baseline(config: Config) -> dict[str, Any]:
    """Read the training-time baseline.

    Args:
        config: Effective configuration.

    Returns:
        The baseline document, or an empty mapping when it has not been written.
    """
    path = config.path("paths.baseline_path")
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run(
    config: Config,
    endpoint_url: str | None = None,
    sample_size: int | None = None,
    client: Callable[[bytes, str], HttpResponse] | None = None,
    use_mlflow: bool | None = None,
) -> PerfCheckResult:
    """Run the post-deployment check against a live endpoint.

    Args:
        config: Effective configuration.
        endpoint_url: Base URL to exercise; defaults to ``ui.api_url``.
        sample_size: Number of labelled images to send.
        client: Injected HTTP client, for tests.
        use_mlflow: Force MLflow tracking on or off.

    Returns:
        The check result, also written to ``paths.perf_report``.

    Raises:
        PerfCheckError: If the manifest is missing or every request failed.
    """
    base_url = endpoint_url or str(config.get("ui.api_url", "http://127.0.0.1:8000"))
    size = int(sample_size or config.get("monitoring.perf_check.sample_size", 40))
    seed = int(config.get("project.seed", 42))
    send = client or default_client(base_url)

    manifest = read_manifest(config.path("paths.manifest_csv"))
    test_rows = [row for row in manifest if row.split == "test"]
    sample = stratified_sample(test_rows, size, seed)

    truths: list[int] = []
    scores: list[float] = []
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []
    model_version = ""

    for row in sample:
        path = Path(row.path)
        if not path.is_absolute():
            path = config.root / path
        payload = path.read_bytes()
        started = time.perf_counter()
        try:
            response = send(payload, path.name)
        except Exception as exc:  # noqa: BLE001 - a transport failure is a data point
            failures.append({"path": row.path, "error": str(exc)})
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        status = getattr(response, "status_code", 0)
        if status != 200:
            failures.append({"path": row.path, "status": status})
            continue
        try:
            body = _json_body(response)
        except Exception as exc:  # noqa: BLE001
            failures.append({"path": row.path, "error": f"unparseable body: {exc}"})
            continue

        probabilities = body.get("probabilities") or {}
        positive_class = body.get("classes", ["cat", "dog"])[-1] if body.get("classes") else "dog"
        score = probabilities.get(positive_class)
        if score is None and probabilities:
            score = list(probabilities.values())[-1]
        if score is None:
            failures.append({"path": row.path, "error": "response carried no probabilities"})
            continue

        truths.append(int(row.label))
        scores.append(float(score))
        latencies.append(elapsed_ms)
        model_version = body.get("model_version", model_version)

    if not truths:
        raise PerfCheckError(
            f"every one of the {len(sample)} requests to {base_url} failed; "
            "is the service running and a model loaded?"
        )

    baseline = load_baseline(config)
    threshold = float(baseline.get("threshold", config.get("evaluation.threshold", 0.5)))
    labels = np.asarray(truths, dtype=np.int64)
    probabilities_array = np.asarray(scores, dtype=np.float64)
    metrics = compute_metrics(labels, probabilities_array, threshold)

    predictions = (probabilities_array >= threshold).astype(int)
    confusion = [[0, 0], [0, 0]]
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        confusion[truth][prediction] += 1

    baseline_metrics = {
        key: float(value)
        for key, value in (baseline.get("metrics") or {}).items()
        if isinstance(value, (int, float))
    }
    deltas = {
        key: round(metrics[key] - baseline_metrics[key], 6)
        for key in metrics
        if key in baseline_metrics
    }

    min_accuracy = float(config.get("monitoring.perf_check.min_accuracy", 0.70))
    max_drop = float(config.get("monitoring.perf_check.max_accuracy_drop", 0.10))
    accuracy_drop = baseline_metrics.get("accuracy", metrics["accuracy"]) - metrics["accuracy"]

    gates = [
        {
            "name": "absolute_accuracy",
            "detail": f"live accuracy {metrics['accuracy']:.4f} >= floor {min_accuracy:.4f}",
            "passed": metrics["accuracy"] >= min_accuracy,
        },
        {
            "name": "accuracy_drop_vs_baseline",
            "detail": f"drop {accuracy_drop:.4f} <= allowed {max_drop:.4f}",
            "passed": accuracy_drop <= max_drop,
        },
        {
            "name": "request_success",
            "detail": f"{len(truths)}/{len(sample)} requests returned a prediction",
            "passed": not failures,
        },
    ]

    result = PerfCheckResult(
        checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        endpoint=base_url,
        sample_size=len(truths),
        requested=len(sample),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        deltas=deltas,
        confusion=confusion,
        latency_ms={
            "mean": round(float(np.mean(latencies)), 3),
            "p50": round(float(np.percentile(latencies, 50)), 3),
            "p95": round(float(np.percentile(latencies, 95)), 3),
            "max": round(float(np.max(latencies)), 3),
        },
        failures=failures,
        gates=gates,
        passed=all(gate["passed"] for gate in gates),
        model_version=str(model_version),
    )

    report_path = config.path("paths.perf_report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    with Tracker(
        config,
        run_name="perf-check",
        tags={"stage": "monitoring", "endpoint": base_url, "verdict": "PASS" if result.passed else "FAIL"},
        use_mlflow=use_mlflow,
    ) as tracker:
        tracker.log_params({"sample_size": len(truths), "endpoint": base_url, "threshold": threshold})
        tracker.log_metrics({f"live_{key}": value for key, value in metrics.items()})
        tracker.log_metrics({f"delta_{key}": value for key, value in deltas.items()})
        tracker.log_metrics({"live_latency_ms_p95": result.latency_ms["p95"]})
        tracker.log_artifact(report_path, artifact_subdir="monitoring")

    _LOGGER.info(
        "performance check complete",
        extra={
            "passed": result.passed,
            "accuracy": round(metrics["accuracy"], 4),
            "baseline_accuracy": baseline_metrics.get("accuracy"),
            "failures": len(failures),
        },
    )
    return result


def load_report(config: Config) -> dict[str, Any] | None:
    """Read the most recent performance report.

    Args:
        config: Effective configuration.

    Returns:
        The report, or ``None`` when no check has been run.
    """
    path = config.path("paths.perf_report")
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


__all__ = [
    "PerfCheckError",
    "PerfCheckResult",
    "default_client",
    "load_baseline",
    "load_report",
    "run",
    "stratified_sample",
]
