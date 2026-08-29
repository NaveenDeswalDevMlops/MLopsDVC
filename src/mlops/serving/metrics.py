"""Metrics in the Prometheus text exposition format.

The brief allows "logs, Prometheus, or simple in-app counters". This module is both
of the last two: the counters are in-process and dependency-free, and they render
as valid Prometheus exposition text, so a real Prometheus server scrapes ``/metrics``
without any adapter. That keeps the container small and keeps the dashboard able to
read the same numbers by parsing the same endpoint.

Everything is guarded by a lock because a Flask app served by more than one thread
will increment these concurrently.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

LabelKey = tuple[tuple[str, str], ...]


def _labels_to_key(labels: dict[str, str] | None) -> LabelKey:
    """Normalise a label mapping into a hashable key.

    Args:
        labels: Label mapping or ``None``.

    Returns:
        A sorted tuple of label pairs.
    """
    return tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))


def _render_labels(key: LabelKey, extra: dict[str, str] | None = None) -> str:
    """Render a label key as Prometheus label syntax.

    Args:
        key: Normalised label key.
        extra: Additional labels to merge in.

    Returns:
        A string such as ``{label="value"}`` or an empty string.
    """
    pairs = dict(key)
    pairs.update(extra or {})
    if not pairs:
        return ""
    inner = ",".join(
        f'{name}="{str(value).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for name, value in sorted(pairs.items())
    )
    return "{" + inner + "}"


@dataclass
class Counter:
    """A monotonically increasing counter.

    Attributes:
        name: Metric name.
        help_text: HELP line shown in the exposition output.
        values: Per-label-set totals.
    """

    name: str
    help_text: str
    values: dict[LabelKey, float] = field(default_factory=dict)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        """Increment the counter.

        Args:
            amount: Amount to add.
            **labels: Label values for this series.
        """
        key = _labels_to_key(labels)
        self.values[key] = self.values.get(key, 0.0) + amount

    def total(self) -> float:
        """Return the sum across all label sets."""
        return float(sum(self.values.values()))

    def render(self) -> list[str]:
        """Render this metric as exposition lines."""
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} counter"]
        if not self.values:
            lines.append(f"{self.name} 0")
        for key, value in sorted(self.values.items()):
            lines.append(f"{self.name}{_render_labels(key)} {value:g}")
        return lines


@dataclass
class Gauge:
    """A value that can go up or down.

    Attributes:
        name: Metric name.
        help_text: HELP line shown in the exposition output.
        values: Per-label-set values.
    """

    name: str
    help_text: str
    values: dict[LabelKey, float] = field(default_factory=dict)

    def set(self, value: float, **labels: str) -> None:
        """Set the gauge value.

        Args:
            value: New value.
            **labels: Label values for this series.
        """
        self.values[_labels_to_key(labels)] = float(value)

    def render(self) -> list[str]:
        """Render this metric as exposition lines."""
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} gauge"]
        if not self.values:
            lines.append(f"{self.name} 0")
        for key, value in sorted(self.values.items()):
            lines.append(f"{self.name}{_render_labels(key)} {value:g}")
        return lines


@dataclass
class Histogram:
    """A cumulative histogram with explicit buckets.

    Attributes:
        name: Metric name.
        help_text: HELP line shown in the exposition output.
        buckets: Upper bounds, ascending.
        counts: Per-label-set bucket counts.
        sums: Per-label-set observation sums.
        totals: Per-label-set observation counts.
        recent: Recent raw observations, used for live percentiles in the UI.
    """

    name: str
    help_text: str
    buckets: tuple[float, ...]
    counts: dict[LabelKey, list[float]] = field(default_factory=dict)
    sums: dict[LabelKey, float] = field(default_factory=dict)
    totals: dict[LabelKey, float] = field(default_factory=dict)
    recent: list[float] = field(default_factory=list)

    def observe(self, value: float, **labels: str) -> None:
        """Record one observation.

        Args:
            value: The observed value.
            **labels: Label values for this series.
        """
        key = _labels_to_key(labels)
        bucket_counts = self.counts.setdefault(key, [0.0] * len(self.buckets))
        for index, bound in enumerate(self.buckets):
            if value <= bound:
                bucket_counts[index] += 1.0
        self.sums[key] = self.sums.get(key, 0.0) + float(value)
        self.totals[key] = self.totals.get(key, 0.0) + 1.0
        self.recent.append(float(value))
        if len(self.recent) > 1000:
            del self.recent[: len(self.recent) - 1000]

    def quantile(self, fraction: float) -> float:
        """Return an approximate quantile of the recent observations.

        Args:
            fraction: Quantile in ``[0, 1]``.

        Returns:
            The observation at that quantile, or ``0.0`` when there is no data.
        """
        if not self.recent:
            return 0.0
        ordered = sorted(self.recent)
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[index]

    def mean(self) -> float:
        """Return the mean of all observations, or ``0.0`` when there are none."""
        total = sum(self.totals.values())
        return float(sum(self.sums.values()) / total) if total else 0.0

    def render(self) -> list[str]:
        """Render this metric as exposition lines."""
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} histogram"]
        if not self.totals:
            for bound in self.buckets:
                lines.append(f'{self.name}_bucket{{le="{bound:g}"}} 0')
            lines.append(f'{self.name}_bucket{{le="+Inf"}} 0')
            lines.append(f"{self.name}_sum 0")
            lines.append(f"{self.name}_count 0")
            return lines
        for key in sorted(self.totals):
            bucket_counts = self.counts.get(key, [0.0] * len(self.buckets))
            for bound, count in zip(self.buckets, bucket_counts):
                lines.append(
                    f"{self.name}_bucket{_render_labels(key, {'le': f'{bound:g}'})} {count:g}"
                )
            lines.append(
                f"{self.name}_bucket{_render_labels(key, {'le': '+Inf'})} {self.totals[key]:g}"
            )
            lines.append(f"{self.name}_sum{_render_labels(key)} {self.sums.get(key, 0.0):g}")
            lines.append(f"{self.name}_count{_render_labels(key)} {self.totals[key]:g}")
        return lines


class MetricsRegistry:
    """The service's metric set.

    Attributes:
        requests: Predictions by predicted label and status.
        http_requests: HTTP requests by method, path and status code.
        latency: Inference latency in seconds, by endpoint.
        confidence: Predicted-class confidence, by label.
        errors: Failures by error code.
        model_info: Identity of the loaded model, carried as labels.
        ready: ``1`` when a checkpoint is loaded.
        started_at: Process start timestamp.
    """

    def __init__(self, buckets: Iterable[float] | None = None) -> None:
        """Create the registry.

        Args:
            buckets: Latency histogram bounds in seconds.
        """
        bounds = tuple(sorted(float(b) for b in (buckets or (0.005, 0.01, 0.05, 0.1, 0.5, 1.0))))
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.requests = Counter(
            "catsdogs_predictions_total", "Predictions served, by label and status."
        )
        self.http_requests = Counter(
            "catsdogs_http_requests_total", "HTTP requests, by method, path and status."
        )
        self.latency = Histogram(
            "catsdogs_inference_latency_seconds", "Inference latency in seconds.", bounds
        )
        self.confidence = Histogram(
            "catsdogs_prediction_confidence",
            "Confidence of the predicted class.",
            (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
        )
        self.errors = Counter("catsdogs_errors_total", "Errors, by error code.")
        self.model_info = Gauge("catsdogs_model_info", "Loaded model identity, as labels.")
        self.ready = Gauge("catsdogs_model_ready", "1 when a checkpoint is loaded.")
        self.ready.set(0.0)

    def record_prediction(self, label: str, confidence: float, latency_seconds: float, endpoint: str) -> None:
        """Record a successful prediction.

        Args:
            label: Predicted class name.
            confidence: Confidence of the predicted class.
            latency_seconds: Inference latency.
            endpoint: Endpoint that served it.
        """
        with self._lock:
            self.requests.inc(label=label, status="success")
            self.latency.observe(latency_seconds, endpoint=endpoint)
            self.confidence.observe(confidence, label=label)

    def record_error(self, code: str) -> None:
        """Record a failed request.

        Args:
            code: Stable error code.
        """
        with self._lock:
            self.errors.inc(error_code=code)
            self.requests.inc(label="none", status="error")

    def record_http(self, method: str, path: str, status: int) -> None:
        """Record an HTTP exchange.

        Args:
            method: HTTP method.
            path: Request path.
            status: Response status code.
        """
        with self._lock:
            self.http_requests.inc(method=method, path=path, status=str(status))

    def set_model(self, version: str, model_type: str, ready: bool) -> None:
        """Publish the identity and readiness of the loaded model.

        Args:
            version: Model version string.
            model_type: Model architecture name.
            ready: Whether a checkpoint is loaded.
        """
        with self._lock:
            self.model_info.values.clear()
            self.model_info.set(1.0, version=version, model_type=model_type)
            self.ready.set(1.0 if ready else 0.0)

    def uptime_seconds(self) -> float:
        """Return process uptime in seconds."""
        return max(0.0, time.time() - self.started_at)

    def render(self) -> str:
        """Render every metric in Prometheus exposition format.

        Returns:
            The exposition text, newline-terminated.
        """
        with self._lock:
            lines: list[str] = []
            for metric in (
                self.requests,
                self.http_requests,
                self.errors,
                self.latency,
                self.confidence,
                self.model_info,
                self.ready,
            ):
                lines.extend(metric.render())
            lines.extend(
                [
                    "# HELP catsdogs_uptime_seconds Seconds since the service started.",
                    "# TYPE catsdogs_uptime_seconds gauge",
                    f"catsdogs_uptime_seconds {self.uptime_seconds():g}",
                ]
            )
            return "\n".join(lines) + "\n"

    def snapshot(self) -> dict[str, Any]:
        """Summarise the metrics for the dashboard.

        Returns:
            A compact JSON-serialisable summary.
        """
        with self._lock:
            def collapse(counter: Counter, label: str, only: dict[str, str] | None = None) -> dict[str, float]:
                """Sum a counter's series down to one label.

                Several series can share the same value for the label being kept —
                ``/predict`` and ``/predict/batch`` both produce status 200 — so the
                values must be added, not overwritten.

                Args:
                    counter: The counter to collapse.
                    label: The label to group by.
                    only: Optional label filters that a series must match.

                Returns:
                    Mapping of label value to summed count.
                """
                grouped: dict[str, float] = {}
                for key, value in counter.values.items():
                    labels = dict(key)
                    if only and any(labels.get(name) != want for name, want in only.items()):
                        continue
                    name = labels.get(label, "?")
                    grouped[name] = grouped.get(name, 0.0) + value
                return grouped

            by_label = collapse(self.requests, "label", {"status": "success"})
            return {
                "predictions_total": sum(by_label.values()),
                "predictions_by_label": by_label,
                "errors_total": self.errors.total(),
                "errors_by_code": collapse(self.errors, "error_code"),
                "http_requests_total": self.http_requests.total(),
                "http_by_status": collapse(self.http_requests, "status"),
                "latency_ms": {
                    "mean": round(self.latency.mean() * 1000, 3),
                    "p50": round(self.latency.quantile(0.50) * 1000, 3),
                    "p90": round(self.latency.quantile(0.90) * 1000, 3),
                    "p99": round(self.latency.quantile(0.99) * 1000, 3),
                },
                "confidence_mean": round(self.confidence.mean(), 4),
                "model_ready": bool(next(iter(self.ready.values.values()), 0.0)),
                "uptime_seconds": round(self.uptime_seconds(), 1),
            }


def parse_exposition(text: str) -> dict[str, float]:
    """Parse Prometheus exposition text into a flat mapping.

    Used by the dashboard, which reads the API's ``/metrics`` over HTTP rather than
    reaching into the API process.

    Args:
        text: Exposition text.

    Returns:
        Mapping of full series name (including labels) to value.
    """
    series: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        if not name:
            continue
        try:
            series[name.strip()] = float(value)
        except ValueError:
            continue
    return series


__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "parse_exposition",
]
