"""Collect logs from wherever the service happens to be running.

Three sources, one shape of output, chosen by where the dashboard itself is:

* **local** — the JSONL file the service writes. Used during development.
* **kubectl** — ``kubectl logs`` against the minikube cluster. Used when the
  dashboard runs on the developer's machine and the service runs in minikube.
* **in-cluster** — the Kubernetes API over HTTPS using the pod's own service
  account token. Used when the dashboard itself is a pod, where no ``kubectl``
  binary exists.

The in-cluster path matters because the alternative — baking ``kubectl`` into the
image — adds a 50 MB binary and a second authentication path to keep working. The
service account token and CA certificate are already mounted in every pod, so the
API call needs nothing extra beyond the RBAC Role in ``k8s/rbac.yaml``.

Every collected line is normalised into the same record shape the local JSONL uses,
so the monitoring view renders one merged stream regardless of source.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlops.config import Config
from mlops.logging_setup import get_logger, read_jsonl_tail

_LOGGER = get_logger(__name__)

SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
KUBECTL_TIMEOUT = 60


@dataclass
class LogBundle:
    """Logs collected from one or more sources.

    Attributes:
        source: Which collector produced the records.
        collected_at: ISO-8601 UTC timestamp.
        records: Normalised log records, oldest first.
        pods: Pod names the records came from, when applicable.
        available: Whether the source could be reached at all.
        message: Human-readable explanation, especially on failure.
    """

    source: str
    collected_at: str
    records: list[dict[str, Any]] = field(default_factory=list)
    pods: list[str] = field(default_factory=list)
    available: bool = True
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of this bundle."""
        return {
            "source": self.source,
            "collected_at": self.collected_at,
            "records": self.records,
            "pods": self.pods,
            "available": self.available,
            "message": self.message,
            "count": len(self.records),
        }


def _now() -> str:
    """Return the current UTC timestamp in ISO-8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalise(line: str, pod: str, source: str) -> dict[str, Any]:
    """Turn one raw log line into the standard record shape.

    Args:
        line: The raw line, which may or may not be JSON.
        pod: Pod the line came from, if known.
        source: Collector name.

    Returns:
        A normalised record.
    """
    line = line.strip()
    if not line:
        return {}
    record: dict[str, Any]
    try:
        parsed = json.loads(line)
        record = parsed if isinstance(parsed, dict) else {"message": str(parsed)}
    except json.JSONDecodeError:
        record = {"level": "RAW", "logger": "-", "message": line, "ts": ""}
    record.setdefault("ts", "")
    record.setdefault("level", "INFO")
    record.setdefault("logger", "-")
    record["pod"] = pod
    record["source"] = source
    return record


def in_cluster() -> bool:
    """Report whether this process is running inside a Kubernetes pod.

    Returns:
        ``True`` when the service account token is mounted.
    """
    return (SERVICE_ACCOUNT_DIR / "token").is_file()


def kubectl_available() -> bool:
    """Report whether a ``kubectl`` binary is on PATH.

    Returns:
        ``True`` when kubectl can be executed.
    """
    return shutil.which("kubectl") is not None


def collect_local(config: Config, limit: int | None = None) -> LogBundle:
    """Read the service's own JSONL log file.

    Args:
        config: Effective configuration.
        limit: Maximum records to return.

    Returns:
        The collected bundle.
    """
    tail = int(limit or config.get("monitoring.log_tail_lines", 300))
    records: list[dict[str, Any]] = []
    for key, label in (("monitoring.log_file", "api"), ("monitoring.ui_log_file", "ui")):
        try:
            path = config.path(key)
        except KeyError:
            continue
        for record in read_jsonl_tail(path, tail):
            record["pod"] = label
            record["source"] = "local"
            records.append(record)
    records.sort(key=lambda item: str(item.get("ts", "")))
    return LogBundle(
        source="local",
        collected_at=_now(),
        records=records[-tail:],
        pods=sorted({str(record.get("pod", "-")) for record in records}),
        available=True,
        message=f"{len(records)} records from the local log files",
    )


def _run_kubectl(config: Config, args: list[str]) -> subprocess.CompletedProcess[str] | None:
    """Run a kubectl subcommand.

    Args:
        config: Effective configuration.
        args: Arguments after the kubectl binary.

    Returns:
        The completed process, or ``None`` when kubectl is unavailable or failed
        to launch.
    """
    binary = str(config.get("monitoring.kubernetes.kubectl_binary", "kubectl"))
    context = str(config.get("monitoring.kubernetes.context", "") or "")
    command = [binary]
    if context:
        command += ["--context", context]
    command += args
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=KUBECTL_TIMEOUT, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _LOGGER.warning("kubectl failed to run", extra={"error": str(exc)})
        return None


def list_pods_kubectl(config: Config) -> list[dict[str, Any]]:
    """List the deployment's pods via kubectl.

    Args:
        config: Effective configuration.

    Returns:
        Pod summaries; empty when the cluster cannot be reached.
    """
    namespace = str(config.get("monitoring.kubernetes.namespace", "mlops"))
    selector = str(config.get("monitoring.kubernetes.label_selector", ""))
    args = ["get", "pods", "-n", namespace, "-o", "json"]
    if selector:
        args += ["-l", selector]
    completed = _run_kubectl(config, args)
    if completed is None or completed.returncode != 0:
        return []
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return []
    return [_pod_summary(item) for item in document.get("items", [])]


def _pod_summary(item: dict[str, Any]) -> dict[str, Any]:
    """Reduce a Kubernetes pod object to the fields the dashboard shows.

    Args:
        item: A pod object from the API.

    Returns:
        Name, phase, readiness, restart count, node and start time.
    """
    status = item.get("status", {})
    containers = status.get("containerStatuses", []) or []
    return {
        "name": item.get("metadata", {}).get("name", "?"),
        "phase": status.get("phase", "?"),
        "ready": all(container.get("ready") for container in containers) if containers else False,
        "restarts": sum(int(container.get("restartCount", 0)) for container in containers),
        "node": item.get("spec", {}).get("nodeName", ""),
        "started_at": status.get("startTime", ""),
        "image": containers[0].get("image", "") if containers else "",
    }


def collect_kubectl(config: Config, tail: int | None = None) -> LogBundle:
    """Collect pod logs from minikube using kubectl.

    Args:
        config: Effective configuration.
        tail: Lines per pod.

    Returns:
        The collected bundle. A missing binary or unreachable cluster is reported
        in ``message`` rather than raised, so the dashboard stays usable.
    """
    namespace = str(config.get("monitoring.kubernetes.namespace", "mlops"))
    lines = int(tail or config.get("monitoring.kubernetes.tail_lines", 200))

    if not kubectl_available():
        return LogBundle(
            source="kubectl",
            collected_at=_now(),
            available=False,
            message="kubectl is not on PATH. Install it, or deploy the dashboard "
            "into the cluster where the in-cluster collector is used instead.",
        )

    pods = list_pods_kubectl(config)
    if not pods:
        return LogBundle(
            source="kubectl",
            collected_at=_now(),
            available=False,
            message=f"no pods found in namespace {namespace!r}. Run `make k8s-deploy` "
            "and check that minikube is running.",
        )

    records: list[dict[str, Any]] = []
    for pod in pods:
        completed = _run_kubectl(
            config,
            ["logs", pod["name"], "-n", namespace, f"--tail={lines}", "--all-containers=true"],
        )
        if completed is None or completed.returncode != 0:
            detail = completed.stderr.strip() if completed else "kubectl could not be executed"
            records.append(
                {
                    "ts": _now(),
                    "level": "WARNING",
                    "logger": "log_collector",
                    "message": f"could not read logs for {pod['name']}: {detail}",
                    "pod": pod["name"],
                    "source": "kubectl",
                }
            )
            continue
        for line in completed.stdout.splitlines():
            record = _normalise(line, pod["name"], "kubectl")
            if record:
                records.append(record)

    records.sort(key=lambda item: str(item.get("ts", "")))
    return LogBundle(
        source="kubectl",
        collected_at=_now(),
        records=records,
        pods=[pod["name"] for pod in pods],
        available=True,
        message=f"{len(records)} lines from {len(pods)} pod(s) in namespace {namespace!r}",
    )


def _api_request(path: str, params: dict[str, str] | None = None) -> tuple[int, str]:
    """Call the Kubernetes API using the pod's own service account.

    Args:
        path: API path beginning with ``/api``.
        params: Optional query parameters.

    Returns:
        Tuple of HTTP status and response text. Status ``0`` means the call could
        not be made at all.
    """
    import os

    import requests

    token_file = SERVICE_ACCOUNT_DIR / "token"
    ca_file = SERVICE_ACCOUNT_DIR / "ca.crt"
    if not token_file.is_file():
        return 0, "no service account token is mounted"

    host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
    port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    url = f"https://{host}:{port}{path}"
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token_file.read_text().strip()}"},
            verify=str(ca_file) if ca_file.is_file() else False,
            params=params or {},
            timeout=20,
        )
    except Exception as exc:  # noqa: BLE001 - any transport failure is reportable
        return 0, str(exc)
    return response.status_code, response.text


def collect_in_cluster(config: Config, tail: int | None = None) -> LogBundle:
    """Collect pod logs through the Kubernetes API from inside the cluster.

    Args:
        config: Effective configuration.
        tail: Lines per pod.

    Returns:
        The collected bundle.
    """
    namespace = str(config.get("monitoring.kubernetes.namespace", "mlops"))
    selector = str(config.get("monitoring.kubernetes.label_selector", ""))
    lines = int(tail or config.get("monitoring.kubernetes.tail_lines", 200))

    if not in_cluster():
        return LogBundle(
            source="in-cluster",
            collected_at=_now(),
            available=False,
            message="not running inside a Kubernetes pod",
        )

    status, body = _api_request(
        f"/api/v1/namespaces/{namespace}/pods",
        {"labelSelector": selector} if selector else None,
    )
    if status != 200:
        return LogBundle(
            source="in-cluster",
            collected_at=_now(),
            available=False,
            message=f"pod list failed with status {status}: {body[:400]}. "
            "Check that k8s/rbac.yaml is applied.",
        )
    try:
        pods = [_pod_summary(item) for item in json.loads(body).get("items", [])]
    except json.JSONDecodeError as exc:
        return LogBundle(
            source="in-cluster",
            collected_at=_now(),
            available=False,
            message=f"pod list was not JSON: {exc}",
        )

    records: list[dict[str, Any]] = []
    for pod in pods:
        log_status, log_body = _api_request(
            f"/api/v1/namespaces/{namespace}/pods/{pod['name']}/log",
            {"tailLines": str(lines)},
        )
        if log_status != 200:
            records.append(
                {
                    "ts": _now(),
                    "level": "WARNING",
                    "logger": "log_collector",
                    "message": f"log read for {pod['name']} returned {log_status}",
                    "pod": pod["name"],
                    "source": "in-cluster",
                }
            )
            continue
        for line in log_body.splitlines():
            record = _normalise(line, pod["name"], "in-cluster")
            if record:
                records.append(record)

    records.sort(key=lambda item: str(item.get("ts", "")))
    return LogBundle(
        source="in-cluster",
        collected_at=_now(),
        records=records,
        pods=[pod["name"] for pod in pods],
        available=True,
        message=f"{len(records)} lines from {len(pods)} pod(s) via the Kubernetes API",
    )


def list_pods(config: Config) -> dict[str, Any]:
    """List pods using whichever access path is available.

    Args:
        config: Effective configuration.

    Returns:
        Mapping with the chosen source and the pod summaries.
    """
    if in_cluster():
        namespace = str(config.get("monitoring.kubernetes.namespace", "mlops"))
        selector = str(config.get("monitoring.kubernetes.label_selector", ""))
        status, body = _api_request(
            f"/api/v1/namespaces/{namespace}/pods",
            {"labelSelector": selector} if selector else None,
        )
        if status == 200:
            try:
                return {
                    "source": "in-cluster",
                    "pods": [_pod_summary(item) for item in json.loads(body).get("items", [])],
                }
            except json.JSONDecodeError:
                pass
        return {"source": "in-cluster", "pods": [], "error": f"status {status}"}
    if kubectl_available():
        return {"source": "kubectl", "pods": list_pods_kubectl(config)}
    return {"source": "none", "pods": [], "error": "kubectl is not installed and this is not a pod"}


def collect(config: Config, source: str = "auto", tail: int | None = None) -> LogBundle:
    """Collect logs from the requested source.

    Args:
        config: Effective configuration.
        source: ``auto``, ``local``, ``kubectl`` or ``in-cluster``.
        tail: Lines to fetch per pod or file.

    Returns:
        The collected bundle. ``auto`` prefers the cluster and falls back to the
        local file, merging the fallback message so the dashboard can explain why.
    """
    requested = source.lower()
    if requested == "local":
        return collect_local(config, tail)
    if requested == "kubectl":
        return collect_kubectl(config, tail)
    if requested == "in-cluster":
        return collect_in_cluster(config, tail)

    if not bool(config.get("monitoring.kubernetes.enabled", True)):
        return collect_local(config, tail)

    cluster = collect_in_cluster(config, tail) if in_cluster() else collect_kubectl(config, tail)
    if cluster.available and cluster.records:
        return cluster
    local = collect_local(config, tail)
    local.message = f"{local.message} (cluster source unavailable: {cluster.message})"
    local.source = "local (fallback)"
    return local


__all__ = [
    "LogBundle",
    "collect",
    "collect_in_cluster",
    "collect_kubectl",
    "collect_local",
    "in_cluster",
    "kubectl_available",
    "list_pods",
    "list_pods_kubectl",
]
