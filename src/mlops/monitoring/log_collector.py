"""Collect logs from wherever the service happens to be running.

Two sources are supported, producing the same record shape so the monitoring
view renders a single merged stream:

- **local** — the JSONL files the services write (development and fallback).
- **docker** — read container status and logs using the `docker` CLI. Used when
    the dashboard runs on the host and services are launched with Docker Compose
    or plain Docker.

The Docker approach avoids any Kubernetes dependencies in the dashboard image
and keeps the UI usable when started from `docker compose` as provided.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from mlops.config import Config
from mlops.logging_setup import get_logger, read_jsonl_tail

_LOGGER = get_logger(__name__)

DOCKER_TIMEOUT = 60


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


def docker_available() -> bool:
    """Return True when a `docker` binary is available on PATH."""
    return shutil.which("docker") is not None


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


# kubectl-based collectors and in-cluster collection were removed: this module
# provides Docker and local-file collectors only.


def list_containers(config: Config) -> dict[str, Any]:
    """List containers visible to the host Docker daemon.

    Returns a mapping similar to the former pod listing so the UI can render it.
    """
    if not docker_available():
        return {"source": "none", "containers": [], "error": "docker not available"}

    try:
        completed = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}||{{.Image}}||{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=DOCKER_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _LOGGER.warning("docker ps failed", extra={"error": str(exc)})
        return {"source": "docker", "containers": [], "error": str(exc)}

    if completed.returncode != 0:
        return {"source": "docker", "containers": [], "error": completed.stderr.strip()}

    containers: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("||")
        name = parts[0] if parts else "?"
        image = parts[1] if len(parts) > 1 else ""
        status = parts[2] if len(parts) > 2 else ""
        containers.append({"name": name, "image": image, "status": status})
    return {"source": "docker", "containers": containers}


def _normalise_container_summary(name: str, image: str, status: str) -> dict[str, Any]:
    """Create a lightweight container summary for the UI."""
    return {
        "name": name,
        "image": image,
        "status": status,
    }


def collect_docker(config: Config, tail: int | None = None) -> LogBundle:
    """Collect logs from containers visible to the local Docker daemon.

    The function lists containers and reads the last `tail` lines from each via
    `docker logs`. When Docker is unavailable a descriptive bundle is returned
    rather than raising so the dashboard remains usable.
    """
    lines = int(tail or config.get("monitoring.log_tail_lines", 200))

    if not docker_available():
        return LogBundle(
            source="docker",
            collected_at=_now(),
            available=False,
            message="docker is not on PATH. Start the services with Docker Compose.",
        )

    listing = list_containers(config)
    containers = listing.get("containers", [])
    if not containers:
        return LogBundle(
            source="docker",
            collected_at=_now(),
            available=False,
            message="no running containers were found",
        )

    records: list[dict[str, Any]] = []
    names = []
    for c in containers:
        name = c.get("name")
        names.append(name)
        try:
            completed = subprocess.run(
                ["docker", "logs", "--tail", str(lines), name],
                capture_output=True,
                text=True,
                timeout=DOCKER_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            records.append(
                {
                    "ts": _now(),
                    "level": "WARNING",
                    "logger": "log_collector",
                    "message": f"could not read logs for {name}: {exc}",
                    "pod": name,
                    "source": "docker",
                }
            )
            continue

        if completed.returncode != 0:
            detail = completed.stderr.strip()
            records.append(
                {
                    "ts": _now(),
                    "level": "WARNING",
                    "logger": "log_collector",
                    "message": f"docker logs for {name} failed: {detail}",
                    "pod": name,
                    "source": "docker",
                }
            )
            continue

        for line in completed.stdout.splitlines():
            record = _normalise(line, name, "docker")
            if record:
                records.append(record)

    records.sort(key=lambda item: str(item.get("ts", "")))
    return LogBundle(
        source="docker",
        collected_at=_now(),
        records=records,
        pods=names,
        available=True,
        message=f"{len(records)} lines from {len(containers)} container(s)",
    )


# Kubernetes in-cluster collection and API access were removed in favour of
# a Docker-only dashboard. The functions above implement local file and
# docker-based collection paths.


def list_pods(config: Config) -> dict[str, Any]:
    """Return the container listing under the old key name for views.

    This helper keeps the view code straightforward: it returns a mapping with
    keys similar to the original pod listing but sourced from Docker.
    """
    listing = list_containers(config)
    return {"source": listing.get("source", "docker"), "pods": listing.get("containers", []), "error": listing.get("error")}


def collect(config: Config, source: str = "auto", tail: int | None = None) -> LogBundle:
    """Collect logs from the requested source.

    Args:
        config: Effective configuration.
        source: ``auto``, ``local`` or ``docker``.
        tail: Lines to fetch per pod or file.

    Returns:
        The collected bundle. ``auto`` prefers Docker and falls back to local
        files so the dashboard can explain why.
    """
    requested = source.lower()
    if requested == "local":
        return collect_local(config, tail)
    if requested == "docker":
        return collect_docker(config, tail)

    # auto: prefer Docker then local. Kubernetes collection hooks were removed
    # in favour of a Compose/Docker-first dashboard.

    # non-kubernetes flow: prefer Docker, otherwise local
    if docker_available():
        docker = collect_docker(config, tail)
        if docker.available and docker.records:
            return docker
        # fall through to local
    local = collect_local(config, tail)
    return local


__all__ = [
    "LogBundle",
    "collect",
    "collect_docker",
    "collect_local",
    "docker_available",
    "list_pods",
]
