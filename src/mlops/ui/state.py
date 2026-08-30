"""Aggregate the state of every pipeline stage for the dashboard.

The dashboard's spine shows seven stages. Each one is either done or not, and the
answer has to come from artifacts on disk rather than from a flag the UI sets
itself — otherwise the spine reports what the UI *thinks* happened rather than what
actually did, and a restarted container would show a green pipeline with no model.
Every ``done`` below is therefore derived from a file existing or a service
answering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlops.config import Config
from mlops.logging_setup import get_logger

_LOGGER = get_logger(__name__)

STAGES = (
    ("data", "Data"),
    ("version", "Versioning"),
    ("train", "Training"),
    ("evaluate", "Evaluation"),
    ("serve", "Serving"),
    ("monitor", "Monitoring"),
    ("deploy", "Deployment"),
)


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file, tolerating absence and corruption.

    Args:
        path: File to read.

    Returns:
        The parsed document, or ``None``.
    """
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def api_health(config: Config, timeout: float = 3.0) -> dict[str, Any]:
    """Probe the inference API.

    Args:
        config: Effective configuration.
        timeout: Per-request timeout in seconds.

    Returns:
        Reachability, health body, readiness body and the metrics summary.
    """
    import requests

    base = str(config.get("ui.api_url", "http://127.0.0.1:8000")).rstrip("/")
    result: dict[str, Any] = {"base_url": base, "reachable": False}
    try:
        health = requests.get(f"{base}/health", timeout=timeout)
        result["reachable"] = health.status_code == 200
        result["health"] = health.json() if health.status_code == 200 else None
        result["health_status"] = health.status_code
    except Exception as exc:  # noqa: BLE001 - an unreachable API is a normal state
        result["error"] = str(exc)
        return result

    for name, path in (("ready", "/ready"), ("model_info", "/model-info"), ("metrics", "/metrics-summary")):
        try:
            response = requests.get(f"{base}{path}", timeout=timeout)
            result[name] = response.json()
            result[f"{name}_status"] = response.status_code
        except Exception as exc:  # noqa: BLE001
            result[name] = None
            result[f"{name}_error"] = str(exc)
    return result


def dataset_state(config: Config) -> dict[str, Any]:
    """Summarise the dataset.

    Args:
        config: Effective configuration.

    Returns:
        Raw counts, preprocessing statistics and the dataset lock.
    """
    from mlops.data.versioning import read_dataset_lock

    raw_dir = config.path("paths.raw_dir")
    classes = list(config.get("data.class_names", ["cat", "dog"]))
    raw_counts = {
        name: len([p for p in (raw_dir / name).glob("*") if p.is_file()])
        if (raw_dir / name).is_dir()
        else 0
        for name in classes
    }
    stats = _read_json(config.path("paths.preprocess_stats"))
    return {
        "raw_dir": str(raw_dir),
        "raw_counts": raw_counts,
        "raw_total": sum(raw_counts.values()),
        "processed": stats,
        "lock": read_dataset_lock(config),
        "manifest_exists": config.path("paths.manifest_csv").is_file(),
    }


def model_state(config: Config) -> dict[str, Any]:
    """Summarise the trained model.

    Args:
        config: Effective configuration.

    Returns:
        Checkpoint presence and size, training history and baseline metrics.
    """
    model_path = config.path("paths.model_path")
    history = _read_json(config.path("paths.history_path"))
    baseline = _read_json(config.path("paths.baseline_path"))
    promotion = _read_json(config.path("paths.metrics_dir") / "promotion.json")
    card_path = config.path("paths.model_card")
    return {
        "checkpoint": str(model_path),
        "exists": model_path.is_file(),
        "bytes": model_path.stat().st_size if model_path.is_file() else 0,
        "history": history,
        "baseline": baseline,
        "promotion": promotion,
        "model_card": card_path.read_text(encoding="utf-8") if card_path.is_file() else "",
    }


def plot_files(config: Config) -> list[dict[str, str]]:
    """List the generated figures.

    Args:
        config: Effective configuration.

    Returns:
        Name and URL for each PNG in the plots directory.
    """
    plots_dir = config.path("paths.plots_dir")
    if not plots_dir.is_dir():
        return []
    return [
        {"name": path.stem.replace("_", " "), "file": path.name, "url": f"/api/plot/{path.name}"}
        for path in sorted(plots_dir.glob("*.png"))
    ]


def stage_status(config: Config, api: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Compute the state of each pipeline stage.

    Args:
        config: Effective configuration.
        api: A prior API probe result, to avoid probing twice.

    Returns:
        One entry per stage with a ``done`` flag and a short detail line.
    """
    from mlops.data.versioning import dvc_available, read_dataset_lock
    from mlops.monitoring.perf_tracker import load_report
    from mlops.tracking.tracker import list_runs

    data = dataset_state(config)
    model = model_state(config)
    api = api if api is not None else {"reachable": False}
    lock = read_dataset_lock(config)
    runs = list_runs(config, limit=200)
    report = load_report(config)
    docker_compose = (config.root / "docker" / "docker-compose.yml").is_file()

    processed_total = (data.get("processed") or {}).get("total_images", 0)
    train_runs = [run for run in runs if run.get("tags", {}).get("stage") == "train"]

    details = {
        "data": (
            data["raw_total"] > 0 and processed_total > 0,
            f"{data['raw_total']} raw, {processed_total} processed images",
        ),
        "version": (
            lock is not None,
            f"lock digest {lock['combined_digest'][:12]}" if lock else "no dataset lock yet",
        ),
        "train": (
            model["exists"] and bool(train_runs),
            f"{len(train_runs)} training run(s), checkpoint {model['bytes'] / 1024:.0f} KB"
            if model["exists"]
            else "no checkpoint",
        ),
        "evaluate": (
            model["baseline"] is not None,
            f"test accuracy {model['baseline']['metrics']['accuracy']:.3f}"
            if model.get("baseline")
            else "not evaluated",
        ),
        "serve": (
            bool(api.get("reachable")),
            f"API up at {api.get('base_url', '')}" if api.get("reachable") else "API not reachable",
        ),
        "monitor": (
            report is not None,
            f"last check {'PASS' if (report or {}).get('passed') else 'FAIL'}"
            if report
            else "no performance check yet",
        ),
        "deploy": (
            bool(docker_compose),
            "docker-compose ready" if docker_compose else "no compose file",
        ),
    }

    stages = []
    for key, label in STAGES:
        done, detail = details[key]
        stages.append({"key": key, "label": label, "done": bool(done), "detail": detail})
    stages_by_key = {stage["key"]: stage for stage in stages}
    stages_by_key["version"]["detail"] += " · dvc " + ("installed" if dvc_available() else "not installed")
    return stages


def project_status(config: Config) -> dict[str, Any]:
    """Build the full status document the dashboard renders.

    Args:
        config: Effective configuration.

    Returns:
        Everything the overview needs in one payload.
    """
    from mlops.data.versioning import versioning_status
    from mlops.monitoring.perf_tracker import load_report
    from mlops.tracking.tracker import list_runs, mlflow_available

    api = api_health(config)
    return {
        "project": {
            "name": str(config.get("project.name", "mlops-catsdogs")),
            "version": str(config.get("project.version", "1.0.0")),
            "root": str(config.root),
        },
        "stages": stage_status(config, api=api),
        "dataset": dataset_state(config),
        "versioning": versioning_status(config),
        "model": model_state(config),
        "runs": list_runs(config, limit=25),
        "mlflow_available": mlflow_available(),
        "api": api,
        "perf_report": load_report(config),
        "plots": plot_files(config),
    }


__all__ = [
    "STAGES",
    "api_health",
    "dataset_state",
    "model_state",
    "plot_files",
    "project_status",
    "stage_status",
]
