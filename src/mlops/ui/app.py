"""The control-room dashboard.

Every capability the assignment asks to be visible has a panel here, and every
panel is backed by a real action rather than a screenshot: the buttons run the same
code paths the Makefile and CI run, through :mod:`mlops.cli`'s underlying functions.

The API surface is deliberately small and uniform:

* ``GET  /api/status``            everything the overview renders
* ``POST /api/actions/<name>``    start a background job, returns a job id
* ``GET  /api/jobs/<job_id>``     poll status and captured logs
* ``GET  /api/...``               read-only views (runs, metrics, logs, pods)

Actions that touch shared artifacts run through :class:`~mlops.ui.jobs.JobRunner`,
which serialises them. Read-only views answer immediately.
"""

from __future__ import annotations

import base64
import json
import random
import time
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_file

from mlops.config import Config, get_config
from mlops.logging_setup import configure_logging, get_logger
from mlops.ui.jobs import Job, JobRunner
from mlops.ui.state import api_health, project_status, stage_status

_LOGGER = get_logger(__name__)


def _api_base(config: Config) -> str:
    """Return the inference API base URL.

    Args:
        config: Effective configuration.

    Returns:
        The base URL without a trailing slash.
    """
    return str(config.get("ui.api_url", "http://127.0.0.1:8000")).rstrip("/")


def create_ui_app(config: Config | None = None) -> Flask:
    """Build the dashboard application.

    Args:
        config: Effective configuration; loaded from disk when omitted.

    Returns:
        The configured Flask app, with ``app.jobs`` attached for tests.
    """
    config = config or get_config()
    config.ensure_dirs()
    configure_logging(
        level=str(config.get("logging.level", "INFO")),
        log_file=config.path("monitoring.ui_log_file"),
    )

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    runner = JobRunner()
    app.jobs = runner  # type: ignore[attr-defined]
    app.mlops_config = config  # type: ignore[attr-defined]

    # -- helpers -----------------------------------------------------------

    def _submit(name: str, action: Any) -> tuple[Response, int]:
        """Start a job and return its id.

        Args:
            name: Action name.
            action: Callable receiving the job.

        Returns:
            The job payload and status code, or 409 when another job is running.
        """
        try:
            job = runner.submit(name, action)
        except RuntimeError as exc:
            return jsonify({"error": str(exc), "busy": True}), 409
        return jsonify(job.to_dict()), 202

    def _proxy(path: str, timeout: float = 10.0) -> tuple[dict[str, Any], int]:
        """Call the inference API and return its JSON body.

        Args:
            path: Path beginning with ``/``.
            timeout: Request timeout in seconds.

        Returns:
            Tuple of body and status code; status ``0`` means unreachable.
        """
        import requests

        started = time.perf_counter()
        try:
            response = requests.get(f"{_api_base(config)}{path}", timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "endpoint": path, "reachable": False}, 0
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text[:4000]}
        return {
            "endpoint": path,
            "status_code": response.status_code,
            "latency_ms": elapsed_ms,
            "body": body,
            "reachable": True,
        }, response.status_code

    # -- pages -------------------------------------------------------------

    @app.get("/")
    def index() -> str:
        """Render the dashboard shell.

        Returns:
            The rendered HTML page.
        """
        return render_template(
            "index.html",
            project_name=str(config.get("project.name", "mlops-catsdogs")),
            project_version=str(config.get("project.version", "1.0.0")),
            api_url=_api_base(config),
            refresh_seconds=int(config.get("ui.refresh_seconds", 5)),
        )

    @app.get("/health")
    def ui_health() -> Response:
        """Liveness probe for the dashboard itself.

        Returns:
            A small JSON body.
        """
        return jsonify({"status": "ok", "component": "ui"})

    # -- read-only views ---------------------------------------------------

    @app.get("/api/status")
    def api_status() -> Response:
        """Return the full project status.

        Returns:
            The status document.
        """
        return jsonify(project_status(config))

    @app.get("/api/stages")
    def api_stages() -> Response:
        """Return just the pipeline spine.

        Returns:
            Stage completion for the header.
        """
        return jsonify({"stages": stage_status(config, api=api_health(config))})

    @app.get("/api/runs")
    def api_runs() -> Response:
        """Return the tracked experiment runs.

        Returns:
            Run records and whether MLflow is available.
        """
        from mlops.tracking.tracker import list_runs, mlflow_available

        return jsonify(
            {
                "runs": list_runs(config, limit=int(request.args.get("limit", 50))),
                "mlflow_available": mlflow_available(),
                "tracking_uri": str(config.get("tracking.mlflow_tracking_uri", "")),
                "experiment": str(config.get("tracking.experiment_name", "")),
            }
        )

    @app.get("/api/runs/<run_id>")
    def api_run(run_id: str) -> tuple[Response, int] | Response:
        """Return one run record.

        Args:
            run_id: The run identifier.

        Returns:
            The record, or 404.
        """
        from mlops.tracking.tracker import get_run

        record = get_run(config, run_id)
        if record is None:
            return jsonify({"error": f"run {run_id} not found"}), 404
        return jsonify(record)

    @app.get("/api/model-card")
    def api_model_card() -> Response:
        """Return the generated model card.

        Returns:
            The Markdown text, empty when evaluation has not run.
        """
        path = config.path("paths.model_card")
        return jsonify(
            {
                "exists": path.is_file(),
                "markdown": path.read_text(encoding="utf-8") if path.is_file() else "",
            }
        )

    @app.get("/api/plot/<name>")
    def api_plot(name: str) -> tuple[Response, int] | Response:
        """Serve a generated figure.

        Args:
            name: File name inside the plots directory.

        Returns:
            The PNG, or 404.
        """
        safe = Path(name).name
        path = config.path("paths.plots_dir") / safe
        if not path.is_file():
            return jsonify({"error": f"plot {safe} not found"}), 404
        return send_file(path, mimetype="image/png")

    @app.get("/api/sample-images")
    def api_sample_images() -> Response:
        """Return a few processed images as data URLs.

        Returns:
            Sample images with their split and class, for the data panel.
        """
        from mlops.data.preprocess import read_manifest

        manifest_path = config.path("paths.manifest_csv")
        if not manifest_path.is_file():
            return jsonify({"samples": [], "message": "no manifest yet; run preprocessing"})
        rows = read_manifest(manifest_path)
        rng = random.Random(int(request.args.get("seed", 7)))
        chosen = rng.sample(rows, min(8, len(rows)))
        samples = []
        for row in chosen:
            path = config.root / row.path
            if not path.is_file():
                continue
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            samples.append(
                {
                    "path": row.path,
                    "split": row.split,
                    "class_name": row.class_name,
                    "sha256_12": row.sha256[:12],
                    "data_url": f"data:image/jpeg;base64,{encoded}",
                }
            )
        return jsonify({"samples": samples, "total": len(rows)})

    @app.get("/api/metrics")
    def api_metrics() -> Response:
        """Return the API's metrics, both parsed and raw.

        Returns:
            The metrics summary, the raw exposition text and reachability.
        """
        import requests

        from mlops.serving.metrics import parse_exposition

        base = _api_base(config)
        payload: dict[str, Any] = {"base_url": base, "reachable": False}
        try:
            summary = requests.get(f"{base}/metrics-summary", timeout=5)
            raw = requests.get(f"{base}/metrics", timeout=5)
            payload.update(
                {
                    "reachable": True,
                    "summary": summary.json() if summary.status_code == 200 else None,
                    "raw": raw.text if raw.status_code == 200 else "",
                    "series": parse_exposition(raw.text) if raw.status_code == 200 else {},
                }
            )
        except Exception as exc:  # noqa: BLE001
            payload["error"] = str(exc)
        return jsonify(payload)

    @app.get("/api/logs")
    def api_logs() -> Response:
        """Collect logs from the requested source.

        Returns:
            A log bundle, plus which sources are currently possible.
        """
        from mlops.monitoring.log_collector import collect, docker_available

        source = request.args.get("source", "auto")
        tail = request.args.get("tail")
        bundle = collect(config, source=source, tail=int(tail) if tail else None)
        payload = bundle.to_dict()
        payload["capabilities"] = {
            "docker": docker_available(),
        }
        return jsonify(payload)

    @app.get("/api/containers")
    def api_containers() -> Response:
        """List running containers for the deployment.

        Returns:
            Container summaries and which access path was used.
        """
        from mlops.monitoring.log_collector import list_containers

        return jsonify(list_containers(config))

    # Kubernetes-specific endpoints removed — the UI is Docker-only.

    @app.get("/api/deployment")
    def api_deployment() -> Response:
        """Describe the deployment assets on disk.

        Returns:
            Manifests, workflows, Docker files and the configured image name.
        """
        def _listing(folder: str, pattern: str) -> list[dict[str, Any]]:
            base = config.root / folder
            if not base.is_dir():
                return []
            return [
                {"name": path.name, "bytes": path.stat().st_size, "path": f"{folder}/{path.name}"}
                for path in sorted(base.glob(pattern))
            ]

        return jsonify(
            {
                "k8s": [],
                "workflows": _listing(".github/workflows", "*.yml"),
                "docker": _listing("docker", "*"),
                "image": str(config.get("deployment.image", "mlops-catsdogs:local")),
            }
        )

    @app.get("/api/file")
    def api_file() -> tuple[Response, int] | Response:
        """Return the text of a repository file, for the deployment panel.

        Args:
            None directly; ``path`` is a query parameter.

        Returns:
            The file text, or 404/403. Paths are resolved and confined to the
            project root so the panel cannot be turned into a file browser for the
            whole host.
        """
        raw = request.args.get("path", "")
        candidate = (config.root / raw).resolve()
        try:
            candidate.resolve().relative_to(Path(config.root).resolve())
        except ValueError:
            return jsonify({"error": "path is outside the project"}), 403
        allowed = {".yaml", ".yml", ".md", ".txt", ".json", ".cfg", ".toml", ""}
        if candidate.suffix.lower() not in allowed or not candidate.is_file():
            return jsonify({"error": f"cannot read {raw}"}), 404
        return jsonify({"path": raw, "text": candidate.read_text(encoding="utf-8")[:200_000]})

    @app.get("/api/jobs")
    def api_jobs() -> Response:
        """Return recent background jobs.

        Returns:
            Job summaries and whether a job is currently running.
        """
        return jsonify({"jobs": runner.recent(limit=12), "busy": runner.busy})

    @app.get("/api/jobs/<job_id>")
    def api_job(job_id: str) -> tuple[Response, int] | Response:
        """Return one job with its captured logs.

        Args:
            job_id: The job identifier.

        Returns:
            The job payload, or 404.
        """
        job = runner.get(job_id)
        if job is None:
            return jsonify({"error": f"job {job_id} not found"}), 404
        return jsonify(job.to_dict())

    # -- API probes and prediction ----------------------------------------

    @app.post("/api/probe")
    def api_probe() -> Response:
        """Call one endpoint on the inference API and show the exchange.

        Returns:
            The status code, latency and body returned by the API.
        """
        document = request.get_json(silent=True) or {}
        path = str(document.get("path", "/health"))
        if not path.startswith("/"):
            path = "/" + path
        body, _status = _proxy(path)
        return jsonify(body)

    @app.post("/api/reload-model")
    def api_reload_model() -> tuple[Response, int] | Response:
        """Ask the inference API to re-read the checkpoint.

        Returns:
            The API's response, so the panel can show which model is now live.
        """
        import requests

        try:
            response = requests.post(f"{_api_base(config)}/reload", timeout=60)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc), "reachable": False}), 502
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text[:2000]}
        return jsonify({"status_code": response.status_code, "body": body})

    @app.post("/api/predict")
    def api_predict() -> tuple[Response, int] | Response:
        """Forward an uploaded image to the inference API.

        Returns:
            The API's prediction alongside the round-trip latency.
        """
        import requests

        if "file" not in request.files:
            return jsonify({"error": "attach an image in the 'file' field"}), 400
        upload = request.files["file"]
        payload = upload.read()
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{_api_base(config)}/predict",
                files={"file": (upload.filename or "upload.jpg", payload, upload.mimetype or "image/jpeg")},
                timeout=30,
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc), "reachable": False}), 502
        elapsed = round((time.perf_counter() - started) * 1000.0, 2)
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text[:2000]}
        return jsonify(
            {
                "status_code": response.status_code,
                "round_trip_ms": elapsed,
                "body": body,
                "preview": f"data:{upload.mimetype or 'image/jpeg'};base64,"
                + base64.b64encode(payload).decode("ascii")
                if len(payload) < 1_500_000
                else "",
            }
        )

    @app.post("/api/predict-sample")
    def api_predict_sample() -> tuple[Response, int] | Response:
        """Send a random held-out test image to the API.

        Returns:
            The prediction, the true label and whether they agree.
        """
        import requests

        from mlops.data.preprocess import read_manifest

        manifest_path = config.path("paths.manifest_csv")
        if not manifest_path.is_file():
            return jsonify({"error": "no manifest yet; run preprocessing first"}), 400
        rows = [row for row in read_manifest(manifest_path) if row.split == "test"]
        if not rows:
            return jsonify({"error": "the test split is empty"}), 400
        row = random.choice(rows)
        path = config.root / row.path
        payload = path.read_bytes()
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{_api_base(config)}/predict",
                files={"file": (path.name, payload, "image/jpeg")},
                timeout=30,
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": str(exc), "reachable": False}), 502
        elapsed = round((time.perf_counter() - started) * 1000.0, 2)
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text[:2000]}
        predicted = body.get("label")
        return jsonify(
            {
                "status_code": response.status_code,
                "round_trip_ms": elapsed,
                "body": body,
                "true_label": row.class_name,
                "correct": predicted == row.class_name if predicted else None,
                "source_path": row.path,
                "preview": "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii"),
            }
        )

    @app.post("/api/load-test")
    def api_load_test() -> tuple[Response, int] | Response:
        """Fire a short burst of predictions so the metrics panel has data.

        Returns:
            Per-request outcomes and a latency summary.
        """
        import requests

        from mlops.data.preprocess import read_manifest

        document = request.get_json(silent=True) or {}
        count = max(1, min(60, int(document.get("count", 20))))
        manifest_path = config.path("paths.manifest_csv")
        if not manifest_path.is_file():
            return jsonify({"error": "no manifest yet; run preprocessing first"}), 400
        rows = [row for row in read_manifest(manifest_path) if row.split == "test"] or read_manifest(
            manifest_path
        )
        rng = random.Random(11)
        latencies: list[float] = []
        statuses: dict[str, int] = {}
        correct = 0
        for _ in range(count):
            row = rng.choice(rows)
            payload = (config.root / row.path).read_bytes()
            started = time.perf_counter()
            try:
                response = requests.post(
                    f"{_api_base(config)}/predict",
                    files={"file": (Path(row.path).name, payload, "image/jpeg")},
                    timeout=20,
                )
            except Exception as exc:  # noqa: BLE001
                statuses["error"] = statuses.get("error", 0) + 1
                _LOGGER.warning("load test request failed", extra={"error": str(exc)})
                continue
            latencies.append((time.perf_counter() - started) * 1000.0)
            statuses[str(response.status_code)] = statuses.get(str(response.status_code), 0) + 1
            if response.status_code == 200:
                try:
                    if response.json().get("label") == row.class_name:
                        correct += 1
                except ValueError:
                    pass
        if not latencies:
            return jsonify({"error": "every request failed; is the API running?"}), 502
        ordered = sorted(latencies)
        return jsonify(
            {
                "requests": count,
                "statuses": statuses,
                "accuracy_on_burst": round(correct / max(1, len(latencies)), 4),
                "latency_ms": {
                    "mean": round(sum(ordered) / len(ordered), 2),
                    "p50": round(ordered[len(ordered) // 2], 2),
                    "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 2),
                    "max": round(ordered[-1], 2),
                },
            }
        )

    # -- actions -----------------------------------------------------------

    @app.post("/api/actions/generate-data")
    def action_generate_data() -> tuple[Response, int]:
        """Generate the synthetic raw dataset.

        Returns:
            The started job.
        """
        document = request.get_json(silent=True) or {}
        per_class = int(document.get("per_class", config.get("data.images_per_class", 300)))

        def action(_job: Job) -> Any:
            from mlops.data.generate import generate_dataset

            return generate_dataset(config, per_class=per_class)

        return _submit("generate-data", action)

    @app.post("/api/actions/preprocess")
    def action_preprocess() -> tuple[Response, int]:
        """Preprocess the raw dataset and refresh the dataset lock.

        Returns:
            The started job.
        """
        def action(_job: Job) -> Any:
            from mlops.data.preprocess import run
            from mlops.data.versioning import write_dataset_lock

            stats = run(config)
            stats["lock"] = write_dataset_lock(config)
            return stats

        return _submit("preprocess", action)

    @app.post("/api/actions/train")
    def action_train() -> tuple[Response, int]:
        """Train a model with the current configuration.

        Returns:
            The started job.
        """
        document = request.get_json(silent=True) or {}
        overrides = {
            key: document[key]
            for key in ("model_type", "epochs", "learning_rate")
            if key in document and document[key] not in (None, "")
        }

        def action(_job: Job) -> Any:
            from mlops.config import Config as ConfigType
            from mlops.models.train import run

            effective = config
            if overrides:
                raw = json.loads(json.dumps(config.raw))
                if "model_type" in overrides:
                    raw["model"]["type"] = str(overrides["model_type"])
                if "epochs" in overrides:
                    raw["training"]["epochs"] = int(overrides["epochs"])
                if "learning_rate" in overrides:
                    raw["training"]["learning_rate"] = float(overrides["learning_rate"])
                effective = ConfigType(raw=raw, root=config.root)
            return run(effective).to_dict()

        return _submit("train", action)

    @app.post("/api/actions/evaluate")
    def action_evaluate() -> tuple[Response, int]:
        """Evaluate the checkpoint on the held-out split.

        Returns:
            The started job.
        """
        def action(_job: Job) -> Any:
            from mlops.models.evaluate import run

            return run(config).to_dict()

        return _submit("evaluate", action)

    @app.post("/api/actions/promote")
    def action_promote() -> tuple[Response, int]:
        """Apply the promotion gate to the best run.

        Returns:
            The started job.
        """
        def action(_job: Job) -> Any:
            from mlops.tracking.tracker import register_best_model

            return register_best_model(config)

        return _submit("promote", action)

    @app.post("/api/actions/pipeline")
    def action_pipeline() -> tuple[Response, int]:
        """Run data, preprocessing, training, evaluation and promotion in order.

        Returns:
            The started job.
        """
        document = request.get_json(silent=True) or {}
        include_data = bool(document.get("include_data", True))
        per_class = int(document.get("per_class", config.get("data.images_per_class", 300)))

        def action(_job: Job) -> Any:
            from mlops.data.generate import generate_dataset
            from mlops.data.preprocess import run as preprocess_run
            from mlops.data.versioning import write_dataset_lock
            from mlops.models.evaluate import run as evaluate_run
            from mlops.models.train import run as train_run
            from mlops.tracking.tracker import register_best_model

            steps: dict[str, Any] = {}
            if include_data:
                steps["generate"] = generate_dataset(config, per_class=per_class)
            steps["preprocess"] = preprocess_run(config)
            steps["lock"] = write_dataset_lock(config)
            steps["train"] = train_run(config).to_dict()
            steps["evaluate"] = evaluate_run(config).to_dict()
            steps["promote"] = register_best_model(config)
            return steps

        return _submit("full-pipeline", action)

    @app.post("/api/actions/dvc")
    def action_dvc() -> tuple[Response, int]:
        """Run a DVC subcommand.

        Returns:
            The started job.
        """
        document = request.get_json(silent=True) or {}
        allowed = {
            "status": ["status"],
            "repro": ["repro", "--force"],
            "dag": ["dag", "--dot"],
            # No "add": every data path is a dvc.yaml stage output, and DVC
            # rejects a path that is both a pipeline output and a manual add.
            "commit": ["commit", "--force"],
            "push": ["push"],
            "pull": ["pull"],
        }
        command = str(document.get("command", "status"))
        if command not in allowed:
            return jsonify({"error": f"unknown dvc command {command!r}"}), 400

        def action(_job: Job) -> Any:
            from mlops.data.versioning import run_dvc

            return run_dvc(config, allowed[command]).to_dict()

        return _submit(f"dvc-{command}", action)

    @app.post("/api/actions/dataset-lock")
    def action_dataset_lock() -> tuple[Response, int]:
        """Recompute the content-hash dataset lock.

        Returns:
            The started job.
        """
        def action(_job: Job) -> Any:
            from mlops.data.versioning import write_dataset_lock

            return write_dataset_lock(config)

        return _submit("dataset-lock", action)

    @app.post("/api/actions/perf-check")
    def action_perf_check() -> tuple[Response, int]:
        """Score the live endpoint against the training baseline.

        Returns:
            The started job.
        """
        document = request.get_json(silent=True) or {}
        sample_size = int(document.get("sample_size", config.get("monitoring.perf_check.sample_size", 40)))
        endpoint = str(document.get("endpoint", _api_base(config)))

        def action(_job: Job) -> Any:
            from mlops.monitoring.perf_tracker import run

            return run(config, endpoint_url=endpoint, sample_size=sample_size).to_dict()

        return _submit("perf-check", action)

    @app.post("/api/actions/collect-docker-logs")
    def action_collect_docker_logs() -> tuple[Response, int]:
        """Pull logs from local Docker containers and merge them into the monitoring view."""
        document = request.get_json(silent=True) or {}
        source = str(document.get("source", "docker"))

        def action(_job: Job) -> Any:
            from mlops.monitoring.log_collector import collect

            bundle = collect(config, source=source)
            payload = bundle.to_dict()
            archive = config.path("paths.logs_dir") / "docker-logs.jsonl"
            archive.parent.mkdir(parents=True, exist_ok=True)
            with archive.open("w", encoding="utf-8") as handle:
                for record in bundle.records:
                    handle.write(json.dumps(record) + "\n")
            payload["archive"] = str(archive.resolve().relative_to(Path(config.root).resolve()))
            payload.pop("records", None)
            return payload

        return _submit("collect-docker-logs", action)

    _LOGGER.info("dashboard ready", extra={"api_url": _api_base(config)})
    return app


def main() -> int:
    """Run the dashboard with the built-in server.

    Returns:
        Process exit status.
    """
    config = get_config()
    app = create_ui_app(config)
    app.run(
        host=str(config.get("ui.host", "0.0.0.0")),
        port=int(config.get("ui.port", 8501)),
        threaded=True,
        use_reloader=False,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["create_ui_app", "main"]
