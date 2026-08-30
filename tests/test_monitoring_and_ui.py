"""Tests for monitoring, log collection, versioning and the dashboard.

The performance checker normally posts over HTTP. Here its client is swapped for
one backed by the API's Flask test client, so the same code path is exercised end
to end without a socket — which is also what makes the check testable in CI.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

from helpers import TempProject, prepared_project, sample_image_bytes, trained_project
from mlops.data.versioning import read_dataset_lock, run_dvc, versioning_status, write_dataset_lock
from mlops.logging_setup import read_jsonl_tail, safe_payload_meta
from mlops.monitoring import log_collector, perf_tracker
from mlops.serving.app import create_app
from mlops.ui.app import create_ui_app
from mlops.ui.jobs import JobRunner
from mlops.ui.state import project_status, stage_status

# ---------------------------------------------------------------- versioning

def test_dataset_lock_records_both_trees() -> None:
    """The lock indexes the raw and processed trees and produces one digest."""
    with prepared_project() as project:
        lock = write_dataset_lock(project.config)
        assert lock["raw"]["files"] > 0
        assert lock["processed"]["files"] > 0
        assert len(lock["combined_digest"]) == 64
        assert read_dataset_lock(project.config)["combined_digest"] == lock["combined_digest"]


def test_dataset_lock_changes_when_the_data_changes() -> None:
    """Touching one byte of data changes the digest."""
    with prepared_project() as project:
        before = write_dataset_lock(project.config)["combined_digest"]
        extra = project.config.path("paths.raw_dir") / "cat" / "extra.jpg"
        extra.write_bytes(sample_image_bytes())
        assert write_dataset_lock(project.config)["combined_digest"] != before


def test_missing_dvc_binary_is_reported_not_raised() -> None:
    """A missing DVC install produces an actionable result object."""
    with TempProject() as project:
        result = run_dvc(project.config, ["status"])
        if not result.available:
            assert "pip install" in result.stderr
            assert result.ok is False


def test_versioning_status_is_renderable_without_dvc() -> None:
    """The dashboard's versioning payload is complete whether or not DVC exists."""
    with prepared_project() as project:
        write_dataset_lock(project.config)
        status = versioning_status(project.config)
        for key in ("dvc_installed", "dvc_initialised", "lock", "git"):
            assert key in status
        assert status["lock"]["combined_digest"]


# ---------------------------------------------------------------- logging

def test_payload_metadata_never_contains_the_payload() -> None:
    """Log metadata describes an upload without retaining any of its bytes."""
    payload = sample_image_bytes()
    meta = safe_payload_meta(payload, "image/jpeg")
    assert meta["bytes"] == len(payload)
    assert meta["content_type"] == "image/jpeg"
    assert len(meta["sha256_12"]) == 12
    serialised = json.dumps(meta)
    assert "\\xff" not in serialised
    assert len(serialised) < 200


def test_log_tail_survives_a_corrupt_line() -> None:
    """A half-written line is wrapped rather than dropped or fatal."""
    with TempProject() as project:
        path = project.root / "logs" / "test.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"level":"INFO","message":"one"}\nnot json at all\n{"level":"ERROR","message":"two"}\n',
            encoding="utf-8",
        )
        records = read_jsonl_tail(path, 10)
        assert len(records) == 3
        assert records[1]["level"] == "RAW"
        assert records[2]["message"] == "two"


def test_log_tail_of_a_missing_file_is_empty() -> None:
    """Tailing a file that does not exist returns nothing rather than raising."""
    assert read_jsonl_tail(Path("/nonexistent/nowhere.jsonl")) == []


# ---------------------------------------------------------------- log collection

def test_local_collection_reads_the_service_log() -> None:
    """The local collector returns the records the service wrote."""
    with TempProject() as project:
        path = project.config.path("monitoring.log_file")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"level":"INFO","message":"hello","ts":"2026-01-01T00:00:00Z"}\n', encoding="utf-8")
        bundle = log_collector.collect_local(project.config)
        assert bundle.available is True
        assert any(record["message"] == "hello" for record in bundle.records)
        assert bundle.records[0]["source"] == "local"


def test_docker_collection_parses_container_logs(tmp_path: Path | None = None) -> None:
    """With a stubbed docker on PATH, container logs are collected and normalised.

    A stub is used rather than a mock so subprocess handling and parsing are
    exercised similarly to the real runtime.
    """
    with TempProject() as project:
        bin_dir = project.root / "stubbin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        stub = bin_dir / "docker"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            'if [[ "$*" == *"ps"* ]]; then\n'
            '  echo \"service-api||mlops-catsdogs:local||Up 2 minutes\"\n'
            '  echo \"service-ui||mlops-catsdogs:local||Up 2 minutes\"\n'
            "else\n"
            '  # logs --tail N NAME\n'
            '  echo \"{\\\"level\\\":\\\"INFO\\\",\\\"message\\\":\\\"served a prediction\\\",\\\"ts\\\":\\\"2026-01-01T00:00:01Z\\\"}\"\n'
            "  echo 'a plain non-JSON line'\n"
            "fi\n",
            encoding="utf-8",
        )
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        original = os.environ["PATH"]
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{original}"
        try:
            listing = log_collector.list_containers(project.config)
            assert listing["source"] == "docker"
            assert any(c["name"] == "service-api" for c in listing["containers"]) or listing["containers"]

            bundle = log_collector.collect_docker(project.config)
            assert bundle.available is True
            messages = [record["message"] for record in bundle.records]
            assert "served a prediction" in " ".join(messages)
            assert any(record["source"] == "docker" for record in bundle.records)
        finally:
            os.environ["PATH"] = original


def test_collection_falls_back_to_local_and_says_why() -> None:
    """When Docker is unavailable the collector falls back to local logs."""
    with TempProject() as project:
        path = project.config.path("monitoring.log_file")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"level":"INFO","message":"local record"}\n', encoding="utf-8")
        original = os.environ["PATH"]
        os.environ["PATH"] = str(project.root / "empty-bin")
        try:
            bundle = log_collector.collect(project.config, source="auto")
            # Docker is unavailable in this PATH, so auto should fall back to local
            assert bundle.source == "local"
            assert any(record["message"] == "local record" for record in bundle.records)
        finally:
            os.environ["PATH"] = original

def test_no_incluster_collector_present() -> None:
    """The project no longer exposes an in-cluster collector."""
    with TempProject():
        assert not hasattr(log_collector, "collect_in_cluster")


# ---------------------------------------------------------------- performance

def _flask_client_adapter(app):
    """Adapt a Flask test client to the performance checker's client interface.

    Args:
        app: The API application.

    Returns:
        A callable matching ``(payload, filename) -> response``.
    """
    client = app.test_client()

    def send(payload: bytes, filename: str):
        import io

        return client.post(
            "/predict",
            data={"file": (io.BytesIO(payload), filename)},
            content_type="multipart/form-data",
        )

    return send


def test_performance_check_scores_live_traffic_against_the_baseline() -> None:
    """The checker sends labelled images through the API and compares to baseline."""
    with trained_project() as project:
        app = create_app(project.config)
        result = perf_tracker.run(
            project.config,
            endpoint_url="http://testclient",
            sample_size=6,
            client=_flask_client_adapter(app),
            use_mlflow=False,
        )
        assert result.sample_size > 0
        assert result.failures == []
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert result.baseline_metrics["accuracy"] >= 0.0
        assert len(result.gates) == 3
        assert sum(sum(row) for row in result.confusion) == result.sample_size
        assert project.config.path("paths.perf_report").is_file()


def test_performance_gates_fail_when_accuracy_is_below_the_floor() -> None:
    """An impossible accuracy floor produces a FAIL verdict, not an exception."""
    with trained_project(**{"monitoring.perf_check.min_accuracy": 1.01}) as project:
        app = create_app(project.config)
        result = perf_tracker.run(
            project.config,
            endpoint_url="http://testclient",
            sample_size=6,
            client=_flask_client_adapter(app),
            use_mlflow=False,
        )
        assert result.passed is False
        assert any(gate["name"] == "absolute_accuracy" and not gate["passed"] for gate in result.gates)


def test_performance_check_reports_an_unreachable_endpoint() -> None:
    """Every request failing raises a diagnosable error rather than reporting 0%."""
    with trained_project() as project:
        def broken(_payload: bytes, _filename: str):
            raise ConnectionError("connection refused")

        try:
            perf_tracker.run(
                project.config,
                endpoint_url="http://127.0.0.1:1",
                sample_size=4,
                client=broken,
                use_mlflow=False,
            )
        except perf_tracker.PerfCheckError as exc:
            assert "is the service running" in str(exc)
        else:
            raise AssertionError("expected PerfCheckError")


def test_sampling_is_stratified_and_deterministic() -> None:
    """The labelled batch is class-balanced and identical between runs."""
    with prepared_project() as project:
        from mlops.data.preprocess import read_manifest

        rows = [r for r in read_manifest(project.config.path("paths.manifest_csv")) if r.split == "test"]
        first = perf_tracker.stratified_sample(rows, 4, seed=42)
        second = perf_tracker.stratified_sample(rows, 4, seed=42)
        assert [row.path for row in first] == [row.path for row in second]
        assert len({row.class_name for row in first}) == 2


# ---------------------------------------------------------------- jobs and UI

def test_job_runner_captures_results_and_logs() -> None:
    """A submitted job records its return value and the logs it emitted."""
    import logging

    runner = JobRunner()

    def action(_job):
        logging.getLogger("test.job").info("working")
        return {"answer": 42}

    # The suite silences INFO globally so the pass/fail lines stay readable. This
    # one test is about log capture, so it lifts that for its own duration.
    previous = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    try:
        job = runner.submit("demo", action)
        for _ in range(100):
            if runner.get(job.job_id).status != "running":
                break
            time.sleep(0.05)
    finally:
        logging.disable(previous)

    finished = runner.get(job.job_id)
    assert finished.status == "succeeded"
    assert finished.result == {"answer": 42}
    assert any("working" in line for line in finished.logs)


def test_job_runner_records_failures_with_a_traceback() -> None:
    """A failing job is marked failed and keeps the error for the UI."""
    runner = JobRunner()

    def action(_job):
        raise RuntimeError("it broke")

    job = runner.submit("boom", action)
    for _ in range(100):
        if runner.get(job.job_id).status != "running":
            break
        time.sleep(0.05)

    finished = runner.get(job.job_id)
    assert finished.status == "failed"
    assert "it broke" in finished.error


def test_job_runner_refuses_to_run_two_jobs_at_once() -> None:
    """Concurrent stages would corrupt shared artifacts, so the second is refused."""
    runner = JobRunner()

    def slow(_job):
        time.sleep(1.0)
        return "done"

    runner.submit("first", slow)
    try:
        runner.submit("second", slow)
    except RuntimeError as exc:
        assert "still running" in str(exc)
    else:
        raise AssertionError("expected the second submission to be refused")


def test_stage_status_reflects_artifacts_on_disk() -> None:
    """Stages report done only when their own artifact actually exists."""
    with TempProject() as project:
        empty = {stage["key"]: stage["done"] for stage in stage_status(project.config)}
        assert empty["data"] is False
        assert empty["train"] is False
        assert empty["evaluate"] is False

    with trained_project() as project:
        done = {stage["key"]: stage["done"] for stage in stage_status(project.config)}
        assert done["data"] is True
        assert done["train"] is True
        assert done["evaluate"] is True


def test_project_status_has_every_section_the_dashboard_renders() -> None:
    """The status payload is complete, so no panel renders as undefined."""
    with trained_project() as project:
        status = project_status(project.config)
        for key in ("project", "stages", "dataset", "versioning", "model", "runs", "api", "plots"):
            assert key in status, f"status payload is missing {key}"
        assert len(status["stages"]) == 7


def test_dashboard_routes_answer() -> None:
    """Every read-only dashboard route returns a usable response."""
    with trained_project() as project:
        client = create_ui_app(project.config).test_client()
        assert client.get("/").status_code == 200
        assert client.get("/health").get_json()["status"] == "ok"
        for path in ("/api/status", "/api/stages", "/api/runs", "/api/model-card",
                 "/api/sample-images", "/api/logs?source=local", "/api/containers",
                 "/api/deployment", "/api/jobs"):
            response = client.get(path)
            assert response.status_code == 200, f"{path} returned {response.status_code}"
            assert response.get_json() is not None


def test_dashboard_file_preview_cannot_escape_the_project() -> None:
    """The file preview is confined to the project root."""
    with trained_project() as project:
        client = create_ui_app(project.config).test_client()
        assert client.get("/api/file?path=../../etc/passwd").status_code == 403
        assert client.get("/api/file?path=configs/config.yaml").status_code == 200


def test_dashboard_actions_start_jobs() -> None:
    """Action endpoints accept the request and hand back a job id."""
    with trained_project() as project:
        app = create_ui_app(project.config)
        client = app.test_client()
        response = client.post("/api/actions/dataset-lock", json={})
        assert response.status_code == 202
        job_id = response.get_json()["job_id"]

        for _ in range(100):
            body = client.get(f"/api/jobs/{job_id}").get_json()
            if body["status"] != "running":
                break
            time.sleep(0.05)
        assert body["status"] == "succeeded"
        assert client.get("/api/jobs/nonexistent").status_code == 404


def test_dashboard_rejects_an_unknown_dvc_command() -> None:
    """Only an allow-list of DVC subcommands can be triggered from the browser."""
    with trained_project() as project:
        client = create_ui_app(project.config).test_client()
        response = client.post("/api/actions/dvc", json={"command": "rm -rf /"})
        assert response.status_code == 400
        assert "unknown dvc command" in response.get_json()["error"]
