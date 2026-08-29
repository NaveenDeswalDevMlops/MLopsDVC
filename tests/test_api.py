"""Tests for the inference service and its metrics (M2.1, M5.1).

These use Flask's test client, so every route is exercised through the real WSGI
stack — routing, error handlers, the before/after request hooks — without binding
a socket.
"""

from __future__ import annotations

import base64
import json

from helpers import TempProject, prepared_project, sample_image_bytes, trained_project
from mlops.serving.app import create_app
from mlops.serving.metrics import MetricsRegistry, parse_exposition


def _client(project: TempProject):
    """Build a Flask test client for a project.

    Args:
        project: The project whose configuration the app should use.

    Returns:
        A tuple of the app and its test client.
    """
    app = create_app(project.config)
    return app, app.test_client()


def test_health_is_independent_of_model_state() -> None:
    """Liveness succeeds even with no checkpoint, so pods are not restart-looped."""
    with TempProject() as project:
        _app, client = _client(project)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.get_json()["status"] == "ok"


def test_readiness_fails_without_a_checkpoint() -> None:
    """Readiness reports 503 and names the missing file."""
    with TempProject() as project:
        _app, client = _client(project)
        response = client.get("/ready")
        assert response.status_code == 503
        body = response.get_json()
        assert body["status"] == "not_ready"
        assert body["checks"]["checkpoint_present"] is False


def test_readiness_succeeds_once_a_model_is_present() -> None:
    """With a checkpoint the pod reports itself ready to serve."""
    with trained_project() as project:
        _app, client = _client(project)
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.get_json()["status"] == "ready"


def test_predict_accepts_a_multipart_upload() -> None:
    """A file upload returns a label, a confidence and a full distribution."""
    with trained_project() as project:
        _app, client = _client(project)
        response = client.post(
            "/predict",
            data={"file": (__import__("io").BytesIO(sample_image_bytes()), "cat.jpg")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200
        body = response.get_json()
        assert body["label"] in ("cat", "dog")
        assert 0.0 <= body["confidence"] <= 1.0
        assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
        assert body["latency_ms"] >= 0.0
        assert body["request_id"]


def test_predict_accepts_base64_json() -> None:
    """The JSON body form works identically to the multipart form."""
    with trained_project() as project:
        _app, client = _client(project)
        encoded = base64.b64encode(sample_image_bytes()).decode("ascii")
        response = client.post("/predict", json={"image_base64": encoded})
        assert response.status_code == 200
        assert response.get_json()["label"] in ("cat", "dog")


def test_predict_accepts_a_data_url_prefix() -> None:
    """A browser-style data URL is handled rather than rejected as bad base64."""
    with trained_project() as project:
        _app, client = _client(project)
        encoded = base64.b64encode(sample_image_bytes()).decode("ascii")
        response = client.post("/predict", json={"image_base64": f"data:image/jpeg;base64,{encoded}"})
        assert response.status_code == 200


def test_predict_rejects_bad_payloads_with_the_right_status() -> None:
    """Each class of bad request maps onto its own status and error code."""
    with trained_project() as project:
        _app, client = _client(project)

        undecodable = client.post(
            "/predict", data=b"not an image", content_type="image/jpeg"
        )
        assert undecodable.status_code == 422
        assert undecodable.get_json()["error"]["code"] == "invalid_image"

        wrong_type = client.post("/predict", data=b"hello", content_type="text/plain")
        assert wrong_type.status_code == 415
        assert wrong_type.get_json()["error"]["code"] == "unsupported_media_type"

        bad_base64 = client.post("/predict", json={"image_base64": "!!!not base64!!!"})
        assert bad_base64.status_code == 422

        empty_json = client.post("/predict", json={})
        assert empty_json.status_code == 422


def test_predict_returns_503_when_no_model_is_loaded() -> None:
    """Without a checkpoint the service refuses rather than inventing an answer."""
    with prepared_project() as project:
        _app, client = _client(project)
        response = client.post("/predict", json={"image_base64": base64.b64encode(sample_image_bytes()).decode()})
        assert response.status_code == 503
        assert response.get_json()["error"]["code"] == "model_not_loaded"


def test_batch_prediction_returns_one_result_per_image() -> None:
    """The batch endpoint preserves order and count."""
    with trained_project() as project:
        _app, client = _client(project)
        images = [
            {"image_base64": base64.b64encode(sample_image_bytes(colour)).decode("ascii")}
            for colour in ((200, 90, 40), (40, 60, 200), (120, 120, 120))
        ]
        response = client.post("/predict/batch", json={"images": images})
        assert response.status_code == 200
        body = response.get_json()
        assert body["count"] == 3
        assert len(body["results"]) == 3


def test_batch_rejects_a_malformed_body() -> None:
    """A batch without an images array is a validation error, not a 500."""
    with trained_project() as project:
        _app, client = _client(project)
        assert client.post("/predict/batch", json={}).status_code == 422
        assert client.post("/predict/batch", json={"images": "nope"}).status_code == 422
        assert client.post("/predict/batch", json={"images": []}).status_code == 422


def test_model_info_reports_provenance() -> None:
    """The model-info payload carries the input contract and training provenance."""
    with trained_project() as project:
        _app, client = _client(project)
        body = client.get("/model-info").get_json()
        assert body["classes"] == ["cat", "dog"]
        assert body["input"]["image_size"] == int(project.config.get("data.image_size"))
        assert body["training"]["trained_at"]
        assert body["versions"]["scikit-learn"]


def test_reload_picks_up_a_newly_trained_checkpoint() -> None:
    """POST /reload swaps in the model on disk without a restart."""
    with trained_project() as project:
        app, client = _client(project)
        before = app.predictor.model.metadata.trained_at

        from mlops.models import train

        train.run(project.config, use_mlflow=False)
        response = client.post("/reload")
        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "reloaded"
        assert body["previous_trained_at"] == before


def test_reload_without_a_checkpoint_reports_503() -> None:
    """Reloading when there is nothing to load fails cleanly."""
    with TempProject() as project:
        _app, client = _client(project)
        response = client.post("/reload")
        assert response.status_code == 503
        assert response.get_json()["error"]["code"] == "reload_failed"


def test_unknown_route_returns_json_not_html() -> None:
    """A 404 stays machine-readable so clients can parse every response."""
    with trained_project() as project:
        _app, client = _client(project)
        response = client.get("/does-not-exist")
        assert response.status_code == 404
        assert response.get_json()["error"]["code"] == "not_found"


def test_request_id_is_echoed_and_honoured() -> None:
    """A caller-supplied correlation id flows through to the response header."""
    with trained_project() as project:
        _app, client = _client(project)
        response = client.get("/health", headers={"X-Request-ID": "trace-me-123"})
        assert response.headers["X-Request-ID"] == "trace-me-123"
        assert response.get_json()["request_id"] == "trace-me-123"


def test_metrics_endpoint_is_valid_prometheus_exposition() -> None:
    """The metrics endpoint parses as Prometheus text and counts real requests."""
    with trained_project() as project:
        _app, client = _client(project)
        encoded = base64.b64encode(sample_image_bytes()).decode("ascii")
        for _ in range(3):
            client.post("/predict", json={"image_base64": encoded})

        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.mimetype == "text/plain"
        text = response.get_data(as_text=True)
        assert "# TYPE catsdogs_predictions_total counter" in text
        assert "catsdogs_inference_latency_seconds_bucket" in text

        series = parse_exposition(text)
        predictions = sum(
            value for name, value in series.items() if name.startswith("catsdogs_predictions_total")
        )
        assert predictions >= 3


def test_metrics_summary_totals_are_aggregated() -> None:
    """Series sharing a label value are summed, not overwritten."""
    with trained_project() as project:
        _app, client = _client(project)
        encoded = base64.b64encode(sample_image_bytes()).decode("ascii")
        for _ in range(4):
            client.post("/predict", json={"image_base64": encoded})
        client.post("/predict/batch", json={"images": [{"image_base64": encoded}]})

        summary = client.get("/metrics-summary").get_json()
        assert summary["predictions_total"] == 5
        assert summary["http_requests_total"] >= 5
        assert summary["http_by_status"]["200"] >= 5
        assert summary["model_ready"] is True


def test_metrics_registry_counts_and_quantiles() -> None:
    """The registry sums counters and returns sensible latency quantiles."""
    registry = MetricsRegistry(buckets=(0.01, 0.1, 1.0))
    for index in range(10):
        registry.record_prediction("dog", 0.8, latency_seconds=index / 100.0, endpoint="/predict")
    registry.record_error("invalid_image")

    snapshot = registry.snapshot()
    assert snapshot["predictions_by_label"]["dog"] == 10
    assert snapshot["errors_total"] == 1.0
    assert snapshot["latency_ms"]["p50"] > 0
    assert snapshot["latency_ms"]["p99"] >= snapshot["latency_ms"]["p50"]
    assert "catsdogs_errors_total{error_code=\"invalid_image\"} 1" in registry.render()


def test_exposition_escapes_label_values() -> None:
    """Quotes in a label value cannot break the exposition format."""
    registry = MetricsRegistry()
    registry.http_requests.inc(method="GET", path='/a"b', status="200")
    text = registry.render()
    assert '\\"' in text
    assert parse_exposition(text)
