"""The inference REST API.

Endpoints:

===========================  ======================================================
``GET  /health``             Liveness. Independent of model state on purpose.
``GET  /ready``              Readiness. 503 until a checkpoint is loaded.
``POST /predict``            One image, as multipart upload or base64 JSON.
``POST /predict/batch``      Several base64 images in one call.
``GET  /model-info``         Identity and provenance of the loaded model.
``POST /reload``             Re-read the checkpoint without restarting.
``GET  /metrics``            Prometheus exposition text.
``GET  /``                   Endpoint index, so a bare curl is not a 404.
===========================  ======================================================

Liveness and readiness are separate because they answer different questions. A pod
whose checkpoint is missing should be taken out of the load-balancer rotation
(readiness fails) but must not be restarted in a loop (liveness passes), since
restarting cannot conjure a model file.

The checkpoint is loaded once at start-up. A missing checkpoint is not fatal: the
service starts, reports ``/ready`` 503 and says why, which is far easier to
diagnose in Kubernetes than a crash-looping container.
"""

from __future__ import annotations

import base64
import binascii
import time
from typing import Any

from flask import Flask, Response, g, jsonify, request

from mlops.config import Config, get_config
from mlops.logging_setup import (
    configure_logging,
    get_logger,
    new_request_id,
    set_request_id,
)
from mlops.models.model import ModelError
from mlops.serving.metrics import MetricsRegistry
from mlops.serving.predictor import (
    BatchSizeError,
    InvalidImageError,
    PayloadTooLargeError,
    Predictor,
    UnsupportedMediaTypeError,
)

_LOGGER = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

ERROR_STATUS: dict[type[Exception], tuple[int, str]] = {
    UnsupportedMediaTypeError: (415, "unsupported_media_type"),
    PayloadTooLargeError: (413, "payload_too_large"),
    InvalidImageError: (422, "invalid_image"),
    BatchSizeError: (422, "invalid_batch"),
}


def _error(status: int, code: str, message: str, **details: Any) -> tuple[Response, int]:
    """Build a consistent error response.

    Args:
        status: HTTP status code.
        code: Stable machine-readable code.
        message: Human-readable explanation.
        **details: Extra structured context; never request payload bytes.

    Returns:
        The Flask response and its status code.
    """
    body = {"error": {"code": code, "message": message, "details": details or None}}
    return jsonify(body), status


def create_app(config: Config | None = None) -> Flask:
    """Build the Flask application.

    Args:
        config: Effective configuration; loaded from disk when omitted.

    Returns:
        The configured application. ``app.metrics`` and ``app.predictor`` are
        attached so tests and the dashboard can inspect them directly.
    """
    config = config or get_config()
    config.ensure_dirs()
    configure_logging(
        level=str(config.get("logging.level", "INFO")),
        log_file=config.path("monitoring.log_file"),
    )

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(config.get("serving.max_upload_bytes", 5_242_880)) * 2
    metrics = MetricsRegistry(buckets=config.get("serving.latency_buckets"))
    app.metrics = metrics  # type: ignore[attr-defined]
    app.mlops_config = config  # type: ignore[attr-defined]
    app.started_at = time.time()  # type: ignore[attr-defined]

    try:
        predictor: Predictor | None = Predictor.from_config(config)
        metrics.set_model(
            version=predictor.model_version,
            model_type=predictor.model.metadata.model_type,
            ready=True,
        )
        _LOGGER.info(
            "model loaded",
            extra={
                "path": str(config.path("paths.model_path")),
                "model_type": predictor.model.metadata.model_type,
                "version": predictor.model_version,
            },
        )
    except ModelError as exc:
        predictor = None
        metrics.set_model(version="none", model_type="none", ready=False)
        _LOGGER.warning(
            "starting without a model; /ready will report 503", extra={"error": str(exc)}
        )
    app.predictor = predictor  # type: ignore[attr-defined]

    metrics_path = str(config.get("serving.metrics_path", "/metrics"))

    # -- request lifecycle ------------------------------------------------

    @app.before_request
    def _begin() -> None:
        """Bind a correlation id and start the request timer."""
        request_id = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        set_request_id(request_id)
        g.request_id = request_id
        g.started = time.perf_counter()

    @app.after_request
    def _finish(response: Response) -> Response:
        """Log and count the exchange, and echo the correlation id."""
        duration_ms = (time.perf_counter() - getattr(g, "started", time.perf_counter())) * 1000.0
        response.headers[REQUEST_ID_HEADER] = getattr(g, "request_id", "-")
        if request.path != metrics_path:
            metrics.record_http(request.method, request.path, response.status_code)
            _LOGGER.info(
                "http request",
                extra={
                    "method": request.method,
                    "path": request.path,
                    "status": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                    "remote": request.remote_addr,
                },
            )
        return response

    @app.errorhandler(413)
    def _too_large(_exc: Any) -> tuple[Response, int]:
        """Translate Werkzeug's own size rejection into the API error shape."""
        metrics.record_error("payload_too_large")
        return _error(413, "payload_too_large", "the uploaded file exceeds the size limit")

    @app.errorhandler(404)
    def _not_found(_exc: Any) -> tuple[Response, int]:
        """Return a JSON 404 rather than Werkzeug's HTML page."""
        return _error(404, "not_found", f"no route for {request.path}")

    @app.errorhandler(500)
    def _server_error(exc: Any) -> tuple[Response, int]:
        """Return a JSON 500 and record it."""
        metrics.record_error("internal_error")
        _LOGGER.exception("unhandled error", extra={"path": request.path})
        return _error(500, "internal_error", str(exc))

    # -- helpers -----------------------------------------------------------

    def _require_predictor() -> Predictor:
        """Return the loaded predictor.

        Returns:
            The predictor.

        Raises:
            RuntimeError: Never; a missing model is turned into a 503 response by
                the caller, which checks ``app.predictor`` first.
        """
        assert app.predictor is not None  # noqa: S101 - guarded by the caller
        return app.predictor  # type: ignore[return-value]

    def _read_single() -> tuple[bytes, str | None]:
        """Extract image bytes from a multipart upload or a base64 JSON body.

        Returns:
            Tuple of raw bytes and the declared content type.

        Raises:
            InvalidImageError: If no usable field is present.
            UnsupportedMediaTypeError: If the content type is neither supported form.
        """
        if request.files:
            for field in ("file", "image"):
                if field in request.files:
                    upload = request.files[field]
                    return upload.read(), upload.mimetype
            raise InvalidImageError("multipart request must contain a 'file' or 'image' field")

        if request.is_json:
            document = request.get_json(silent=True) or {}
            encoded = document.get("image_base64") or document.get("image")
            if not encoded:
                raise InvalidImageError("JSON body must contain 'image_base64'")
            if isinstance(encoded, str) and encoded.startswith("data:"):
                encoded = encoded.split(",", 1)[-1]
            try:
                return base64.b64decode(encoded, validate=True), document.get("content_type")
            except (binascii.Error, ValueError) as exc:
                raise InvalidImageError(f"'image_base64' is not valid base64: {exc}") from exc

        if request.data:
            return request.data, request.content_type

        raise UnsupportedMediaTypeError(
            "send multipart/form-data with a 'file' field, or JSON with 'image_base64'"
        )

    def _handle_prediction_error(exc: Exception) -> tuple[Response, int]:
        """Map a predictor exception onto an HTTP response.

        Args:
            exc: The raised exception.

        Returns:
            The error response.
        """
        status, code = ERROR_STATUS.get(type(exc), (422, "prediction_failed"))
        metrics.record_error(code)
        _LOGGER.warning("prediction rejected", extra={"code": code, "status": status})
        return _error(status, code, str(exc))

    # -- routes ------------------------------------------------------------

    @app.get("/")
    def index() -> Response:
        """List the available endpoints.

        Returns:
            A JSON index of the API.
        """
        return jsonify(
            {
                "service": str(config.get("project.name", "mlops-catsdogs")),
                "version": str(config.get("project.version", "1.0.0")),
                "endpoints": [
                    "GET /health",
                    "GET /ready",
                    "POST /predict",
                    "POST /predict/batch",
                    "GET /model-info",
                    "POST /reload",
                    f"GET {metrics_path}",
                ],
            }
        )

    @app.get("/health")
    def health() -> Response:
        """Report that the process is alive.

        Returns:
            Liveness payload with uptime.
        """
        return jsonify(
            {
                "status": "ok",
                "service": str(config.get("project.name", "mlops-catsdogs")),
                "version": str(config.get("project.version", "1.0.0")),
                "uptime_seconds": round(metrics.uptime_seconds(), 2),
                "request_id": getattr(g, "request_id", "-"),
            }
        )

    @app.get("/ready")
    def ready() -> tuple[Response, int]:
        """Report whether the service can actually serve predictions.

        Returns:
            Readiness payload, with status 503 when no checkpoint is loaded.
        """
        checkpoint = config.path("paths.model_path")
        checks = {
            "model_loaded": app.predictor is not None,
            "checkpoint_present": checkpoint.is_file(),
        }
        is_ready = all(checks.values())
        body = {
            "status": "ready" if is_ready else "not_ready",
            "checks": checks,
            "model_version": app.predictor.model_version if app.predictor else None,
            "checkpoint": str(checkpoint),
            "request_id": getattr(g, "request_id", "-"),
        }
        return jsonify(body), (200 if is_ready else 503)

    @app.post("/predict")
    def predict() -> tuple[Response, int] | Response:
        """Classify one image.

        Returns:
            The prediction, or an error response.
        """
        if app.predictor is None:
            metrics.record_error("model_not_loaded")
            return _error(503, "model_not_loaded", "no checkpoint is loaded; train one first")
        try:
            payload, content_type = _read_single()
            result = _require_predictor().predict_bytes(payload, content_type=content_type)
        except (
            InvalidImageError,
            PayloadTooLargeError,
            UnsupportedMediaTypeError,
            BatchSizeError,
        ) as exc:
            return _handle_prediction_error(exc)

        metrics.record_prediction(
            label=result.label,
            confidence=result.confidence,
            latency_seconds=result.latency_ms / 1000.0,
            endpoint="/predict",
        )
        body = result.to_dict()
        body.update(
            {
                "model_version": _require_predictor().model_version,
                "model_type": _require_predictor().model.metadata.model_type,
                "request_id": getattr(g, "request_id", "-"),
            }
        )
        return jsonify(body)

    @app.post("/predict/batch")
    def predict_batch() -> tuple[Response, int] | Response:
        """Classify several base64-encoded images.

        Returns:
            One result per input, or an error response.
        """
        if app.predictor is None:
            metrics.record_error("model_not_loaded")
            return _error(503, "model_not_loaded", "no checkpoint is loaded; train one first")
        document = request.get_json(silent=True)
        if not isinstance(document, dict) or "images" not in document:
            metrics.record_error("validation_error")
            return _error(422, "validation_error", "JSON body must contain an 'images' array")

        images = document["images"]
        if not isinstance(images, list):
            metrics.record_error("validation_error")
            return _error(422, "validation_error", "'images' must be an array")

        payloads: list[bytes] = []
        for item in images:
            encoded = item.get("image_base64") if isinstance(item, dict) else item
            if not isinstance(encoded, str):
                metrics.record_error("validation_error")
                return _error(422, "validation_error", "each item needs an 'image_base64' string")
            try:
                payloads.append(base64.b64decode(encoded, validate=True))
            except (binascii.Error, ValueError) as exc:
                metrics.record_error("validation_error")
                return _error(422, "validation_error", f"invalid base64: {exc}")

        started = time.perf_counter()
        try:
            results = _require_predictor().predict_batch(payloads)
        except (
            BatchSizeError,
            InvalidImageError,
            PayloadTooLargeError,
            UnsupportedMediaTypeError,
        ) as exc:
            return _handle_prediction_error(exc)

        for result in results:
            metrics.record_prediction(
                label=result.label,
                confidence=result.confidence,
                latency_seconds=result.latency_ms / 1000.0,
                endpoint="/predict/batch",
            )
        return jsonify(
            {
                "count": len(results),
                "results": [result.to_dict() for result in results],
                "total_latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "request_id": getattr(g, "request_id", "-"),
            }
        )

    @app.get("/model-info")
    def model_info() -> tuple[Response, int] | Response:
        """Describe the model being served.

        Returns:
            Model identity and provenance, or 503 when nothing is loaded.
        """
        if app.predictor is None:
            return _error(503, "model_not_loaded", "no checkpoint is loaded; train one first")
        body = _require_predictor().info()
        body["request_id"] = getattr(g, "request_id", "-")
        return jsonify(body)

    @app.post("/reload")
    def reload_model() -> tuple[Response, int] | Response:
        """Re-read the checkpoint from disk without restarting the process.

        The model is loaded once at start-up, so retraining alone changes nothing
        for clients — the running process keeps serving the weights it already has.
        In Kubernetes the answer is a rollout; in development, restarting the
        server to see a new model is friction that makes people skip the check.
        This endpoint reloads in place and reports the identity of what is now
        being served, so a train-then-verify loop is two clicks.

        Returns:
            The new model identity, or 503 when the checkpoint cannot be loaded.
        """
        try:
            new_predictor = Predictor.from_config(config)
        except ModelError as exc:
            metrics.record_error("reload_failed")
            _LOGGER.warning("model reload failed", extra={"error": str(exc)})
            return _error(503, "reload_failed", str(exc))

        previous = (
            app.predictor.model.metadata.trained_at if app.predictor else None
        )
        app.predictor = new_predictor  # type: ignore[attr-defined]
        metrics.set_model(
            version=new_predictor.model_version,
            model_type=new_predictor.model.metadata.model_type,
            ready=True,
        )
        _LOGGER.info(
            "model reloaded",
            extra={
                "previous_trained_at": previous,
                "trained_at": new_predictor.model.metadata.trained_at,
            },
        )
        return jsonify(
            {
                "status": "reloaded",
                "previous_trained_at": previous,
                "model": new_predictor.info(),
                "request_id": getattr(g, "request_id", "-"),
            }
        )

    @app.get(metrics_path)
    def prometheus_metrics() -> Response:
        """Expose metrics in Prometheus text format.

        Returns:
            The exposition response.
        """
        return Response(metrics.render(), mimetype="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/metrics-summary")
    def metrics_summary() -> Response:
        """Expose a compact JSON summary of the metrics for the dashboard.

        Returns:
            The summary response.
        """
        return jsonify(metrics.snapshot())

    _LOGGER.info(
        "api ready",
        extra={"model_loaded": app.predictor is not None, "metrics_path": metrics_path},
    )
    return app


def main() -> int:
    """Run the API with the built-in server.

    Returns:
        Process exit status.
    """
    config = get_config()
    app = create_app(config)
    host = str(config.get("serving.host", "0.0.0.0"))
    port = int(config.get("serving.port", 8000))
    _LOGGER.info("starting api", extra={"host": host, "port": port})
    app.run(host=host, port=port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["create_app", "main"]
