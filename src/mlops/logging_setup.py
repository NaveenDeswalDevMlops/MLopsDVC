"""Structured JSON logging with request correlation.

Every component logs one JSON object per line to stderr *and* to a rotating file.
Stderr is captured by ``docker logs`` and ``kubectl logs`` exactly like stdout; the
file is what the dashboard tails when it runs outside a cluster. Both carry the
same records, so the monitoring view is identical in either deployment.

Logs go to stderr rather than stdout so that stdout stays a clean channel for
command results. ``python -m mlops.cli train | jq .test_accuracy`` has to work, and
it cannot if log lines are interleaved with the result document.

Payload bytes are never logged. :func:`safe_payload_meta` produces size, declared
type and a short digest instead, which is enough to correlate a request with an
image without storing the image.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "taskName", "thread", "threadName",
}

_CONFIGURED = False


def new_request_id() -> str:
    """Generate a short correlation id.

    Returns:
        A 12-character hex id.
    """
    return uuid.uuid4().hex[:12]


def set_request_id(request_id: str) -> None:
    """Bind a correlation id to the current context.

    Args:
        request_id: The id to bind.
    """
    _REQUEST_ID.set(request_id)


def get_request_id() -> str:
    """Return the correlation id bound to the current context.

    Returns:
        The bound id, or ``"-"`` when none is set.
    """
    return _REQUEST_ID.get()


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise one record.

        Args:
            record: The record to render.

        Returns:
            A JSON string.
        """
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": get_request_id(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                try:
                    json.dumps(value)
                except (TypeError, ValueError):
                    value = str(value)
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    force: bool = False,
) -> None:
    """Install the JSON handlers on the root logger.

    Args:
        level: Logging level name.
        log_file: Optional file that receives the same records.
        force: Reconfigure even if logging was already set up.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = JsonFormatter()
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        rotating.setFormatter(formatter)
        root.addHandler(rotating)

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("werkzeug").setLevel(
        logging.WARNING if os.environ.get("MLOPS_QUIET_HTTP", "1") == "1" else logging.INFO
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Logger name, normally ``__name__``.

    Returns:
        The logger.
    """
    return logging.getLogger(name)


def safe_payload_meta(payload: bytes, content_type: str | None = None) -> dict[str, Any]:
    """Describe a request payload without retaining any of its bytes.

    Args:
        payload: The raw bytes received.
        content_type: The declared content type, if any.

    Returns:
        Size, declared type and a truncated SHA-256 digest.
    """
    return {
        "bytes": len(payload),
        "content_type": content_type or "unknown",
        "sha256_12": hashlib.sha256(payload).hexdigest()[:12],
    }


def read_jsonl_tail(path: Path | str, limit: int = 200) -> list[dict[str, Any]]:
    """Read the last ``limit`` JSON lines from a log file.

    Args:
        path: File to read.
        limit: Maximum number of records to return.

    Returns:
        Parsed records, oldest first. Unparseable lines are wrapped rather than
        dropped, because a crash trace is exactly when the log matters most.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return []
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            records.append(parsed if isinstance(parsed, dict) else {"message": str(parsed)})
        except json.JSONDecodeError:
            records.append({"level": "RAW", "logger": "-", "message": line, "ts": ""})
    return records


__all__ = [
    "configure_logging",
    "get_logger",
    "get_request_id",
    "new_request_id",
    "read_jsonl_tail",
    "safe_payload_meta",
    "set_request_id",
]
