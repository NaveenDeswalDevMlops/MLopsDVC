"""Run long dashboard actions in the background and stream their logs.

Training takes tens of seconds. A button that blocks an HTTP request for that long
looks broken, times out behind a proxy and gives the user nothing to look at while
they wait. So every action that is not instant becomes a :class:`Job`: the request
returns a job id immediately, the work runs on a worker thread, and the browser
polls for status and captured log lines.

A log handler is attached for the duration of each job, which means the dashboard
shows the same structured records the service writes to disk — the actual training
log, not a progress bar invented by the UI.

Only one job runs at a time. The stages mutate shared files (the dataset, the
checkpoint), and letting two run concurrently would produce a corrupted artifact
and an unreproducible run.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from mlops.logging_setup import JsonFormatter, get_logger

_LOGGER = get_logger(__name__)

MAX_LOG_LINES = 600
MAX_HISTORY = 40


@dataclass
class Job:
    """One background action.

    Attributes:
        job_id: Unique identifier.
        name: Action name, e.g. ``train``.
        status: ``running``, ``succeeded`` or ``failed``.
        created_at: ISO-8601 UTC creation time.
        finished_at: ISO-8601 UTC completion time.
        duration_seconds: Wall-clock duration.
        result: Return value of the action, when it succeeded.
        error: Error message, when it failed.
        logs: Captured log lines.
    """

    job_id: str
    name: str
    status: str = "running"
    created_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0
    result: Any = None
    error: str = ""
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))

    def to_dict(self, include_logs: bool = True) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of this job.

        Args:
            include_logs: Whether to include the captured log lines.

        Returns:
            The job summary.
        """
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds, 2),
            "result": self.result,
            "error": self.error,
        }
        if include_logs:
            payload["logs"] = list(self.logs)
        return payload


class _JobLogHandler(logging.Handler):
    """Capture log records emitted while a job runs."""

    def __init__(self, job: Job) -> None:
        """Attach the handler to a job.

        Args:
            job: The job that receives the records.
        """
        super().__init__(level=logging.INFO)
        self._job = job
        self.setFormatter(JsonFormatter())

    def emit(self, record: logging.LogRecord) -> None:
        """Append one formatted record to the job's log buffer.

        Args:
            record: The record to capture.
        """
        try:
            self._job.logs.append(self.format(record))
        except Exception:  # noqa: BLE001 - a logging failure must not kill the job
            pass


class JobRunner:
    """Serialised background execution of dashboard actions."""

    def __init__(self) -> None:
        """Create an empty runner."""
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque(maxlen=MAX_HISTORY)
        self._current: str | None = None

    @property
    def busy(self) -> bool:
        """Return ``True`` while a job is running."""
        with self._lock:
            if self._current is None:
                return False
            return self._jobs[self._current].status == "running"

    def submit(self, name: str, action: Callable[[Job], Any]) -> Job:
        """Start an action on a worker thread.

        Args:
            name: Action name shown in the UI.
            action: Callable receiving the job, returning a JSON-serialisable value.

        Returns:
            The created job.

        Raises:
            RuntimeError: If another job is still running.
        """
        with self._lock:
            if self._current is not None and self._jobs[self._current].status == "running":
                running = self._jobs[self._current]
                raise RuntimeError(
                    f"'{running.name}' is still running; wait for it to finish before starting "
                    f"'{name}'"
                )
            job = Job(
                job_id=uuid.uuid4().hex[:12],
                name=name,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            self._current = job.job_id
            for stale in [key for key in self._jobs if key not in self._order]:
                del self._jobs[stale]

        thread = threading.Thread(
            target=self._run, args=(job, action), name=f"job-{name}", daemon=True
        )
        thread.start()
        return job

    def _run(self, job: Job, action: Callable[[Job], Any]) -> None:
        """Execute one job, capturing logs and errors.

        Args:
            job: The job being executed.
            action: The callable to run.
        """
        handler = _JobLogHandler(job)
        root = logging.getLogger()
        root.addHandler(handler)
        started = time.perf_counter()
        job.logs.append(f'{{"level": "INFO", "message": "job {job.name} started"}}')
        try:
            job.result = action(job)
            job.status = "succeeded"
        except Exception as exc:  # noqa: BLE001 - every failure belongs in the UI
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.logs.append(f'{{"level": "ERROR", "message": {job.error!r}}}')
            for line in traceback.format_exc().splitlines()[-12:]:
                job.logs.append(f'{{"level": "ERROR", "message": {line!r}}}')
            _LOGGER.error("job failed", extra={"job": job.name, "error": job.error})
        finally:
            job.duration_seconds = time.perf_counter() - started
            job.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            root.removeHandler(handler)
            _LOGGER.info(
                "job finished",
                extra={
                    "job": job.name,
                    "status": job.status,
                    "seconds": round(job.duration_seconds, 2),
                },
            )

    def get(self, job_id: str) -> Job | None:
        """Return one job by id.

        Args:
            job_id: The identifier.

        Returns:
            The job, or ``None`` when it is unknown or has been evicted.
        """
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent jobs, newest first.

        Args:
            limit: Maximum number of jobs.

        Returns:
            Job summaries without their log bodies.
        """
        with self._lock:
            ids = list(self._order)[-limit:][::-1]
            return [self._jobs[key].to_dict(include_logs=False) for key in ids if key in self._jobs]


__all__ = ["Job", "JobRunner"]
