"""Data versioning: DVC when it is installed, content hashes always.

``dvc.yaml`` defines the real pipeline and ``make dvc-repro`` runs it. This module
is the thin layer the rest of the code and the dashboard talk to, and it does two
distinct jobs:

1. Shell out to DVC (``dvc status``, ``dvc add``, ``dvc repro``, ``dvc dag``) and
   report the result verbatim, including the failure when DVC is not installed.
2. Write ``data/dataset.lock.json`` — a sorted file-by-file SHA-256 index of the
   raw and processed trees, plus a single digest over all of it.

The lock file exists because the version identity of a dataset should be verifiable
from the repository alone. DVC keeps the *contents* in a cache that a fresh clone
has to pull from a remote; the lock keeps the *identity* in Git, so anyone can tell
whether two runs used the same data without access to the remote at all. The two
are complementary, and the dashboard shows both.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mlops.config import Config
from mlops.logging_setup import get_logger

_LOGGER = get_logger(__name__)

DVC_TIMEOUT = 900


@dataclass
class CommandResult:
    """Result of an external command.

    Attributes:
        command: The argv that was executed.
        returncode: Process exit status; ``-1`` when the binary is missing.
        stdout: Captured standard output.
        stderr: Captured standard error.
        available: Whether the binary was found at all.
    """

    command: str
    returncode: int
    stdout: str
    stderr: str
    available: bool = True

    @property
    def ok(self) -> bool:
        """Return ``True`` when the command ran and exited zero."""
        return self.available and self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping of this result."""
        return {
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "available": self.available,
            "ok": self.ok,
        }


def dvc_available() -> bool:
    """Report whether the ``dvc`` binary is on PATH.

    Returns:
        ``True`` when DVC can be executed.
    """
    return shutil.which("dvc") is not None


def run_dvc(config: Config, args: list[str], timeout: int = DVC_TIMEOUT) -> CommandResult:
    """Run a DVC subcommand from the project root.

    Args:
        config: Effective configuration, used for the working directory.
        args: Arguments after the ``dvc`` binary, e.g. ``["status"]``.
        timeout: Seconds before the command is killed.

    Returns:
        The captured result. A missing binary is reported, never raised, so the
        dashboard can render an actionable message instead of a stack trace.
    """
    command = ["dvc", *args]
    printable = " ".join(command)
    if not dvc_available():
        return CommandResult(
            command=printable,
            returncode=-1,
            stdout="",
            stderr="dvc is not installed. Install it with `pip install -r requirements.txt`.",
            available=False,
        )
    try:
        completed = subprocess.run(
            command,
            cwd=config.root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(printable, 124, "", f"timed out after {timeout}s")
    except OSError as exc:
        return CommandResult(printable, -1, "", str(exc), available=False)

    _LOGGER.info(
        "dvc command finished",
        extra={"command": printable, "returncode": completed.returncode},
    )
    return CommandResult(printable, completed.returncode, completed.stdout, completed.stderr)


def _hash_tree(root: Path, limit: int = 20_000) -> tuple[list[dict[str, Any]], str]:
    """Hash every file under a directory.

    Args:
        root: Directory to walk.
        limit: Safety cap on the number of files hashed.

    Returns:
        Tuple of per-file records and a single digest over all of them.
    """
    entries: list[dict[str, Any]] = []
    hasher = hashlib.sha256()
    if not root.is_dir():
        return entries, ""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if len(entries) >= limit:
            break
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = str(path.relative_to(root))
        entries.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
        hasher.update(f"{relative}:{digest}".encode())
    return entries, hasher.hexdigest()


def write_dataset_lock(config: Config) -> dict[str, Any]:
    """Compute and persist the dataset lock file.

    Args:
        config: Effective configuration.

    Returns:
        The lock document, also written to ``paths.dataset_lock``.
    """
    raw_dir = config.path("paths.raw_dir")
    processed_dir = config.path("paths.processed_dir")

    raw_entries, raw_digest = _hash_tree(raw_dir)
    processed_entries, processed_digest = _hash_tree(processed_dir)

    lock = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw": {
            "path": str(raw_dir.relative_to(config.root)) if raw_dir.exists() else str(raw_dir),
            "files": len(raw_entries),
            "bytes": sum(entry["bytes"] for entry in raw_entries),
            "digest": raw_digest,
        },
        "processed": {
            "path": str(processed_dir.relative_to(config.root))
            if processed_dir.exists()
            else str(processed_dir),
            "files": len(processed_entries),
            "bytes": sum(entry["bytes"] for entry in processed_entries),
            "digest": processed_digest,
        },
        "combined_digest": hashlib.sha256(f"{raw_digest}:{processed_digest}".encode()).hexdigest(),
    }

    lock_path = config.path("paths.dataset_lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    _LOGGER.info(
        "dataset lock written",
        extra={
            "raw_files": lock["raw"]["files"],
            "processed_files": lock["processed"]["files"],
            "digest": lock["combined_digest"][:12],
        },
    )
    return lock


def read_dataset_lock(config: Config) -> dict[str, Any] | None:
    """Read the persisted dataset lock.

    Args:
        config: Effective configuration.

    Returns:
        The lock document, or ``None`` when it has not been written yet.
    """
    lock_path = config.path("paths.dataset_lock")
    if not lock_path.is_file():
        return None
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def versioning_status(config: Config) -> dict[str, Any]:
    """Summarise the state of data versioning for the dashboard.

    Args:
        config: Effective configuration.

    Returns:
        DVC availability and repo state, the lock file, and Git metadata.
    """
    dvc_repo = (config.root / ".dvc").is_dir()
    status: dict[str, Any] = {
        "dvc_installed": dvc_available(),
        "dvc_initialised": dvc_repo,
        "dvc_yaml": (config.root / "dvc.yaml").is_file(),
        "dvc_lock_file": (config.root / "dvc.lock").is_file(),
        "lock": read_dataset_lock(config),
        "git": git_status(config),
    }
    if status["dvc_installed"] and dvc_repo:
        result = run_dvc(config, ["status"], timeout=120)
        status["dvc_status"] = result.to_dict()
    return status


def git_status(config: Config) -> dict[str, Any]:
    """Collect the Git facts the dashboard shows next to the data version.

    Args:
        config: Effective configuration.

    Returns:
        Branch, commit, dirty flag and recent commit subjects.
    """
    def _git(args: list[str]) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=config.root,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return completed.stdout.strip() if completed.returncode == 0 else ""

    if not (config.root / ".git").exists():
        return {"repository": False}
    log = _git(["log", "-5", "--pretty=%h %s"])
    return {
        "repository": True,
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["rev-parse", "--short", "HEAD"]),
        "dirty": bool(_git(["status", "--porcelain"])),
        "recent": [line for line in log.splitlines() if line],
    }


__all__ = [
    "CommandResult",
    "dvc_available",
    "git_status",
    "read_dataset_lock",
    "run_dvc",
    "versioning_status",
    "write_dataset_lock",
]
