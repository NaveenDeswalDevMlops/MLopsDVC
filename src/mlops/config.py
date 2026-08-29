"""Single source of truth for configuration.

Values come from ``configs/config.yaml`` and may be overridden by ``MLOPS_*``
environment variables using ``__`` for nesting, e.g.::

    MLOPS_TRAINING__EPOCHS=40
    MLOPS_SERVING__PORT=9000

Resolution order: YAML file -> environment variables. Nothing else. Keeping the
override syntax mechanical means the Kubernetes ConfigMap, the Compose file and a
developer shell all speak the same language.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "MLOPS_"


def project_root() -> Path:
    """Return the repository root.

    Resolved from this file's location so the code works whether it is run from
    the source tree, an installed package or a container image.

    Returns:
        Absolute path to the project root.
    """
    override = os.environ.get("MLOPS_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` without mutating either.

    Args:
        base: Baseline mapping.
        overlay: Mapping whose values win.

    Returns:
        The merged mapping.
    """
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce(raw: str) -> Any:
    """Convert an environment string into a Python scalar.

    Args:
        raw: The raw environment value.

    Returns:
        A bool, int, float, list/dict (via JSON) or the original string.
    """
    lowered = raw.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith(("[", "{")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _env_overrides() -> dict[str, Any]:
    """Build a nested mapping from ``MLOPS_*`` environment variables.

    Returns:
        Nested overrides ready to merge over the YAML document.
    """
    overrides: dict[str, Any] = {}
    for name, value in os.environ.items():
        if not name.startswith(ENV_PREFIX) or name == "MLOPS_PROJECT_ROOT":
            continue
        path = name[len(ENV_PREFIX) :].lower().split("__")
        cursor = overrides
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):  # pragma: no cover - defensive
                break
        if isinstance(cursor, dict):
            cursor[path[-1]] = _coerce(value)
    return overrides


@dataclass(frozen=True)
class Config:
    """Effective configuration.

    Attributes:
        raw: The merged configuration document.
        root: Project root used to resolve every relative path.
    """

    raw: dict[str, Any] = field(default_factory=dict)
    root: Path = field(default_factory=project_root)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Read a value by dotted path.

        Args:
            dotted: Path such as ``training.epochs``.
            default: Returned when the path is absent.

        Returns:
            The configured value or ``default``.
        """
        cursor: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor

    def path(self, dotted: str) -> Path:
        """Read a configured path and resolve it against the project root.

        Args:
            dotted: Path such as ``paths.model_path``.

        Returns:
            An absolute path.

        Raises:
            KeyError: If the key is not configured.
        """
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"path {dotted!r} is not configured")
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else (self.root / candidate).resolve()

    def flat_params(self) -> dict[str, str]:
        """Flatten the whole document for experiment tracking.

        Returns:
            Dotted-key to stringified-value mapping.
        """
        flat: dict[str, str] = {}

        def walk(node: Any, prefix: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{prefix}.{key}" if prefix else str(key))
            else:
                flat[prefix] = json.dumps(node) if isinstance(node, (list, tuple)) else str(node)

        walk(self.raw, "")
        return flat

    def ensure_dirs(self) -> None:
        """Create every directory the pipeline writes into."""
        for key in (
            "paths.data_root",
            "paths.raw_dir",
            "paths.processed_dir",
            "paths.artifacts_dir",
            "paths.metrics_dir",
            "paths.plots_dir",
            "paths.runs_dir",
            "paths.logs_dir",
        ):
            self.path(key).mkdir(parents=True, exist_ok=True)


def load_config(config_path: Path | str | None = None) -> Config:
    """Load configuration from YAML and environment.

    Args:
        config_path: Optional explicit path to the YAML file.

    Returns:
        The effective configuration.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    root = project_root()
    path = Path(config_path) if config_path else Path(
        os.environ.get("MLOPS_CONFIG_FILE", root / "configs" / "config.yaml")
    )
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged = _deep_merge(document, _env_overrides())
    return Config(raw=merged, root=root)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return a process-wide cached configuration.

    Returns:
        The effective configuration.
    """
    return load_config()


def reset_config_cache() -> None:
    """Clear the cached configuration; used by tests that change the environment."""
    get_config.cache_clear()


__all__ = ["Config", "get_config", "load_config", "project_root", "reset_config_cache"]
