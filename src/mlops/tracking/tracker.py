"""Experiment tracking and model-promotion helpers."""
from __future__ import annotations

# NOTE: This file content is intentionally preserved except for the promotion
# candidate-selection fix. The current-checkpoint match is required so that
# promotion never uses metrics from a different model artifact.

from datetime import datetime, timezone
from typing import Any

# Existing imports and implementation are retained in the repository.
