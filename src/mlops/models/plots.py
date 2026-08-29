"""Plots logged with every run: learning curves, confusion matrix, ROC.

Matplotlib is forced onto the ``Agg`` backend at import time because this code runs
in CI and in containers with no display; a backend probe that opens a window is a
classic way for a headless pipeline to hang instead of fail.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  - must follow the backend selection
import numpy as np  # noqa: E402

from mlops.logging_setup import get_logger  # noqa: E402

_LOGGER = get_logger(__name__)

INK = "#12263A"
ACCENT = "#0B6E99"
SECOND = "#B4690E"
GRID = "#C7D2DC"


def _style(axis: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the shared plot styling.

    Args:
        axis: Axes to style.
        title: Plot title.
        xlabel: X axis label.
        ylabel: Y axis label.
    """
    axis.set_title(title, color=INK, fontsize=12, pad=12)
    axis.set_xlabel(xlabel, color=INK, fontsize=10)
    axis.set_ylabel(ylabel, color=INK, fontsize=10)
    axis.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    axis.tick_params(colors=INK, labelsize=9)
    for spine in axis.spines.values():
        spine.set_color(GRID)


def loss_curve(history: dict[str, list[float]], path: Path | str) -> Path:
    """Plot training and validation loss per epoch.

    Args:
        history: Mapping with ``train_loss`` and optionally ``val_loss``.
        path: Output PNG path.

    Returns:
        The written path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 3.8), dpi=130)
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    axis.plot(epochs, history.get("train_loss", []), color=ACCENT, linewidth=2, label="train loss")
    if history.get("val_loss"):
        axis.plot(epochs, history["val_loss"], color=SECOND, linewidth=2, label="val loss")
    _style(axis, "Loss per epoch", "epoch", "log loss")
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(destination, facecolor="white")
    plt.close(figure)
    return destination


def accuracy_curve(history: dict[str, list[float]], path: Path | str) -> Path:
    """Plot training and validation accuracy per epoch.

    Args:
        history: Mapping with ``train_accuracy`` and ``val_accuracy``.
        path: Output PNG path.

    Returns:
        The written path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 3.8), dpi=130)
    epochs = range(1, len(history.get("train_accuracy", [])) + 1)
    axis.plot(
        epochs, history.get("train_accuracy", []), color=ACCENT, linewidth=2, label="train accuracy"
    )
    if history.get("val_accuracy"):
        axis.plot(epochs, history["val_accuracy"], color=SECOND, linewidth=2, label="val accuracy")
    axis.set_ylim(0.0, 1.02)
    _style(axis, "Accuracy per epoch", "epoch", "accuracy")
    axis.legend(frameon=False, fontsize=9, loc="lower right")
    figure.tight_layout()
    figure.savefig(destination, facecolor="white")
    plt.close(figure)
    return destination


def confusion_matrix_plot(
    matrix: Sequence[Sequence[int]], class_names: Sequence[str], path: Path | str
) -> Path:
    """Render a labelled confusion matrix.

    Args:
        matrix: 2x2 counts, rows are true classes.
        class_names: Class names in label-index order.
        path: Output PNG path.

    Returns:
        The written path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(matrix, dtype=float)
    figure, axis = plt.subplots(figsize=(4.6, 4.2), dpi=130)
    axis.imshow(data, cmap="Blues", vmin=0.0)
    axis.set_xticks(range(len(class_names)), [f"pred {name}" for name in class_names])
    axis.set_yticks(range(len(class_names)), [f"true {name}" for name in class_names])
    total = max(data.sum(), 1.0)
    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            value = int(data[row, column])
            axis.text(
                column,
                row,
                f"{value}\n{value / total:.1%}",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if data[row, column] > data.max() / 2 else INK,
            )
    axis.set_title("Confusion matrix (test split)", color=INK, fontsize=12, pad=12)
    axis.tick_params(colors=INK, labelsize=9)
    figure.tight_layout()
    figure.savefig(destination, facecolor="white")
    plt.close(figure)
    return destination


def roc_curve_plot(
    fpr: Sequence[float], tpr: Sequence[float], auc: float, path: Path | str
) -> Path:
    """Plot the ROC curve.

    Args:
        fpr: False-positive rates.
        tpr: True-positive rates.
        auc: Area under the curve.
        path: Output PNG path.

    Returns:
        The written path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(4.8, 4.2), dpi=130)
    axis.plot(fpr, tpr, color=ACCENT, linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    axis.plot([0, 1], [0, 1], color=GRID, linewidth=1.2, linestyle="--", label="chance")
    _style(axis, "ROC curve (test split)", "false positive rate", "true positive rate")
    axis.legend(frameon=False, fontsize=9, loc="lower right")
    figure.tight_layout()
    figure.savefig(destination, facecolor="white")
    plt.close(figure)
    return destination


__all__ = ["accuracy_curve", "confusion_matrix_plot", "loss_curve", "roc_curve_plot"]
