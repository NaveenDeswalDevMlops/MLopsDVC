"""Tests for training, evaluation and experiment tracking (M1.2, M1.3)."""

from __future__ import annotations

import numpy as np

from helpers import TempProject, prepared_project, read_json, trained_project
from mlops.models import evaluate, train
from mlops.models.evaluate import compute_metrics
from mlops.tracking.tracker import Tracker, get_run, list_runs, register_best_model


def test_training_produces_a_checkpoint_and_stepped_metrics() -> None:
    """Training writes a model and records one metric point per epoch."""
    with prepared_project(**{"training.epochs": 4}) as project:
        result = train.run(project.config, use_mlflow=False)
        assert project.config.path("paths.model_path").is_file()
        assert result.epochs_run == 4
        for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy"):
            assert len(result.history[key]) == 4, f"{key} has no per-epoch series"
        assert 1 <= result.best_epoch <= 4


def test_training_history_is_persisted() -> None:
    """The epoch history is written to disk for the dashboard to read."""
    with prepared_project(**{"training.epochs": 3}) as project:
        train.run(project.config, use_mlflow=False)
        history = read_json(project.config.path("paths.history_path"))
        assert len(history["history"]["val_accuracy"]) == 3
        assert history["epochs_run"] == 3


def test_early_stopping_halts_before_the_epoch_budget() -> None:
    """With a patience of one and no improvement, training stops early."""
    with prepared_project(
        **{
            "training.epochs": 40,
            "training.early_stopping.enabled": True,
            "training.early_stopping.patience": 1,
            "training.early_stopping.min_delta": 0.9,  # nothing can clear this
        }
    ) as project:
        result = train.run(project.config, use_mlflow=False)
        assert result.epochs_run < 40, "early stopping never fired"


def test_the_best_epoch_is_the_one_that_is_saved() -> None:
    """The checkpoint scores at least as well as the final epoch on validation."""
    with prepared_project(**{"training.epochs": 6}) as project:
        result = train.run(project.config, use_mlflow=False)
        best_val = max(result.history["val_accuracy"])
        assert result.final_metrics["val_accuracy"] >= best_val - 1e-9, (
            "the saved model is worse than the best epoch, so checkpointing is wrong"
        )


def test_evaluation_writes_the_baseline_and_its_artifacts() -> None:
    """Evaluation produces the baseline, the report, the plots and the model card."""
    with trained_project() as project:
        baseline = read_json(project.config.path("paths.baseline_path"))
        assert baseline["split"] == "test"
        assert 0.0 <= baseline["metrics"]["accuracy"] <= 1.0
        assert len(baseline["confusion_matrix"]) == 2

        plots = project.config.path("paths.plots_dir")
        for name in ("loss_curve.png", "accuracy_curve.png", "confusion_matrix.png", "roc_curve.png"):
            path = plots / name
            assert path.is_file() and path.stat().st_size > 0, f"{name} was not written"

        card = project.config.path("paths.model_card").read_text(encoding="utf-8")
        assert "Model card" in card and "Test-split metrics" in card
        assert "Intended use and limits" in card


def test_confusion_matrix_totals_match_the_sample_count() -> None:
    """Every evaluated image appears exactly once in the confusion matrix."""
    with trained_project() as project:
        result = evaluate.run(project.config, use_mlflow=False)
        total = sum(sum(row) for row in result.confusion)
        assert total == int(result.metrics["samples"])


def test_compute_metrics_on_a_known_case() -> None:
    """Metrics are correct for a hand-checked example."""
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.6, 0.8, 0.4])
    metrics = compute_metrics(labels, probabilities, threshold=0.5)
    # Predictions: 0, 1, 1, 0 -> one true negative, one false positive,
    # one true positive, one false negative.
    assert abs(metrics["accuracy"] - 0.5) < 1e-9
    assert abs(metrics["precision"] - 0.5) < 1e-9
    assert abs(metrics["recall"] - 0.5) < 1e-9
    assert abs(metrics["f1"] - 0.5) < 1e-9
    assert metrics["samples"] == 4.0


def test_compute_metrics_survives_a_single_class_split() -> None:
    """A degenerate split reports ROC-AUC 0.5 instead of raising."""
    metrics = compute_metrics(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]), 0.5)
    assert metrics["roc_auc"] == 0.5
    assert metrics["accuracy"] == 1.0


def test_tracker_records_params_metrics_and_artifacts() -> None:
    """A run record captures everything the experiment view needs."""
    with TempProject() as project:
        artifact = project.root / "note.txt"
        artifact.write_text("hello", encoding="utf-8")
        with Tracker(project.config, run_name="unit", tags={"stage": "test"}, use_mlflow=False) as run:
            run.log_params({"epochs": 3, "model": "logreg"})
            run.log_metric("accuracy", 0.9, step=1)
            run.log_metric("accuracy", 0.95, step=2)
            run.log_artifact(artifact, artifact_subdir="misc")
            run_id = run.record.run_id

        record = get_run(project.config, run_id)
        assert record is not None
        assert record["status"] == "FINISHED"
        assert record["params"]["epochs"] == "3"
        assert record["metrics"]["accuracy"] == 0.95
        assert len(record["metric_history"]["accuracy"]) == 2
        assert "misc/note.txt" in record["artifacts"]
        assert (project.config.path("paths.runs_dir") / run_id / "artifacts" / "misc" / "note.txt").is_file()


def test_tracker_marks_a_failed_run_as_failed() -> None:
    """An exception inside the run block is recorded, not swallowed."""
    with TempProject() as project:
        run_id = ""
        try:
            with Tracker(project.config, run_name="boom", use_mlflow=False) as run:
                run_id = run.record.run_id
                raise ValueError("intentional")
        except ValueError:
            pass
        record = get_run(project.config, run_id)
        assert record is not None and record["status"] == "FAILED"


def test_missing_artifacts_do_not_break_a_run() -> None:
    """Logging a file that does not exist warns rather than raising."""
    with TempProject() as project, Tracker(project.config, run_name="missing", use_mlflow=False) as run:
        run.log_artifact(project.root / "absent.png")
        assert run.record.artifacts == []


def test_runs_are_listed_newest_first() -> None:
    """The run list is ordered so the dashboard shows the latest work at the top."""
    with TempProject() as project:
        for name in ("first", "second", "third"):
            with Tracker(project.config, run_name=name, use_mlflow=False):
                pass
        runs = list_runs(project.config)
        assert len(runs) == 3
        assert [run["started_at"] for run in runs] == sorted(
            [run["started_at"] for run in runs], reverse=True
        )


def test_promotion_gate_accepts_and_rejects_on_the_threshold() -> None:
    """The gate promotes above the floor and refuses below it, recording why."""
    with trained_project() as project:
        accuracy = read_json(project.config.path("paths.baseline_path"))["metrics"]["accuracy"]

        project.config.raw["tracking"]["promote_min_accuracy"] = max(0.0, accuracy - 0.01)
        accepted = register_best_model(project.config)
        assert accepted["promoted"] is True
        assert "accuracy" in accepted["reason"]

        # Deliberately unreachable, and not clipped to 1.0: the fixture data is
        # separable enough to score a perfect 1.0, so a clipped threshold would be
        # met rather than missed and the test would prove nothing.
        project.config.raw["tracking"]["promote_min_accuracy"] = accuracy + 0.5
        rejected = register_best_model(project.config)
        assert rejected["promoted"] is False
        decision = read_json(project.config.path("paths.metrics_dir") / "promotion.json")
        assert decision["promoted"] is False


def test_promotion_refuses_when_the_checkpoint_is_not_the_scored_one() -> None:
    """A strong run cannot promote a different, weaker checkpoint.

    Train once, evaluate, then train again without evaluating: the best recorded
    run still describes the first model while the file on disk is the second. The
    gate must refuse, because otherwise it green-lights an artifact it never
    scored.
    """
    from mlops.models import train

    with trained_project() as project:
        project.config.raw["tracking"]["promote_min_accuracy"] = 0.0
        assert register_best_model(project.config)["promoted"] is True

        train.run(project.config, use_mlflow=False)  # new weights, no new evaluation

        decision = register_best_model(project.config)
        assert decision["promoted"] is False
        assert decision["checkpoint_matches_run"] is False
        assert "not the one on disk" in decision["reason"]


def test_promotion_without_any_runs_is_a_clean_refusal() -> None:
    """With nothing to promote the gate declines rather than crashing."""
    with TempProject() as project:
        decision = register_best_model(project.config)
        assert decision["promoted"] is False
        assert "no finished run" in decision["reason"]
