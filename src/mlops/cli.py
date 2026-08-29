"""One entry point for every stage of the pipeline.

    python -m mlops.cli <command> [options]

Every command the Makefile, the DVC pipeline, the Dockerfile and the dashboard run
goes through here, so there is exactly one definition of what "train" means and no
chance of the Makefile and CI drifting apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from mlops.config import get_config, load_config
from mlops.logging_setup import configure_logging, get_logger

_LOGGER = get_logger(__name__)


def _bootstrap(args: argparse.Namespace) -> Any:
    """Load configuration and install logging for a command.

    Args:
        args: Parsed arguments.

    Returns:
        The effective configuration.
    """
    config = load_config(args.config) if getattr(args, "config", None) else get_config()
    configure_logging(
        level=getattr(args, "log_level", "INFO"),
        log_file=config.path("monitoring.log_file"),
        force=True,
    )
    config.ensure_dirs()
    return config


def _emit(payload: dict[str, Any]) -> None:
    """Print a result as indented JSON.

    Args:
        payload: The result to print.
    """
    print(json.dumps(payload, indent=2, default=str))


def cmd_generate_data(args: argparse.Namespace) -> int:
    """Generate the synthetic raw dataset.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    from mlops.data.generate import generate_dataset

    config = _bootstrap(args)
    _emit(generate_dataset(config, per_class=args.per_class))
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    """Run the preprocessing stage.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    from mlops.data.preprocess import run
    from mlops.data.versioning import write_dataset_lock

    config = _bootstrap(args)
    stats = run(config)
    stats["lock"] = write_dataset_lock(config)
    _emit(stats)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train a model.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    from mlops.models.train import run

    config = _bootstrap(args)
    result = run(config, use_mlflow=None if not args.no_mlflow else False)
    _emit(result.to_dict())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate the saved checkpoint.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    from mlops.models.evaluate import run

    config = _bootstrap(args)
    result = run(config, split=args.split, use_mlflow=None if not args.no_mlflow else False)
    _emit(result.to_dict())
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    """Apply the promotion gate to the best run.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status; non-zero when the gate rejects the model.
    """
    from mlops.tracking.tracker import register_best_model

    config = _bootstrap(args)
    decision = register_best_model(config)
    _emit(decision)
    return 0 if decision["promoted"] else 1


def cmd_serve_api(args: argparse.Namespace) -> int:
    """Run the inference API.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    from mlops.serving.app import create_app

    config = _bootstrap(args)
    app = create_app(config)
    host = args.host or str(config.get("serving.host", "0.0.0.0"))
    port = int(args.port or config.get("serving.port", 8000))
    _LOGGER.info("starting api", extra={"host": host, "port": port})
    app.run(host=host, port=port, threaded=True, use_reloader=False)
    return 0


def cmd_serve_ui(args: argparse.Namespace) -> int:
    """Run the dashboard.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    from mlops.ui.app import create_ui_app

    config = _bootstrap(args)
    app = create_ui_app(config)
    host = args.host or str(config.get("ui.host", "0.0.0.0"))
    port = int(args.port or config.get("ui.port", 8501))
    _LOGGER.info("starting dashboard", extra={"host": host, "port": port})
    app.run(host=host, port=port, threaded=True, use_reloader=False)
    return 0


def cmd_perf_check(args: argparse.Namespace) -> int:
    """Run the post-deployment performance check.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status; non-zero when a gate fails.
    """
    from mlops.monitoring.perf_tracker import run

    config = _bootstrap(args)
    result = run(
        config,
        endpoint_url=args.endpoint,
        sample_size=args.sample_size,
        use_mlflow=None if not args.no_mlflow else False,
    )
    _emit(result.to_dict())
    return 0 if result.passed else 1


def cmd_collect_logs(args: argparse.Namespace) -> int:
    """Collect logs from the configured source.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    from mlops.monitoring.log_collector import collect

    config = _bootstrap(args)
    bundle = collect(config, source=args.source, tail=args.tail)
    _emit(bundle.to_dict())
    return 0


def cmd_dataset_lock(args: argparse.Namespace) -> int:
    """Recompute the dataset lock file.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    from mlops.data.versioning import write_dataset_lock

    config = _bootstrap(args)
    _emit(write_dataset_lock(config))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Print a consolidated status of the whole project.

    Args:
        args: Parsed arguments.

    Returns:
        Exit status.
    """
    from mlops.ui.state import project_status

    config = _bootstrap(args)
    _emit(project_status(config))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(prog="mlops", description=__doc__)
    parser.add_argument("--config", help="path to an alternative configs/config.yaml")
    parser.add_argument("--log-level", default="INFO", help="logging level (default: INFO)")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate-data", help="write the synthetic raw dataset")
    generate.add_argument("--per-class", type=int, default=None, help="images per class")
    generate.set_defaults(func=cmd_generate_data)

    preprocess = sub.add_parser("preprocess", help="resize, split and index the raw dataset")
    preprocess.set_defaults(func=cmd_preprocess)

    train = sub.add_parser("train", help="train a model and track the run")
    train.add_argument("--no-mlflow", action="store_true", help="skip MLflow, use the local store")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="evaluate the checkpoint and write the baseline")
    evaluate.add_argument("--split", default="test", choices=["train", "val", "test"])
    evaluate.add_argument("--no-mlflow", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    promote = sub.add_parser("promote", help="apply the accuracy gate to the best run")
    promote.set_defaults(func=cmd_promote)

    serve_api = sub.add_parser("serve-api", help="run the inference API")
    serve_api.add_argument("--host", default=None)
    serve_api.add_argument("--port", type=int, default=None)
    serve_api.set_defaults(func=cmd_serve_api)

    serve_ui = sub.add_parser("serve-ui", help="run the dashboard")
    serve_ui.add_argument("--host", default=None)
    serve_ui.add_argument("--port", type=int, default=None)
    serve_ui.set_defaults(func=cmd_serve_ui)

    perf = sub.add_parser("perf-check", help="score live traffic against the baseline")
    perf.add_argument("--endpoint", default=None, help="base URL of the running service")
    perf.add_argument("--sample-size", type=int, default=None)
    perf.add_argument("--no-mlflow", action="store_true")
    perf.set_defaults(func=cmd_perf_check)

    logs = sub.add_parser("collect-logs", help="collect logs from local files or minikube")
    logs.add_argument("--source", default="auto", choices=["auto", "local", "kubectl", "in-cluster"])
    logs.add_argument("--tail", type=int, default=None)
    logs.set_defaults(func=cmd_collect_logs)

    lock = sub.add_parser("dataset-lock", help="recompute data/dataset.lock.json")
    lock.set_defaults(func=cmd_dataset_lock)

    status = sub.add_parser("status", help="print a consolidated project status")
    status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001 - the CLI is the last line of defence
        _LOGGER.error("command failed", extra={"command": args.command, "error": str(exc)})
        print(json.dumps({"error": str(exc), "command": args.command}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
