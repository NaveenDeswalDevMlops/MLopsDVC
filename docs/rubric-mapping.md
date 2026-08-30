# Where each requirement is implemented

A map from the assignment's five modules to the files, commands and evidence that
satisfy them.

---

## M1 — Model development & experiment tracking

### M1.1 Data & code versioning

| Requirement | Where |
| --- | --- |
| Git for source, scripts, structure | Whole repository; `.gitignore` excludes binaries but **keeps** metrics, plots and the model card so results are readable from a bare clone |
| DVC for dataset versioning | `dvc.yaml` (4 stages), `params.yaml`, `.dvc/config`, `.dvcignore` |
| Track pre-processed data | `preprocess` stage declares `data/processed` as an output |

Commands: `make dvc-init`, `make dvc-repro`, `make dvc-status`. The dashboard's
*Data & versioning* panel runs `dvc status`, `dvc repro`, `dvc dag` and
`dvc add data/raw` from buttons and shows the verbatim output.

Beyond the requirement: `data/dataset.lock.json` is a file-by-file SHA-256 index of
both trees, committed to Git. DVC keeps the contents in a cache that must be pulled
from a remote; the lock keeps the identity in Git, so two runs can be proven to have
used the same data with no remote access at all.

The DVC remote is `.dvcstore` **inside** the repository. A remote outside the tree
cannot travel with a clone or a zip, and `dvc pull` would point at a path the
recipient does not have.

### M1.2 Model building

| Requirement | Where |
| --- | --- |
| Baseline model | `src/mlops/models/model.py` — logistic regression on flattened pixels (`SGDClassifier`, log loss), the brief's named baseline; an MLP is available via `model.type: mlp` on the same harness |
| Serialized format | joblib `.pkl` at `artifacts/model.pkl`, carrying weights, scaler **and** a metadata block (feature geometry, class order, threshold, dataset digest, library versions, git SHA) |

Preprocessing to 224×224 RGB and the 80/10/10 split are in
`src/mlops/data/preprocess.py`; augmentation is in `src/mlops/data/dataset.py` and
applies to the training split only — enforced in code, not by convention.

### M1.3 Experiment tracking

| Requirement | Where |
| --- | --- |
| Open-source tracker | `src/mlops/tracking/tracker.py` — MLflow when importable, plus an always-written local run store |
| Runs, params, metrics | Every stage opens a `Tracker`; per-epoch metrics are logged with a step index |
| Artifacts (confusion matrix, loss curves) | `src/mlops/models/plots.py` writes loss curve, accuracy curve, confusion matrix and ROC; all four are attached to their run |

Evidence: `artifacts/runs/<run_id>/run.json`, `artifacts/plots/*.png`,
`artifacts/metrics/*.json`, and the dashboard's *Experiment tracking* panel.

---

## M2 — Model packaging & containerization

### M2.1 Inference service

| Requirement | Where |
| --- | --- |
| REST API (FastAPI/Flask) | `src/mlops/serving/app.py` — Flask |
| Health check endpoint | `GET /health` (liveness) and `GET /ready` (readiness) |
| Prediction endpoint | `POST /predict` returns label **and** per-class probabilities |

Also: `POST /predict/batch`, `GET /model-info`, `POST /reload`, `GET /metrics`,
`GET /metrics-summary`, and a JSON index at `/`.

`/health` and `/ready` are separate on purpose. A pod with a missing checkpoint
should leave the load-balancer rotation without being restart-looped, because
restarting cannot produce a model file.

### M2.2 Environment specification

`requirements.txt` — every version pinned, including transitive pins for Flask and
Werkzeug. `requirements-dev.txt` layers on pytest, coverage and ruff.

### M2.3 Containerization

`docker/Dockerfile` — multi-stage, non-root uid 10001, stdlib `HEALTHCHECK` (no curl
needed in a slim base), MLflow and DVC stripped from the runtime layer (~300 MB and
a slice of attack surface removed). `.dockerignore` denies everything then allows
only what the image needs.

`make docker-build` refuses to build without `artifacts/model.pkl`, so an image that
would start and then fail readiness forever cannot be produced.

Verify: `make docker-build && make docker-run`, then
`bash scripts/smoke_test.sh http://127.0.0.1:8000` or Postman.

---

## M3 — CI pipeline

### M3.1 Automated testing

`tests/` — 78 tests. The two the brief asks for specifically:

- **Data pre-processing**: `tests/test_preprocess.py` — geometry, manifest
  completeness, split determinism, the "adding data never reshuffles" property,
  empty-split rebalancing, digest sensitivity, corrupt-file tolerance.
- **Model utility / inference**: `tests/test_model.py` and `tests/test_api.py` —
  feature extraction, checkpoint round-trip, probability validity, and every API
  route including each error status.

`make test` runs pytest with coverage and a 70% floor. `python tests/run_tests.py`
runs the identical suite without pytest installed.

### M3.2 CI setup

`.github/workflows/ci.yml` (GitHub Actions). On every push and pull request:
checkout → install → lint → tests → generate/preprocess/train/evaluate → promotion
gate → start the API and smoke test it → performance check → build the image.

### M3.3 Artifact publishing

The same workflow pushes to **GitHub Container Registry** with branch, semver, SHA
and `latest` tags. The image is smoke-tested and Trivy-scanned (CRITICAL/HIGH,
`exit-code: 1`) *before* the push, so a container that fails its own health check
never becomes `latest`.

---

### M4 — CD pipeline & deployment

### M4.1 Deployment target

Local **Docker Compose**. `docker/docker-compose.yml` provides API, dashboard
and Prometheus for local integration testing. The image is still built with the
trained checkpoint baked in (`make docker-build`) and can be started with
`make compose-up`.

### M4.2 CD / GitOps flow

`.github/workflows/cd.yml` — trains and builds the image; CI then smoke tests the
built container and scans it with Trivy. (Some CI steps referenced minikube;
the dashboard and local runtime now default to Docker Compose.)

The trigger is deliberately not `workflow_run` chained to CI: that only fires once
the workflow file already exists on the default branch, which is exactly when a
first-time setup looks broken.

### M4.3 Smoke tests / health check

`scripts/smoke_test.sh` — health, readiness, model-info, metrics, a real prediction
with a real image, and a malformed payload that must be rejected with 422. It exits
non-zero on any failure, so the pipeline fails. Run against the stack with
`make smoke`.

CD additionally checks that the dashboard can read container logs (the Compose
verification mounts the Docker socket into the UI service), so a misconfigured
compose environment that prevents log access will surface as a failure rather than
silently producing an empty log view.

---

## M5 — Monitoring, logs & final submission

### M5.1 Basic monitoring & logging

| Requirement | Where |
| --- | --- |
| Request/response logging | `src/mlops/logging_setup.py` — one JSON object per line, to stderr and a rotating file |
| Excluding sensitive data | `safe_payload_meta` records size, declared type and a truncated digest; payload bytes are never logged |
| Request count and latency | `src/mlops/serving/metrics.py` — counters, latency histogram, confidence histogram, error counter, at `/metrics` in Prometheus text format |

Every request carries an `X-Request-ID`, honoured from the caller when supplied.

Logs: `src/mlops/monitoring/log_collector.py` collects either from the local JSONL
files or, when available, from the host Docker daemon using `docker ps`/`docker logs`.
Both normalise to the same record shape so the dashboard renders one merged stream.
`docker/prometheus.yml` scrapes the API directly.

### M5.2 Model performance tracking (post-deployment)

`src/mlops/monitoring/perf_tracker.py` — a stratified, seeded, labelled batch drawn
from the held-out test split is sent to the **live HTTP endpoint** and scored
against `artifacts/metrics/baseline.json` on three gates: an absolute accuracy
floor, a maximum drop from baseline, and full request success. Results go to
`artifacts/metrics/perf_check.json` and to the tracker as a run.

Run it with `make perf-check`, or the dashboard's *Performance tracking* panel.

---

## Deliverables

1. **Source, configs and artifacts** — this folder. Source, DVC config, CI/CD
  workflows, Dockerfile, Compose file, optional Kubernetes manifests (in `k8s/`), the trained
   checkpoint, metrics, plots and the model card.
2. **Screen recording** — suggested five-minute route in
   [`docs/demo-script.md`](demo-script.md).
