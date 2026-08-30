# Arch: ![Uploading image.png…]()



# mlops-catsdogs

An end-to-end MLOps pipeline for binary image classification (cat vs dog) on a pet
adoption platform: data versioning, experiment tracking, a containerised inference
API, CI/CD, and a control-room dashboard that drives all of it from a browser.

```
make install     # create the virtualenv and install everything
make demo        # train a model, then start the API and the dashboard
```

Then open **http://127.0.0.1:8501**. `make stop` shuts both services down.

---

## What runs where

| Component | URL | What it is |
| --- | --- | --- |
| Inference API | http://127.0.0.1:8000 | Flask REST service: health, readiness, prediction, metrics |
| Dashboard | http://127.0.0.1:8501 | Control room: every stage, every button, live logs |
| Prometheus (optional) | http://127.0.0.1:9090 | `make compose-up` |

---

## The dashboard

The header carries a **pipeline spine** — seven markers, one per stage. A marker
fills in only when that stage's artifact actually exists on disk, so the spine
reports what happened rather than what the UI thinks happened. Click a marker to
jump to its panel.

| Panel | Covers | Buttons |
| --- | --- | --- |
| Overview | Whole-project state | Run the whole pipeline |
| Data & versioning | Splits, digests, Git and DVC state | Generate dataset · Preprocess · Recompute lock · Show samples · `dvc status/repro/dag/add` |
| Model building | Metrics, curves, promotion, model card | Train · Evaluate · Apply promotion gate |
| Experiment tracking | Runs, params, metrics, artifacts | Click any run for its detail |
| Inference API | Live endpoint exercise | `GET /health` `/ready` `/model-info` `/` · `POST /reload` · Predict from a file · Use a random test image |
| Monitoring & logs | Counters, latency, containers, log stream | Send 20 predictions · Show raw `/metrics` · Fetch logs (local / docker) · Collect & archive container logs |
| Performance tracking | Live-vs-baseline scoring | Run performance check |
| Deployment | Manifests, workflows, Docker files | Click any file to read it |

Long actions run as background jobs: the button returns immediately, a drawer
streams the real log lines, and a toast reports the outcome. Only one job runs at a
time, because the stages write shared files and two at once would corrupt them.

---

## The pipeline

```
make data          # generate the dataset (synthetic, no credentials needed)
make data-kaggle   # …or download the real Kaggle archive instead
make preprocess    # 224x224 RGB, deterministic 80/10/10 split, manifest + digests
make train         # train, log per-epoch metrics, keep the best epoch
make evaluate      # score the test split, write the baseline, plots, model card
make promote       # apply the accuracy gate
make pipeline      # all of the above in order
```

Every stage runs through `python -m mlops.cli`, so the Makefile, the DVC stages,
the container entrypoint, CI and the dashboard buttons all execute identical code.

`make help` lists all 30-odd targets.

---

## Design decisions worth knowing about

**The split is hashed, not shuffled.** `sha256(salt + relative path)` decides which
split an image lands in. Adding images therefore never moves an existing image
between train and test, and two machines produce the same split. Random shuffling
promises neither, and a test set that quietly changes between runs makes every
comparison meaningless. (One documented exception: a class with barely more images
than there are splits gets rebalanced so no split is empty.)

**One feature path.** `image_to_features` is the only place an image becomes a model
input. Training, evaluation, the API and the post-deployment checker all call it. A
train/serve preprocessing mismatch produces no error — only confidently wrong
predictions — so the code makes it impossible rather than testing for it.

**Liveness and readiness answer different questions.** `/health` never touches the
model, so a pod with a missing checkpoint is removed from the load balancer
(readiness fails) but not restart-looped (liveness passes). Restarting cannot
conjure a model file.

**The checkpoint is baked into the image.** `make docker-build` refuses to build
without `artifacts/model.pkl`. Pull the tag, run it, get predictions — no volume to
mount, no model to download at start-up, and the image is an immutable bundle of
code plus the exact weights it was tested with.

**Metrics and plots are tracked by Git, not DVC cache.** `.gitignore` deliberately
does *not* exclude `artifacts/metrics/`, `artifacts/plots/` or `model_card.md`. A
reviewer must be able to read the results from a bare clone without access to a DVC
remote.

**The DVC remote lives inside the repository** (`.dvcstore`). A remote at `../store`
cannot travel with a clone or a zip, so `dvc pull` would point at a directory that
does not exist on the recipient's machine. Repoint it at S3, GCS or SSH for real
multi-machine work; nothing else changes.

**The synthetic classes overlap on purpose.** Coat hue, texture frequency, body
proportions and ear shape are each drawn from distributions that overlap their
counterpart, with independent pose, lighting, occlusion and noise on top. A dataset
where one channel gives the answer produces 100% accuracy in the first epoch, flat
curves, early stopping that never fires and a monitoring baseline that cannot move.

---

## Data versioning

Two mechanisms, deliberately:

- **DVC** (`dvc.yaml`, `params.yaml`, `.dvc/config`) is the real pipeline.
  `make dvc-repro` reruns whatever is stale.
- **`data/dataset.lock.json`** is a sorted file-by-file SHA-256 index of the raw and
  processed trees, committed to Git.

DVC keeps the *contents* in a cache a fresh clone must pull from a remote; the lock
keeps the *identity* in Git. Anyone can verify two runs used the same data without
any access to the remote. The dashboard shows both.

---

## Experiment tracking

`Tracker` writes to **MLflow** whenever it is importable, and **always** mirrors
every run to `artifacts/runs/<run_id>/run.json`.

The mirror is not redundancy for its own sake. A dashboard that shows nothing when
MLflow is unreachable is a dashboard nobody trusts, so the experiment view works
from a bare clone, in CI, and inside a pod with no route to a tracking server —
while MLflow still receives everything when it is available.

```bash
mlflow ui --backend-store-uri file:./mlruns   # if you want the MLflow UI too
```

---

## The API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness. Independent of model state. |
| GET | `/ready` | Readiness. 503 until a checkpoint is loaded. |
| POST | `/predict` | One image: multipart upload or base64 JSON. |
| POST | `/predict/batch` | Up to 32 base64 images in one call. |
| GET | `/model-info` | Identity, input contract, training provenance. |
| POST | `/reload` | Re-read the checkpoint without restarting. |
| GET | `/metrics` | Prometheus exposition text. |
| GET | `/metrics-summary` | Compact JSON summary for the dashboard. |

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/predict \
  -F "file=@data/processed/test/dog/dog_00293.jpg;type=image/jpeg"

curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\":\"$(base64 -w0 data/processed/test/cat/cat_00273.jpg)\"}"
```

Errors are typed and consistent: `415` unsupported media type, `413` payload too
large, `422` undecodable image or bad batch, `503` no model loaded. Every response —
including 404 — is JSON, and carries an `X-Request-ID` that is echoed back if you
supply one.

The API loads the checkpoint once at start-up. After retraining, `POST /reload` (or
the dashboard's button) swaps in the new weights in place.

---

## Monitoring, logs and performance tracking

Every component emits one JSON object per line to **stderr** and to a rotating
**file**. Stderr is what `docker logs` and `kubectl logs` collect; the file is what
the dashboard tails outside a cluster. Logs go to stderr so stdout stays a clean
channel for command results — `python -m mlops.cli train | jq .best_epoch` works.

Payload bytes are never logged: uploads are recorded as size, declared type and a
truncated digest.

Counters, a latency histogram and a confidence histogram are exposed at `/metrics`
in Prometheus text format, written by hand so the container needs no client library.

**The post-deployment check** (`make perf-check`) draws a stratified, seeded,
labelled batch from the held-out test split, sends every image to the live HTTP
endpoint exactly as a client would, and scores the responses against
`artifacts/metrics/baseline.json` on three gates. Going over HTTP is the point: it
exercises the deployed container, its preprocessing, its threshold and its
checkpoint together, catching failures an offline metric cannot — a stale image tag,
a checkpoint that never got baked in, a normalisation mismatch.

**Log collection** uses two approaches the dashboard can read:

- Outside a containerised runtime the dashboard tails the local JSONL log files.
- When running the stack via Docker Compose the dashboard reads container status
  and `docker logs` output from the host Docker daemon.

This keeps the dashboard free of any Kubernetes dependencies: you can start the
stack with Docker and get live logs without `kubectl`.

---

## Deploying locally with Docker Compose

The container is deployed **after** training, so the image carries the model.

Start the stack locally with Compose (API, dashboard, Prometheus):

```bash
make pipeline        # train first — the image needs artifacts/model.pkl
make compose-up     # build and start the API, dashboard and Prometheus
```

The dashboard will be available at http://127.0.0.1:8501 and reads container
status and logs from Docker. Use `make compose-down` to stop and remove the
stack.

Manifests: optional Kubernetes manifests live in the `k8s/` folder for reference
only; the project defaults to Docker Compose for local development and verification.

The dashboard runs a single replica on purpose: it holds background-job state in
memory, so a second replica would answer half the status polls with "no such job".

---

## CI/CD

**`ci.yml`** — lint → tests with coverage → generate, preprocess, train, evaluate →
promotion gate (a failing gate fails the build) → start the API and smoke test it →
run the performance check → build the image with that model baked in → smoke test
the built container → Trivy scan → push to GHCR.

**`cd.yml`** — trains, builds the image, runs the stack with Docker Compose for
verification, smoke tests the API, scores it against the baseline, collects
compose logs and attaches them to the run. Rolls back on failure when present.

`cd.yml` triggers on `push` to main and `workflow_dispatch` — deliberately *not*
`workflow_run` chained to CI, because a chained trigger only fires when the workflow
file already exists on the default branch, which is precisely when someone setting
the project up for the first time will conclude CD is broken.

`cd.yml` runs verification with Docker Compose by default. Optionally the workflow
can target a self-hosted runner (for example `naveen-macbook-air`) that pulls the
built image and updates a local Compose stack to achieve a live deployment.

---

## Testing

```bash
make test                    # pytest with coverage, fails under 70%
python tests/run_tests.py    # same suite, no pytest needed
```

78 tests. They are written as plain functions with bare `assert`s and no pytest
fixtures, so both runners work — `make test` uses pytest as the brief requires,
while a reviewer who has only cloned the repository can verify the suite with no
extra installs.

Every test builds its own project root under a temporary directory, so running them
can never destroy a trained model and no test can pass because of state a previous
run left behind.

The log collector is tested against a stub script placed on `PATH` so subprocess
handling and JSON parsing are exercised rather than mocked away. The performance
checker's HTTP client is injectable, so the same code path runs against a Flask
test client with no socket.

---

## Configuration

`configs/config.yaml` is the single source of truth. Any key can be overridden by an
environment variable using `MLOPS_` and `__` for nesting:

```bash
MLOPS_TRAINING__EPOCHS=50 MLOPS_MODEL__TYPE=mlp make train
MLOPS_SERVING__PORT=9000 make serve-api
```

The Kubernetes ConfigMap, the Compose file and a developer shell all speak that same
syntax, so nothing in the image changes to retarget it.

---

## Layout

```
Makefile                 every target; `make help`
configs/config.yaml      single source of truth
params.yaml  dvc.yaml    DVC pipeline definition
src/mlops/
  config.py              YAML + MLOPS_* env overrides
  logging_setup.py       JSON logs, request correlation, safe payload metadata
  cli.py                 one entry point for every stage
  data/                  generate · preprocess · dataset · versioning
  models/                model · train · evaluate · plots
  tracking/tracker.py    MLflow + local run store, promotion gate
  serving/               app (Flask API) · predictor · metrics
  monitoring/            perf_tracker · log_collector
  ui/                    app · state · jobs · templates · static
tests/                   78 tests + a pytest-free runner
scripts/                 smoke_test · wait_for · fetch_kaggle
docker/                  Dockerfile · docker-compose.yml · prometheus.yml
k8s/                     namespace · configmap · rbac · deployments · services
.github/workflows/       ci.yml · cd.yml
docs/                    runbook, architecture, rubric mapping
```

---

## Requirements

Python 3.10+. Docker only for the deployment targets. `make install` handles the rest.
