# Runbook

Operating notes, failure modes and what to do about them.

---

## Common tasks

| Task | Command |
| --- | --- |
| Full setup and demo | `make install && make demo` |
| Retrain only | `make train && make evaluate` |
| Reload the running API after retraining | `curl -X POST localhost:8000/reload` |
| Stop background services | `make stop` |
| Wipe data, artifacts and logs | `make clean` |
| Check everything at once | `make status` |
| Run the tests | `make test` or `python tests/run_tests.py` |
| Deploy locally (Compose) | `make pipeline && make compose-up` |
| Stop the Compose stack | `make compose-down` |

---

## Failure modes

### `/ready` returns 503

The service is alive but has no checkpoint. This is by design — the API starts
without a model and says so, rather than crash-looping in an orchestrated
environment where a restart cannot produce a model file.

```bash
ls -l artifacts/model.pkl     # missing?
make train                    # produce one
curl -X POST localhost:8000/reload
```

In the cluster, the checkpoint is baked into the image, so a 503 there means the
image was built without one. `make docker-build` refuses that, so it usually means
an older image tag is deployed.

### Predictions look wrong after retraining

The API loads the checkpoint once at start-up. Either `POST /reload` or restart it.
With Compose, restart the API service:

```bash
docker compose -f docker/docker-compose.yml restart api
```

### The dashboard shows "API not reachable"

Check `ui.api_url`. Inside Compose it must be the **service name**, not `127.0.0.1`
— each container has its own loopback, so localhost points the dashboard at itself.

When using Compose, inspect the `docker/docker-compose.yml` environment values or
the runtime environment of the `ui` service to verify `MLOPS_UI__API_URL`.

### The log panel is empty in the cluster

If the log panel is empty when running under Compose, check the UI service has
access to the host Docker socket (see `docker/docker-compose.yml` — the UI mounts
`/var/run/docker.sock` by default). The panel's message will also say which
collector it tried and why it fell back.

### `dvc` commands report "not installed"

Expected outside the virtualenv. `make install` provides it; the dashboard reports
the state honestly rather than pretending the command ran.

### A dashboard button returns 409

Another job is still running. Only one runs at a time, because the stages write
shared files — the dataset, the checkpoint — and two at once would corrupt them and
produce an unreproducible run. Wait for the drawer to finish.

### Training is slow

Reduce the work rather than the epochs:

```bash
MLOPS_DATA__IMAGES_PER_CLASS=100 make data
MLOPS_MODEL__FEATURE_SIZE=16 MLOPS_TRAINING__EPOCHS=10 make train
```

### Port already in use

```bash
make stop
MLOPS_SERVING__PORT=9000 make serve-api
```

---

## What to check after a deploy

```bash
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs --tail=50
make smoke
make perf-check
```

Note: To allow the dashboard to inspect container status and `docker logs` when
it runs inside Compose, the UI service mounts the host Docker socket. This is
convenient for local development but grants the container broad control over
the host Docker daemon — treat accordingly and do not use this in multi-tenant
or untrusted environments.

The performance check is the meaningful one: it proves the *deployed* container
returns correct predictions on labelled data, which health checks do not.

---

## Interpreting the performance report

`artifacts/metrics/perf_check.json` has three gates:

| Gate | Meaning when it fails |
| --- | --- |
| `absolute_accuracy` | The live model is below the usable floor, whatever the baseline said |
| `accuracy_drop_vs_baseline` | The deployed model is materially worse than the one that was evaluated — usually a stale image or a preprocessing mismatch |
| `request_success` | Some requests did not return a prediction at all — a transport or capacity problem, not a model problem |

A drop with a healthy absolute accuracy usually means the wrong checkpoint shipped.
A failure on `request_success` alone is an infrastructure issue; look at container
restarts and resource limits before touching the model.

---

## Tuning the thresholds

| Setting | Default | Effect |
| --- | --- | --- |
| `tracking.promote_min_accuracy` | 0.75 | Below this, `make promote` exits non-zero and CI fails |
| `monitoring.perf_check.min_accuracy` | 0.70 | Absolute floor for live traffic |
| `monitoring.perf_check.max_accuracy_drop` | 0.10 | Allowed regression from the baseline |
| `evaluation.threshold` | 0.5 | Decision threshold; stored in the checkpoint so serving cannot drift from evaluation |

---

## Scaling notes

The API is stateless and scales horizontally; the Deployment runs 2 replicas with
`maxUnavailable: 0` so a rollout never drops below capacity.

The dashboard is deliberately single-replica: background-job state lives in memory,
so a second replica would answer half the status polls with "no such job". Moving
jobs to a shared queue is the change that would make it scalable, and it is not
worth it for a control panel.

The built-in Flask server is used for simplicity. `gunicorn` is already pinned in
`requirements.txt` for a production swap:

```bash
gunicorn --bind 0.0.0.0:8000 --workers 4 --timeout 60 \
  "mlops.serving.app:create_app()"
```

Note that per-process counters mean each worker holds its own metrics; with multiple
workers, scrape each one or move to a shared store.
