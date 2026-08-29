# Five-minute demo script

The deliverable asks for a recording of "the complete MLOps workflow from code
change to deployed model prediction". This route covers all five modules and lands
around 4:30, leaving room to talk.

Run `make install` and `make pipeline` **before** recording so you are not filming a
dependency install. Then `make clean` the data only if you want to show generation
live — the timings below assume you do.

---

## 0:00 — Open the dashboard (30s)

```bash
make demo
```

Open http://127.0.0.1:8501.

Point at the **pipeline spine** in the header. Say: seven stages, each marker fills
only when that stage's artifact actually exists on disk — it reports what happened,
not what the UI thinks happened. Click a marker to show it navigates.

---

## 0:30 — Data and versioning, M1.1 (50s)

Go to **Data & versioning**.

1. Click **Generate dataset**. The drawer streams the real log.
2. Click **Preprocess to 224×224**.
3. Point at the splits table and the dataset digest.

Say: the split is `sha256(salt + path)`, not a shuffle — adding images never moves
an existing image between train and test, and two machines get the same split.

4. Click **dvc status** to show DVC is wired in, and point at the lock digest next
   to it: DVC holds the contents, the Git-tracked lock holds the identity.
5. Click **Show sample images** for a quick look at the data.

---

## 1:20 — Training and tracking, M1.2 + M1.3 (60s)

Go to **Model building**.

1. Leave the model on *logistic regression*, click **Train**. Narrate over the
   drawer: per-epoch metrics, best epoch kept rather than the last one.
2. When it finishes, click **Evaluate on test split**.
3. Point at the metrics panel and scroll to the curves — loss, accuracy, confusion
   matrix, ROC.
4. Click **Apply promotion gate** and read the decision.

Go to **Experiment tracking**. Click a run to open params, metrics and artifacts.
Say: MLflow when it is available, plus a local run store written every time, so the
view works in a bare clone or a pod with no tracking server.

---

## 2:20 — The API, M2 (50s)

Go to **Inference API**.

1. Click **GET /health**, then **GET /ready**, then **GET /model-info**. Point out
   that health and readiness answer different questions.
2. Click **Use a random test image** — shows the image, the predicted label with
   confidence, and the true label with a tick or cross.
3. Optionally upload your own image with **Send to /predict**.

Cut to a terminal for one curl so the REST surface is visible outside the browser:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -F "file=@data/processed/test/dog/dog_00293.jpg;type=image/jpeg" | jq
```

---

## 3:10 — Monitoring and performance, M5 (60s)

Go to **Monitoring & logs**.

1. Click **Send 20 predictions**. The counters, latency percentiles and mean
   confidence update.
2. Click **Show raw /metrics** — valid Prometheus exposition, no client library in
   the container.
3. Click **Fetch logs**. Point at the structured JSON records and say that payload
   bytes are never logged, only size, type and a digest.

Go to **Performance tracking**, click **Run performance check**.

Say: this sends a labelled batch to the *live HTTP endpoint* and scores it against
the training baseline on three gates. Going over HTTP is the point — it catches a
stale image tag or a preprocessing mismatch that an offline metric cannot.

---

## 4:10 — Deployment, M3 + M4 (50s)

Go to **Deployment**, click through a manifest and `ci.yml` in the file preview.

Then cut to a terminal:

```bash
make docker-build
make compose-up
```

Open http://127.0.0.1:8501, go to **Monitoring & logs**, set the log source to
**docker** and click **Fetch logs** — the dashboard reads `docker ps`/`docker logs`
from the host so no Kubernetes tooling is required.

Finish by showing `docker compose -f docker/docker-compose.yml ps` with the API
and dashboard services running.

---

## Optional: the code-change loop

If you have time, this is the tightest way to show CI/CD end to end:

1. Change `training.epochs` in `configs/config.yaml`.
2. Commit and push — CI lints, tests, trains, gates, smoke tests and publishes.
3. CD builds inside minikube, applies the manifests and waits for the rollout.
4. Back in the dashboard, click **POST /reload** and show the new `trained_at`.
Note: CI builds the image and the CD workflow verifies it (the default CD
run uses Docker Compose for verification rather than a local Kubernetes
cluster). If you keep a minikube-based CD in your fork, adapt this step.
