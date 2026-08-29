# Single image, two entrypoints: the API and the dashboard run the same code with
# different arguments, so there is one artifact to build, scan and promote rather
# than two that can drift apart.
#
# Multi-stage: wheels are built once in the builder and only the installed
# site-packages are carried into the runtime layer, which keeps compilers and
# build headers out of the shipped image.
#
# The trained checkpoint is COPYed in. That is deliberate — the brief deploys the
# container *after* training, so the image is an immutable bundle of code and the
# exact weights it was tested with. `make docker-build` refuses to build without
# artifacts/model.pkl rather than producing an image that starts and then fails
# readiness forever.

# ----------------------------------------------------------------- builder
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# MLflow and DVC are development-time tools; the serving image does not need them,
# and leaving them out removes roughly 300 MB and a large slice of attack surface.
RUN grep -v -E '^(mlflow|dvc)==' requirements.txt > runtime-requirements.txt \
 && pip install --prefix=/install -r runtime-requirements.txt

# ----------------------------------------------------------------- runtime
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="mlops-catsdogs" \
      org.opencontainers.image.description="Cats vs dogs classifier: inference API and control-room dashboard" \
      org.opencontainers.image.source="https://github.com/your-org/mlops-catsdogs" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    MLOPS_PROJECT_ROOT=/app \
    MLOPS_SERVING__HOST=0.0.0.0 \
    MLOPS_UI__HOST=0.0.0.0

WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/        /app/src/
COPY configs/    /app/configs/
COPY scripts/    /app/scripts/
COPY artifacts/model.pkl              /app/artifacts/model.pkl
COPY artifacts/metrics/baseline.json  /app/artifacts/metrics/baseline.json
COPY artifacts/model_card.md          /app/artifacts/model_card.md

# Run as a non-root user with a fixed uid so the Kubernetes securityContext can
# assert the same value.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin appuser \
 && mkdir -p /app/logs /app/data /app/artifacts/runs /app/artifacts/plots \
 && chown -R 10001:10001 /app

USER 10001

EXPOSE 8000 8501

# Uses only the standard library, so the check does not depend on curl being
# present in a slim base image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "mlops.cli", "serve-api", "--port", "8000"]
