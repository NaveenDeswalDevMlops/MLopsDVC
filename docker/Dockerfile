FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN grep -v -E '^(mlflow|dvc)==' requirements.txt > runtime-requirements.txt \
 && python -m pip install \
      --prefix=/install \
      --no-cache-dir \
      -r runtime-requirements.txt \
 && python -m pip install \
      --prefix=/install \
      --no-cache-dir \
      --upgrade \
      "jaraco.context==6.1.2" \
      "wheel==0.48.0"

RUN rm -rf \
      /install/lib/python3.11/site-packages/setuptools \
      /install/lib/python3.11/site-packages/setuptools-*.dist-info

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

# Copy application dependencies from builder.
COPY --from=builder /install /usr/local


RUN rm -rf \
      /usr/local/lib/python3.11/site-packages/setuptools \
      /usr/local/lib/python3.11/site-packages/setuptools-*.dist-info
      
RUN python -c "import importlib.metadata as m; \
    from packaging.version import Version; \
    jaraco=m.version('jaraco.context'); \
    wheel=m.version('wheel'); \
    print('jaraco.context:', jaraco); \
    print('wheel:', wheel); \
    assert Version(jaraco) >= Version('6.1.0'), \
      f'jaraco.context {jaraco} is below 6.1.0'; \
    assert Version(wheel) >= Version('0.46.2'), \
      f'wheel {wheel} is below 0.46.2'"


RUN if [ -d /usr/local/lib/python3.11/site-packages/setuptools ]; then \
      echo "ERROR: setuptools is still present"; \
      exit 1; \
    else \
      echo "OK: setuptools removed"; \
    fi

RUN if find /usr/local/lib/python3.11/site-packages \
      -path '*/setuptools/_vendor/jaraco.context-*' \
      -o -path '*/setuptools/_vendor/wheel-*' \
      | grep -q .; then \
      echo "ERROR: vulnerable vendored metadata still exists"; \
      find /usr/local/lib/python3.11/site-packages \
        \( -path '*/setuptools/_vendor/jaraco.context-*' \
        -o -path '*/setuptools/_vendor/wheel-*' \) \
        -print; \
      exit 1; \
    else \
      echo "OK: vulnerable setuptools vendored metadata removed"; \
    fi

# -----------------------------------------------------------------
# Application
# -----------------------------------------------------------------

COPY src/        /app/src/
COPY configs/    /app/configs/
COPY scripts/    /app/scripts/

COPY artifacts/model.pkl              /app/artifacts/model.pkl
COPY artifacts/metrics/baseline.json  /app/artifacts/metrics/baseline.json
COPY artifacts/model_card.md          /app/artifacts/model_card.md

# Run as a non-root user with a fixed uid so the Kubernetes securityContext
# can assert the same value.
RUN useradd \
      --uid 10001 \
      --create-home \
      --shell /usr/sbin/nologin \
      appuser \
 && mkdir -p \
      /app/logs \
      /app/data \
      /app/artifacts/runs \
      /app/artifacts/plots \
 && chown -R 10001:10001 /app

USER 10001

EXPOSE 8000 8501

# Uses only the Python standard library.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "mlops.cli", "serve-api", "--port", "8000"]