# One entry point for the whole project. `make help` lists every target.
#
# Everything runs through `python -m mlops.cli`, so the Makefile, the DVC stages,
# the container entrypoint and the dashboard buttons all execute identical code.

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON      ?= python3
VENV        ?= .venv
BIN         := $(VENV)/bin
PY          := $(BIN)/python
PIP         := $(BIN)/pip
export PYTHONPATH := $(CURDIR)/src

IMAGE       ?= mlops-catsdogs
TAG         ?= local
NAMESPACE   ?= mlops
API_PORT    ?= 8000
UI_PORT     ?= 8501
API_URL     ?= http://127.0.0.1:$(API_PORT)
PER_CLASS   ?= 300

.PHONY: help install clean-venv data data-kaggle preprocess train evaluate promote pipeline \
	serve-api serve-ui stop test lint dvc-init dvc-repro dvc-status dataset-lock \
	perf-check smoke docker-build docker-run compose-up compose-down dvc-commit mlflow-ui \
	clean status demo

help: ## Show this help
	@echo "mlops-catsdogs — targets"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick start:  make install && make demo"

# ---------------------------------------------------------------- environment

$(VENV)/bin/activate: requirements.txt requirements-dev.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	@touch $(VENV)/bin/activate
	@echo "Environment ready. Next: make demo"

install: $(VENV)/bin/activate ## Create the virtualenv and install dependencies

clean-venv: ## Remove the virtualenv
	rm -rf $(VENV)

# ---------------------------------------------------------------- pipeline

data: install ## Generate the synthetic raw dataset (no credentials needed)
	$(PY) -m mlops.cli generate-data --per-class $(PER_CLASS)

data-kaggle: install ## Download the real Kaggle cats-vs-dogs archive instead
	@bash scripts/fetch_kaggle.sh

preprocess: install ## Resize to 224x224, split deterministically, write the manifest
	$(PY) -m mlops.cli preprocess

train: install ## Train a model and record the run
	$(PY) -m mlops.cli train

evaluate: install ## Evaluate on the held-out test split and write the baseline
	$(PY) -m mlops.cli evaluate

promote: install ## Apply the accuracy gate to the best run
	$(PY) -m mlops.cli promote

pipeline: data preprocess train evaluate promote ## Run every stage in order

dataset-lock: install ## Recompute data/dataset.lock.json
	$(PY) -m mlops.cli dataset-lock

status: install ## Print a consolidated project status
	$(PY) -m mlops.cli status

# ---------------------------------------------------------------- services

serve-api: install ## Run the inference API in the foreground
	$(PY) -m mlops.cli serve-api --port $(API_PORT)

serve-ui: install ## Run the dashboard in the foreground
	$(PY) -m mlops.cli serve-ui --port $(UI_PORT)

demo: pipeline ## Train, then start the API and the dashboard in the background
	@mkdir -p logs
	@($(PY) -m mlops.cli serve-api --port $(API_PORT) > logs/api.out 2>&1 & echo $$! > logs/api.pid)
	@($(PY) -m mlops.cli serve-ui  --port $(UI_PORT)  > logs/ui.out  2>&1 & echo $$! > logs/ui.pid)
	@bash scripts/wait_for.sh $(API_URL)/health 40
	@bash scripts/wait_for.sh http://127.0.0.1:$(UI_PORT)/health 40
	@echo ""
	@echo "  API       $(API_URL)"
	@echo "  Dashboard http://127.0.0.1:$(UI_PORT)"
	@echo "  Stop both with: make stop"

stop: ## Stop the background API and dashboard
	@for pidfile in logs/api.pid logs/ui.pid; do \
		if [ -f $$pidfile ]; then kill $$(cat $$pidfile) 2>/dev/null || true; rm -f $$pidfile; fi; \
	done
	@echo "stopped"

mlflow-ui: install ## Open the MLflow tracking UI on :5001 (runs are written by every stage)
	@echo "MLflow UI at http://127.0.0.1:5001 — Ctrl-C to stop"
	$(BIN)/mlflow ui --backend-store-uri file:./mlruns --port 5001

perf-check: install ## Score the live endpoint against the training baseline
	$(PY) -m mlops.cli perf-check --endpoint $(API_URL)

smoke: install ## Health, readiness and a real prediction against a running API
	@bash scripts/smoke_test.sh $(API_URL)

# ---------------------------------------------------------------- quality

test: install ## Run the test suite with coverage
	$(BIN)/pytest tests -q --cov=src/mlops --cov-report=term-missing --cov-fail-under=70

lint: install ## Lint the source tree
	$(BIN)/ruff check src tests

# ---------------------------------------------------------------- dvc

dvc-init: install ## Initialise the DVC repository (run `git init` first if you can)
	@if [ -d .git ]; then \
		echo "Git repository found — initialising DVC with SCM integration."; \
		$(BIN)/dvc init --force; \
	else \
		echo "WARNING: no .git directory here."; \
		echo "  Falling back to 'dvc init --no-scm'. DVC will work, but it will not"; \
		echo "  stage .dvc files into Git, so the data version is not recorded"; \
		echo "  alongside the code — which is most of the point of DVC."; \
		echo "  Run 'git init && git add -A && git commit -m initial' first, then"; \
		echo "  re-run 'make dvc-init' to get the SCM-integrated setup."; \
		$(BIN)/dvc init --no-scm --force; \
	fi
	$(BIN)/dvc remote add -d -f localstore .dvcstore
	@echo "DVC ready."
	@echo "  make dvc-repro    run the pipeline through DVC (produces dvc.lock)"
	@echo "  make dvc-commit   record outputs you already built, without re-running"

dvc-commit: install ## Record already-produced pipeline outputs in DVC, then push
	@# NOT `dvc add`: every data path here is declared as a stage output in
	@# dvc.yaml, and DVC refuses to let a path be both a pipeline output and a
	@# manually added one. `dvc commit` records the outputs that already exist on
	@# disk without re-running the stages that made them.
	$(BIN)/dvc commit --force
	$(BIN)/dvc push
	@echo ""
	@echo "dvc.lock now holds the content hashes — commit it to Git."

dvc-repro: install ## Reproduce any stale pipeline stage (writes dvc.lock)
	$(BIN)/dvc repro
	$(BIN)/dvc push
	@echo ""
	@echo "dvc.lock records the hash of every input and output — commit it to Git."

dvc-status: install ## Show which stages are stale
	$(BIN)/dvc status

# ---------------------------------------------------------------- containers

docker-build: ## Build the image with the trained model baked in
	@test -f artifacts/model.pkl || (echo "No artifacts/model.pkl — run 'make pipeline' first." && exit 1)
	docker build -f docker/Dockerfile -t $(IMAGE):$(TAG) .

docker-run: ## Run the API and dashboard from the built image
	docker run --rm -p $(API_PORT):8000 -p $(UI_PORT):8501 --name catsdogs $(IMAGE):$(TAG)

compose-up: ## Start the API, dashboard and Prometheus with Compose
	docker compose -f docker/docker-compose.yml up --build -d

compose-down: ## Stop the Compose stack
	docker compose -f docker/docker-compose.yml down -v


# ---------------------------------------------------------------- cleanup

clean: ## Remove generated data, artifacts and logs
	rm -rf data/raw data/processed data/dataset.lock.json artifacts logs mlruns .dvcstore/files
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned"
