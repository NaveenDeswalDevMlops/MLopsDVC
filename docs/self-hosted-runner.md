# Self-hosted runner: `naveen-macbook-air`

This project’s CD workflow includes a job that targets a self-hosted runner
labelled `naveen-macbook-air`. The runner should be registered on your laptop
and be reachable by GitHub Actions. The steps below show how to add the runner
and configure the minimal secrets used by the workflow.

## 1) Install and register the runner on your machine

1. On GitHub, go to: `Repository` → `Settings` → `Actions` → `Runners` → `New
   self-hosted runner` and follow the instructions for your OS.

2. On your laptop run the registration commands shown on the GitHub page. For
   macOS (example):

```bash
# create a directory for the runner
mkdir -p ~/actions-runner && cd ~/actions-runner
# download the runner (replace VERSION with the one GitHub shows)
curl -O -L https://github.com/actions/runner/releases/download/v2.308.0/actions-runner-osx-x64-2.308.0.tar.gz
tar xzf ./actions-runner-osx-x64-2.308.0.tar.gz
# register (paste the token from the GitHub page)
./config.sh --url https://github.com/<owner>/<repo> --token <RUNNER_TOKEN> --labels "self-hosted,naveen-macbook-air"
# install as a service (optional)
./svc.sh install
./svc.sh start
```

Notes:
- Ensure the runner has Docker and docker-compose available on PATH.
- When registering, add the labels `self-hosted` and `naveen-macbook-air` so
  the workflow picks this runner.

## 2) Recommended repository secrets

Set the following repository secrets at `Repository` → `Settings` → `Secrets`:

- `GHCR_PAT` (optional) — a personal access token with `read:packages` if your
  repository uses private packages on `ghcr.io`. If unset, the workflow will
  still attempt `docker pull` but may fail for private images.
- `NAVEEN_COMPOSE_DIR` (optional) — path to the checked-out repository on the
  runner. Defaults to `~/mlops-catsdogs`.

How to add secrets from the command line (GitHub CLI example):

```bash
gh secret set GHCR_PAT --body "$(cat ~/secrets/ghcr_pat.txt)" --repo <owner>/<repo>
gh secret set NAVEEN_COMPOSE_DIR --body "/Users/naveen/mlops-catsdogs" --repo <owner>/<repo>
```

## 3) What the workflow expects

- The self-hosted runner must be able to run `docker` and `docker compose`.
- The workflow will build and push an image to `ghcr.io/<owner>/mlops-catsdogs:$TAG`
  and then, on the self-hosted runner, it will `docker pull` that image, tag it
  as `mlops-catsdogs:local` and run `docker compose -f docker/docker-compose.yml up -d`.

If your runner cannot access `ghcr.io`, consider using Docker Hub or pushing
the built image via the runner itself (we can change the workflow to build on
the runner if you prefer).

---

If you want, I can add a short `README` section linking to this page and a
one-liner to help you validate the runner (e.g. `docker ps` and `docker compose ps`).
