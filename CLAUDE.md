# CLAUDE.md

KServe model servers for Wikimedia Lift Wing. Monorepo at `machinelearning/liftwing/inference-services` on Gerrit.

## Quick Start

```bash
docker compose build <service> && docker compose up <service>
curl localhost:8080/v1/models/<model>:predict -X POST \
  -d '<payload>' -H "Content-type: application/json"
```

Requires `.env` with `PATH_TO_*_MODEL` vars. **ARM Macs**: services declare `platform: linux/amd64` in docker-compose.yml.

Payload format varies per model — check `src/models/<name>/model_server/model.py` for the expected input schema.

## Code Review

Gerrit — submit with `git review`. CI runs lint + tests per patchset.

## Testing & Linting

```bash
pre-commit run --all-files   # ruff lint+format, yaml, whitespace
pytest test/unit
tox -e ci-lint               # CI-equivalent lint
tox -e ci-unit               # CI-equivalent tests
```

## CI/CD & Deployment

Patch merge → Jenkins builds Docker image → PipelineBot posts image tag in Gerrit comment → update `operations/deployment-charts` with new tag → manual sync on hosts.

## Adding a New Model

1. Create `src/models/<name>/model_server/model.py`
2. Copy and adapt an existing `blubber.yaml` to `.pipeline/<name>/`
3. Add the service to `docker-compose.yml` for local testing
4. Add pipelines in `.pipeline/config.yaml` (test + publish)
5. Wire CI triggers in `integration/config` repo (`jjb/project-pipelines.yaml` + `zuul/layout.yaml`)

## Architecture

Each model subclasses `kserve.Model` (load/preprocess/predict) and starts via `kserve.ModelServer().start()`. Shared Python utilities in `python/`. Each model has a Blubber config in `.pipeline/`. Served via KServe (Knative + Istio) on K8s.

## Key Env Vars

`MODEL_NAME`, `MODEL_PATH` (/mnt/models/). Some models also use: `FORCE_HTTP`, `NUM_OF_WORKERS`, `BATCH_SIZE`.

## Links

- Gerrit: https://gerrit.wikimedia.org/r/q/project:machinelearning/liftwing/inference-services
- Charts: https://gerrit.wikimedia.org/r/q/project:operations/deployment-charts
- Wiki: https://wikitech.wikimedia.org/wiki/Machine_Learning/LiftWing/Inference_Services
