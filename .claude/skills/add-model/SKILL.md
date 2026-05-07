---
name: add-model
description: Scaffold a new KServe model server from scratch — create the model.py, Blubber config, Docker Compose service, pipeline config, and CI wiring. Use when the engineer wants to add a new inference service model to this repo.
argument-hint: "[model-name]"
allowed-tools:
  - Bash(mkdir *)
  - Bash(cp *)
  - Read
  - Write
  - Edit
  - Glob
  - Grep
---

# Add a New Model — Inference Services

Scaffold all the files needed for a new model server.

## Steps

1. Create `src/models/<name>/model_server/` directory.
2. Create `src/models/<name>/model_server/model.py` — subclass `kserve.Model` with `load()`, `preprocess()`, `predict()`, and a `if __name__ == "__main__"` block that reads `MODEL_NAME` and `MODEL_PATH` from the environment and starts `kserve.ModelServer().start([model])`. Reference an existing model like `revert_risk_model` for the pattern.
3. Create `src/models/<name>/model_server/requirements.txt` with the model's Python dependencies. The shared `python/requirements.txt` is always installed alongside.
4. Copy and adapt an existing `blubber.yaml` to `.pipeline/<name>/`. Use a CPU model (e.g., `revertrisk/revertrisk.yaml`) as the template unless this model needs a GPU/pyTorch base image.
5. Add the service to `docker-compose.yml`: `platform: linux/amd64`, build from `.pipeline/<name>/blubber.yaml`, expose port `8080`, set `MODEL_NAME`, mount `/mnt/models/` from a `PATH_TO_<NAME>_MODEL` env var.
6. Add two entries in `.pipeline/config.yaml`: a `<name>` pipeline (test + production stages) and a `<name>-publish` pipeline (publish to registry).
7. Create unit tests in `test/unit/<name>/`.
8. Wire CI triggers in the `integration/config` repo: `jjb/project-pipelines.yaml` and `zuul/layout.yaml`.
9. If the model needs local Makefile support, add a target to the `Makefile`.

## Notes

- Model artifacts are hosted at `https://analytics.wikimedia.org/published/wmf-ml-models/`.
- The shared Python library at `python/` (decorators, preprocess utils, metrics) is available to all models.
- Most models need a `.env` file locally with `PATH_TO_<NAME>_MODEL` pointing to downloaded model files.

## Input

$ARGUMENTS — the model name in kebab-case (e.g., `article-quality`, `revert-risk-wikidata`).
