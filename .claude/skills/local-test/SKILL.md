---
name: local-test
description: Build and run a model server locally via Docker Compose, then test it with curl. Use when the engineer wants to test a model server locally before committing.
argument-hint: "[service-name]"
allowed-tools:
  - Bash(docker *)
  - Bash(curl *)
  - Read
---

# Local Test — Inference Services

Build, run, and test a model server locally.

## Steps

1. Ensure a `.env` file exists in the repo root with `PATH_TO_<MODEL>_MODEL=/local/path/to/model`.
2. Build the image: `docker compose build <service>`.
3. Start the service: `docker compose up <service>`.
4. Once it's running, send a test request. The endpoint is `POST /v1/models/<model-name>:predict` where `<model-name>` is the `MODEL_NAME` env var set for that service in docker-compose.yml. Check `src/models/<service>/model_server/model.py` for the expected input schema. Example for revert-risk language agnostic model: `curl localhost:8080/v1/models/revertrisk-language-agnostic:predict -X POST -d '{"lang":"en","rev_id":12345}' -H "Content-type: application/json"`.
5. To stop: `Ctrl+C` or `docker compose down`.

## Notes

- **ARM Macs**: If the service doesn't start, add `platform: linux/amd64` to the service in `docker-compose.yml`.
- The service listens on `localhost:8080` by default. Check docker-compose.yml if the port is mapped differently. If `:8080` is already in use, stop the other service or remap the port.
- Model files must be downloaded separately from `https://analytics.wikimedia.org/published/wmf-ml-models/` and placed at the path referenced in `.env`.
- To run detached: `docker compose up -d <service>`. Logs: `docker compose logs <service>`.

## Input

$ARGUMENTS — the docker-compose service name (e.g., `revertrisk-language-agnostic`, `articlequality`, `reference-need`).
