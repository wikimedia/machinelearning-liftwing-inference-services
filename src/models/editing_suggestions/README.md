# Editing Suggestions

The editing-suggestions inference service returns pre-computed editing suggestions for a Wikipedia page, keyed by `wiki_id` and `page_id`. Suggestions are loaded from a CSV model artifact at startup and served via in-memory lookup.

* Model: rule-based lookup (no model binary)
* Data artifact: `s3://wmf-ml-models/editing-suggestions/v1/suggestions.csv`
* Published at: https://analytics.wikimedia.org/published/wmf-ml-models/editing-suggestions/v1/suggestions.csv

## API

**Endpoint:** `POST /v1/models/editing-suggestions:predict`

**Request:**
```json
{
  "wiki_id": "enwiki",
  "page_id": 81880701
}
```

**Response:**
```json
{
  "suggestions": [
    {
      "revision_id": 1328789391,
      "page_title": "2025_Taipei_stabbings",
      "page_id": 81880701,
      "suggestion_type": "simplify_language",
      "description": "Divide the compound sentence into two separate sentences to make the information easier to read.",
      "target": "A nationwide task force was formed to investigate these cases, and so far, three suspects have been caught and are set to face legal consequences.",
      "wiki_id": "enwiki",
      "suggestion_id": "8a8acacf-97c5-4e85-99f6-80d5920e012e",
      "static_description": "Simplify language by keeping the meaning but making the wording clearer, shorter, or easier to read, such as replacing jargon or complex sentences with plain language",
      "title": "Simplify language"
    }
  ]
}
```

Returns `{"suggestions": []}` when no suggestions exist for the given wiki/page. `wiki_id` and `page_id` must match the CSV exactly.

## How to run locally

Download the suggestions CSV from [analytics.wikimedia.org](https://analytics.wikimedia.org/published/wmf-ml-models/editing-suggestions/v1/suggestions.csv) and place it in a local directory as `suggestions.csv`.

<details>
<summary>1. Docker Compose (recommended)</summary>

### 1.1. Build and run

From the repo root, add the model directory to `.env`:

```console
PATH_TO_EDITING_SUGGESTIONS_MODEL=/path/to/dir/containing/suggestions.csv
```

Then build and run:

```console
docker compose build editing-suggestions
docker compose up editing-suggestions
```

On ARM Macs, the service already declares `platform: linux/amd64` in `docker-compose.yml`.

The directory mounted at `/mnt/models/` must contain `suggestions.csv`. For a quick local setup, copy or symlink from the test fixture:

```console
mkdir -p models/editing-suggestions/v1
cp src/models/editing_suggestions/data/suggestions_2026_06_01.csv models/editing-suggestions/v1/suggestions.csv
echo 'PATH_TO_EDITING_SUGGESTIONS_MODEL=./models/editing-suggestions/v1' >> .env
```

### 1.2. Query

In a second terminal:

```console
curl -s localhost:8080/v1/models/editing-suggestions:predict -X POST \
  -H "Content-Type: application/json" \
  -d '{"wiki_id": "enwiki", "page_id": 81880701}'
```

Example with no matches:

```console
curl -s localhost:8080/v1/models/editing-suggestions:predict -X POST \
  -H "Content-Type: application/json" \
  -d '{"wiki_id": "enwiki", "page_id": 99999999}'
```

### 1.3. Stop

```console
docker compose down editing-suggestions
```
</details>

<details>
<summary>2. Manual setup (Python venv)</summary>

### 2.1. Install dependencies

From the repo root:

```console
export PYTHONPATH=$PYTHONPATH:.
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/models/editing_suggestions/requirements.txt
pip install -r python/requirements.txt
```

### 2.2. Run the server

**Run from the repo root** (not from `src/models/editing_suggestions/`). The server imports shared code from `python/` via `PYTHONPATH`.

```console
cd /path/to/inference-services
export PYTHONPATH=$PYTHONPATH:.
MODEL_NAME=editing-suggestions \
MODEL_PATH=/path/to/suggestions.csv \
python3 src/models/editing_suggestions/model_server/model.py
```

Wait for `Registering model: editing-suggestions` in the logs before sending requests.

If port 8080 is already in use (e.g. by another model server), pick another port:

```console
HTTP_PORT=8082 MODEL_NAME=editing-suggestions \
MODEL_PATH=/path/to/suggestions.csv \
python3 src/models/editing_suggestions/model_server/model.py
```

### 2.3. Query

In a second terminal (use the same port you started the server on):

```console
curl -s localhost:8080/v1/models/editing-suggestions:predict -X POST \
  -H "Content-Type: application/json" \
  -d '{"wiki_id": "enwiki", "page_id": 81880701}'
```

If you used `HTTP_PORT=8082`, change the URL to `localhost:8082`.
</details>

## Troubleshooting

### `Model with name editing-suggestions does not exist`

This usually means your request hit a **different** server on that port, not the editing-suggestions server.

1. Check which models are registered:
   ```console
   curl -s localhost:8080/v1/models
   ```
   You should see `{"models":["editing-suggestions"]}`. If you see another model name (e.g. `qwen3-embedding`), port 8080 is taken.

2. Confirm the editing-suggestions server is running and logged `Registering model: editing-suggestions`. If you see `address already in use`, the server failed to start — use another port:
   ```console
   HTTP_PORT=8082 MODEL_NAME=editing-suggestions \
   MODEL_PATH=/path/to/suggestions.csv \
   python3 src/models/editing_suggestions/model_server/model.py
   ```

3. Stop the other service using port 8080, or always use a dedicated port for local testing.

### `ModuleNotFoundError: No module named 'python'`

Run from the **repo root** with `PYTHONPATH=.` set (see section 2.2 above). Do not run `python model_server/model.py` from inside `src/models/editing_suggestions/`.

### `FileNotFoundError` on startup

The service reads the CSV from `MODEL_PATH` (default `/mnt/models/suggestions.csv`). Ensure the file exists at that path before starting the server.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `editing-suggestions` | KServe model name (used in the predict URL) |
| `MODEL_PATH` | `/mnt/models/suggestions.csv` | Local path to the suggestions CSV loaded at startup |
| `HTTP_PORT` | `8080` | HTTP port for the local model server |

For Docker Compose, also set `PATH_TO_EDITING_SUGGESTIONS_MODEL` in `.env` to the host directory mounted at `/mnt/models/`.

## Tests

```console
pytest test/unit/editing_suggestions
```

Unit tests use the bundled fixture at `data/suggestions_2026_06_01.csv`; production loads from `MODEL_PATH`.
