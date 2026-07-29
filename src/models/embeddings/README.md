# Embedding Models

Two backends serve text embeddings for downstream tasks such as semantic search:

* **Qwen3-Embedding** — vLLM ([`model.py`](model_server/model.py)), image from
  [`.pipeline/embeddings/blubber.yaml`](../../.pipeline/embeddings/blubber.yaml)
* **Jina Embeddings v5** — SentenceTransformers
  ([`model_transformers.py`](model_server/model_transformers.py)), image from
  [`.pipeline/embeddings/blubber_transformers.yaml`](../../.pipeline/embeddings/blubber_transformers.yaml)

## Supported models

### Qwen3-Embedding

* Model Card: https://github.com/QwenLM/Qwen3-Embedding/blob/44548aa5f0a0aed1c76d64e19afe47727a325b8f/README.md
* Source: https://github.com/QwenLM/Qwen3-Embedding/blob/44548aa5f0a0aed1c76d64e19afe47727a325b8f/examples/qwen3_embedding_vllm.py
* Model: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
* Model license: Apache 2.0 License
* Compose service: `embeddings` (`MODEL_NAME=qwen3-embedding`)
* Backend: vLLM (`model_server/model.py`)

### Jina Embeddings v5 text nano (retrieval)

* Model: https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval
* Base: https://huggingface.co/jinaai/jina-embeddings-v5-text-nano
* Embedding dimension: 768
* Model license: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — confirm WMF commercial-use terms before production deploy
* Compose service: `jina-embeddings` (`MODEL_NAME=jina-embedding`)
* Backend: SentenceTransformers (`model_server/model_transformers.py`)
* Optional request field `prompt_name` (`"query"` or `"document"`); defaults to
  env `PROMPT_NAME` (default `"query"`)

## How to run locally

In order to run the embeddings model-server locally, please follow the steps below:

<details>
<summary>1. Docker Compose (Qwen3)</summary>

### 1.1. Build
In the first terminal run:
```console
docker compose build embeddings
```
This will build an embeddings image with all dependencies installed.

### 1.2. Run
On the the same terminal run the model-server:
```console
docker compose up embeddings
```

### 1.3. Query
On the second terminal query the isvc using:
```console
curl -s localhost:8080/v1/models/qwen3-embedding:predict -X POST -d '{"input": ["text1", "text2"]}' -i -H "Content-type: application/json"
```

### 1.4. Remove
If you would like to remove the setup run:
```console
docker compose down -v --rmi all
```

</details>

<details>
<summary>2. Docker Compose (Jina)</summary>

### 2.1. Build
```console
docker compose build jina-embeddings
```
This builds the SentenceTransformers image (`embeddings-transformers:prod`).

### 2.2. Run
By default mounts `./models/jina` (gitignored under `/models/`) at `/mnt/models/`.
Override with `PATH_TO_JINA_EMBEDDINGS_MODEL` if needed. `MODEL_PATH` still points at
Hugging Face unless you set `MODEL_PATH=/mnt/models/` for local weights.

```console
mkdir -p models/jina
docker compose up jina-embeddings
```

### 2.3. Query
OpenAI-compatible request body (`input` required). Optional `prompt_name`
overrides the default retrieval prompt (`query` / `document`):

```console
curl -s localhost:8080/v1/models/jina-embedding:predict -X POST \
  -H "Content-type: application/json" \
  -d '{"input": ["climate change coastal cities"], "prompt_name": "query"}'
```

Expect each embedding vector to have length **768**.

</details>

<details>
<summary>3. Manual setup</summary>

> [!NOTE]
> The Qwen3 model-server is designed to be hosted in a custom-built Docker image that supports vLLM 0.22 and can be found here: https://docker-registry.wikimedia.org/ml/amd-vllm022/tags/
>
> The software stack used in this vLLM image is: ROCm 7.2.0, Torch 2.10.0, FlashAttention 2.8.3, Aiter 0.1.13, and vLLM 0.22. Since we use AMD GPUs on LiftWing, these software packages were built from source to target both MI210 (gfx90a) and MI300X (gfx942) GPUs.
>
> The Jina SentenceTransformers image reuses the same AMD base for torch/flash-attn, but never imports vLLM — see `blubber_transformers.yaml`.

### 3.1. Build Python venv and install dependencies
If you are running outside the recommended Docker environment:

**Qwen3 (vLLM):**
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/models/embeddings/requirements.txt
```

**Jina (SentenceTransformers):**
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/models/embeddings/requirements_transformers.txt
```

### 3.2. Run Qwen3
```console
MODEL_NAME="qwen3-embedding" MODEL_PATH="Qwen/Qwen3-Embedding-0.6B" MAX_MODEL_LEN="8192" DTYPE="float16" TRUST_REMOTE_CODE="True" python3 src/models/embeddings/model_server/model.py
```

```console
curl -s localhost:8080/v1/models/qwen3-embedding:predict -X POST -d '{"input": ["text1", "text2"]}' -i -H "Content-type: application/json"
```

### 3.3. Run Jina
```console
MODEL_NAME="jina-embedding" \
MODEL_PATH="jinaai/jina-embeddings-v5-text-nano-retrieval" \
DTYPE="bfloat16" \
ATTN_IMPLEMENTATION="eager" \
PROMPT_NAME="query" \
python3 src/models/embeddings/model_server/model_transformers.py
```

```console
curl -s localhost:8080/v1/models/jina-embedding:predict -X POST \
  -H "Content-type: application/json" \
  -d '{"input": ["climate change coastal cities"], "prompt_name": "query"}'
```
</details>
