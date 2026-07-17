# Embedding Models

The embeddings inference service uses vLLM to generate text embeddings for downstream
tasks such as semantic search. The same Docker image serves multiple models via
`MODEL_NAME` / `MODEL_PATH` (and related env vars).

## Supported models

### Qwen3-Embedding

* Model Card: https://github.com/QwenLM/Qwen3-Embedding/blob/44548aa5f0a0aed1c76d64e19afe47727a325b8f/README.md
* Source: https://github.com/QwenLM/Qwen3-Embedding/blob/44548aa5f0a0aed1c76d64e19afe47727a325b8f/examples/qwen3_embedding_vllm.py
* Model: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
* Model license: Apache 2.0 License
* Compose service: `embeddings` (`MODEL_NAME=qwen3-embedding`)

### Jina Embeddings v5 text nano (retrieval)

* Model: https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval
* Base: https://huggingface.co/jinaai/jina-embeddings-v5-text-nano
* Embedding dimension: 768
* Model license: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — confirm WMF commercial-use terms before production deploy
* Compose service: `jina-embeddings` (`MODEL_NAME=jina-embedding`)
* Requires `VLLM_RUNNER=pooling`, `POOLING_TYPE=LAST`, and `TRUST_REMOTE_CODE=True`

> [!WARNING]
> The production Blubber image is based on AMD **vLLM 0.14**. Official Jina nano docs
> mention newer vLLM. If model load fails on architecture or API mismatch, escalate to a
> newer `amd-vllm` base image. Local Mac Docker without an AMD GPU cannot run this image.

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
docker compose build embeddings
```
The `jina-embeddings` service reuses the same `embeddings:prod` image.

### 2.2. Run
By default mounts `./models/jina` (gitignored under `/models/`) at `/mnt/models/`.
Override with `PATH_TO_JINA_EMBEDDINGS_MODEL` if needed. `MODEL_PATH` still points at
Hugging Face unless you set `MODEL_PATH=/mnt/models/` for local weights.

```console
mkdir -p models/jina
docker compose up jina-embeddings
```

### 2.3. Query
OpenAI-compatible request body (`input` required):

```console
curl -s localhost:8080/v1/models/jina-embedding:predict -X POST \
  -H "Content-type: application/json" \
  -d '{"input": ["climate change coastal cities"]}'
```

Expect each embedding vector to have length **768**.

</details>

<details>
<summary>3. Manual setup</summary>

> [!NOTE]
> This model-server is designed to be hosted in a custom-built Docker image that supports vLLM 0.14 and can be found here: https://docker-registry.wikimedia.org/ml/amd-vllm014/tags/
>
> The software stack used in this vLLM image is: ROCm 7.0.0, Torch 2.10.0, MoRi 0.1, FlashAttention 2.8.3, Aiter 0.1.7, and vLLM 0.14. Since we use AMD GPUs on LiftWing, these software packages were built from source to target both MI210 (gfx90a) and MI300X (gfx942) GPUs.
>
> Because these heavy ML dependencies are pre-packaged within the Docker image, the `requirements.txt` file used below only contains the `kserve` dependency needed for the model-server.

### 3.1. Build Python venv and install dependencies
If you are running outside the recommended Docker environment and your system already supports vLLM 0.14, create a virtual environment and install the dependencies using:
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/models/embeddings/requirements.txt
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
DTYPE="float16" \
TRUST_REMOTE_CODE="True" \
VLLM_RUNNER="pooling" \
POOLING_TYPE="LAST" \
python3 src/models/embeddings/model_server/model.py
```

```console
curl -s localhost:8080/v1/models/jina-embedding:predict -X POST \
  -H "Content-type: application/json" \
  -d '{"input": ["climate change coastal cities"]}'
```
</details>
