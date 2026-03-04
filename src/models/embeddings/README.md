# Embedding Models

The embeddings inference service uses the Qwen3-Embedding model to generate high-quality text embeddings for various downstream tasks, such as similarity computation in semantic search.

* Model Card: https://github.com/QwenLM/Qwen3-Embedding/blob/44548aa5f0a0aed1c76d64e19afe47727a325b8f/README.md
* Source: https://github.com/QwenLM/Qwen3-Embedding/blob/44548aa5f0a0aed1c76d64e19afe47727a325b8f/examples/qwen3_embedding_vllm.py
* Model: https://huggingface.co/Qwen
* Model license: Apache 2.0 License


## How to run locally

In order to run the embeddings model-server locally, please follow the steps below:

<details>
<summary>1. Docker Compose</summary>

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
<summary>2. Manual setup</summary>

> [!NOTE]
> This model-server is designed to be hosted in a custom-built Docker image that supports vLLM 0.14 and can be found here: https://docker-registry.wikimedia.org/ml/amd-vllm014/tags/
>
> The software stack used in this vLLM image is: ROCm 7.0.0, Torch 2.10.0, MoRi 0.1, FlashAttention 2.8.3, Aiter 0.1.7, and vLLM 0.14. Since we use AMD GPUs on LiftWing, these software packages were built from source to target both MI210 (gfx90a) and MI300X (gfx942) GPUs.
>
> Because these heavy ML dependencies are pre-packaged within the Docker image, the `requirements.txt` file used below only contains the `kserve` dependency needed for the model-server.

### 2.1. Build Python venv and install dependencies
If you are running outside the recommended Docker environment and your system already supports vLLM 0.14, create a virtual environment and install the dependencies using:
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/models/embeddings/requirements.txt
```

### 2.2. Run the server
We can run the server locally with:
```console
MODEL_NAME="qwen3-embedding" MODEL_PATH="Qwen/Qwen3-Embedding-0.6B" MAX_MODEL_LEN="8192" DTYPE="float16" TRUST_REMOTE_CODE="True" python3 src/models/embeddings/model_server/model.py
```

On a separate terminal we can make a request to the server with:
```console
curl -s localhost:8080/v1/models/qwen3-embedding:predict -X POST -d '{"input": ["text1", "text2"]}' -i -H "Content-type: application/json"
```
</details>
