# Embedding Models

The embeddings inference service uses the Qwen3-Embedding model to generate high-quality text embeddings for various downstream tasks, such as similarity computation in semantic search.

* Model Card: https://github.com/QwenLM/Qwen3-Embedding/blob/44548aa5f0a0aed1c76d64e19afe47727a325b8f/README.md
* Source: https://github.com/QwenLM/Qwen3-Embedding/blob/44548aa5f0a0aed1c76d64e19afe47727a325b8f/examples/qwen3_embedding_transformers.py
* Model: https://huggingface.co/Qwen
* Model license: Apache 2.0 License


## How to run locally

In order to run the qwen3-embedding model-server locally, please follow the steps below:

### Build Python venv and install dependencies
Create a virtual environment and install the dependencies using:
```console
python3 -m venv .venv
source .venv/bin/activate

# on linux
pip install -r src/models/embeddings/requirements.txt

# on macos since it doesn't support rocm-specific packages like fa2
pip install -r src/models/embeddings/requirements-macos.txt
```

### Run the server
We can run the server locally with:
```console
# on linux
MODEL_NAME=qwen3-embedding MODEL_PATH="Qwen/Qwen3-Embedding-0.6B" MAX_LENGTH="300" LOCAL_FILES_ONLY="False" DTYPE="float16" ATTN_IMPLEMENTATION="flash_attention_2" python3 src/models/embeddings/model_server/model.py

# on macos since it doesn't support rocm-specific packages like fa2
MODEL_NAME=qwen3-embedding MODEL_PATH="Qwen/Qwen3-Embedding-0.6B" MAX_LENGTH="300" LOCAL_FILES_ONLY="False" DTYPE="float32" ATTN_IMPLEMENTATION="eager" python3 src/models/embeddings/model_server/model.py
```

On a separate terminal we can make a request to the server with:
```console
curl -s localhost:8080/v1/models/qwen3-embedding:predict -X POST -d '{"input": ["text1", "text2"]}' -i -H "Content-type: application/json"
```
