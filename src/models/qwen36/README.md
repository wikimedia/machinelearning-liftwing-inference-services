# Qwen 3.6-27B Inference Service

Model server for [Qwen 3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), deployed via KServe on Lift Wing.

## Models

| Use | Model | HF Link |
|-----|-------|---------|
| Production | Qwen3.6-27B-FP8 (official, ~30 GB, TP=2) | https://huggingface.co/Qwen/Qwen3.6-27B-FP8 |
| Local CPU testing | Qwen3-0.6B (same architecture, ~0.6 GB) | https://huggingface.co/Qwen/Qwen3-0.6B |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- For local CPU testing: model is downloaded automatically via `MODEL_PATH` HF ID (optional local download shown below)

### 1. (Optional) Download the model for offline CPU testing

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./models/qwen3-0.6b
```

Add the path to your `.env` file to use a local copy instead of downloading at runtime:

```bash
echo "PATH_TO_QWEN36_CPU_MODEL=$(pwd)/models/qwen3-0.6b" >> .env
```

### 2. Build and run locally (CPU)

```bash
docker compose build qwen36-27b-cpu
docker compose up qwen36-27b-cpu
```

### 3. Test the model server

**KServe v1 protocol:**

```bash
curl localhost:8080/v1/models/qwen3-0.6b:predict -X POST \
  -d '{"prompt":"What is 2+2?","max_tokens":50}' \
  -H "Content-type: application/json"
```

With reasoning enabled:

```bash
curl localhost:8080/v1/models/qwen3-0.6b:predict -X POST \
  -d '{"prompt":"Explain quantum computing in one sentence","max_tokens":100,"reasoning":true}' \
  -H "Content-type: application/json"
```

**OpenAI-compatible completions (standard):**

```bash
curl localhost:8080/openai/v1/completions -X POST \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-0.6b","prompt":"What is 2+2?","max_tokens":50}'
```

**OpenAI-compatible completions (streaming):**

```bash
curl -N localhost:8080/openai/v1/completions -X POST \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-0.6b","prompt":"What is 2+2?","max_tokens":50,"stream":true}'
```

**OpenAI-compatible chat completions (streaming):**

```bash
curl -N localhost:8080/openai/v1/chat/completions -X POST \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50,"stream":true}'
```

Streaming responses use SSE (`text/event-stream`). Omit `"stream":true` (and the `-N` flag) for standard non-streaming responses.

### Thinking vs Instruct mode

Qwen 3.6 supports two generation modes:

| Mode | Behavior | Best for |
|------|----------|----------|
| **Instruct** (default) | Direct answer, no reasoning trace | Factual Q&A, classification, summarization |
| **Thinking** | Generates a reasoning trace inside `<think>...</think>` before answering | Math, logic, multi-step reasoning |

**KServe v1 endpoint** — toggle via the `reasoning` field:

```bash
# Instruct mode (default)
curl localhost:8080/v1/models/qwen3-0.6b:predict -X POST \
  -d '{"prompt":"What is 2+2?","max_tokens":50}' \
  -H "Content-type: application/json"

# Thinking mode
curl localhost:8080/v1/models/qwen3-0.6b:predict -X POST \
  -d '{"prompt":"What is 2+2?","max_tokens":50,"reasoning":true}' \
  -H "Content-type: application/json"
```

**OpenAI-compatible endpoints** — instruct mode is the default. There is currently no `reasoning` flag on the OpenAI endpoints; use the KServe v1 endpoint for thinking mode.

Sampling parameters are automatically adjusted per mode based on the [model card recommendations](https://huggingface.co/Qwen/Qwen3.6-27B-FP8#recommended-sampling-parameters):

| Parameter | Instruct | Thinking |
|-----------|----------|----------|
| `temperature` | 0.7 | 1.0 |
| `top_p` | 0.8 | 0.95 |
| `presence_penalty` | 1.5 | 0.0 |

### 4. Run unit tests

```bash
# Run just the qwen36 tests
python -m pytest test/unit/qwen36/ -v

# Run the full CI suite
tox -e ci-unit
```

### 5. Run linting

```bash
# Lint and format check
ruff check src/models/qwen36/
ruff format --check src/models/qwen36/

# Full CI lint (includes pre-commit hooks)
tox -e ci-lint
```

### 6. GPU production build (AMD machine)

```bash
docker compose build qwen36-27b
docker compose up qwen36-27b
```

## Request Format

```json
{
  "prompt": "Explain quantum computing",
  "max_tokens": 2048,
  "temperature": 0.7,
  "top_p": 0.8,
  "reasoning": true
}
```

All fields except `prompt` are optional.

## Response Format

```json
{
  "model_name": "qwen36-27b",
  "response": "...",
  "prompt_tokens": 25,
  "completion_tokens": 200
}
```

## Architecture

- Uses vLLM `AsyncLLMEngine` for continuous batching and streaming
- Reasoning (thinking) mode toggled per-request via the `reasoning` flag and the chat template's `enable_thinking` parameter
- GPU production: 2x AMD GPU partitions via tensor parallelism (TP=2)
- Local CPU: using `vllm/vllm-openai-cpu:v0.16.0` base (Qwen/Qwen3-0.6B for testing)
