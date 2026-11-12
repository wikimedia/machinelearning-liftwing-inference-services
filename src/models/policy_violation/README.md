# Policy Violation Detection

The policy violation detection inference services evaluate user content against defined safety policies and classify whether the content violates those policies. Two models are available:

> [!NOTE]
> `gpt-oss-safeguard-20b` no longer has a model server here. It runs as an LLMInferenceService (`llm-gpt-oss-safeguard-20b`) that exposes vLLM's OpenAI-compatible endpoints directly ([T441822](https://phabricator.wikimedia.org/T441822)).

## CoPE-A-9B

* Paper: https://arxiv.org/html/2512.18027v1
* Model: https://huggingface.co/zentropi-ai/cope-a-9b
* Model license: zentropi-openrail-m (gated model)
* Merged model: `s3://wmf-ml-models/cope-a-9b-merged/`

A Gemma 2 9B LoRA fine-tune by Zentropi for policy-agnostic content moderation. Takes a natural language policy definition and content, outputs binary classification (0 = safe, 1 = violation).

CoPE-A-9B is distributed as a LoRA adapter on top of `google/gemma-2-9b`. The base
model and adapter were merged and the resulting model was saved to
`s3://wmf-ml-models/cope-a-9b-merged/`.

<details>
<summary>How the merged model was produced</summary>

#### Prerequisites

Both models are gated on HuggingFace. Before downloading, you must:

1. Accept the [Gemma 2 license](https://huggingface.co/google/gemma-2-9b) (Google)
2. Accept the [CoPE-A-9B license](https://huggingface.co/zentropi-ai/cope-a-9b) (Zentropi, openrail-m)
3. Set `HF_TOKEN` to a HuggingFace access token with read permissions:
   ```bash
   export HF_TOKEN="hf_..."
   ```

#### 1. Download base model and adapter

```bash
python3 -m venv venv
source venv/bin/activate
pip install huggingface_hub

# Download base model (~18GB)
https_proxy="http://webproxy:8080" python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-2-9b')"

# Download LoRA adapter
https_proxy="http://webproxy:8080" python -c \
  "from huggingface_hub import snapshot_download; snapshot_download('zentropi-ai/cope-a-9b')"
```

#### 2. Merge LoRA adapter into base model

```bash
pip install torch peft transformers accelerate

https_proxy="http://webproxy:8080" python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    'google/gemma-2-9b', torch_dtype=torch.bfloat16, device_map='cpu'
)
model = PeftModel.from_pretrained(base, 'zentropi-ai/cope-a-9b')
merged = model.merge_and_unload()

merged.save_pretrained('./cope-a-9b-merged')
AutoTokenizer.from_pretrained('google/gemma-2-9b').save_pretrained('./cope-a-9b-merged')
print('Merged model saved to ./cope-a-9b-merged')
"
```

This needs ~36GB RAM (CPU, no GPU required). Takes a few minutes.

#### 3. Fix tokenizer compatibility

The host `transformers` version saves `extra_special_tokens` as a list, but the
container's version expects a dict. Fix by patching the tokenizer config:

```bash
python -c "
import json
with open('./cope-a-9b-merged/tokenizer_config.json', 'r') as f:
    config = json.load(f)
if 'extra_special_tokens' in config and isinstance(config['extra_special_tokens'], list):
    config['extra_special_tokens'] = {t: t for t in config['extra_special_tokens']}
with open('./cope-a-9b-merged/tokenizer_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('Fixed tokenizer_config.json')
"
```

The merged model was saved to `s3://wmf-ml-models/cope-a-9b-merged/`.

</details>

### How to run locally

Requires at least 17 GiB of free GPU VRAM (BF16 weights). Preferred host: `ml-lab1002`
(AMD Instinct MI210, 64 GiB VRAM).

```bash
docker run --rm -it \
  --network host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add 105 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e MODEL_NAME="cope-a-9b" \
  -e MODEL_PATH="/mnt/models" \
  -e TRUST_REMOTE_CODE="True" \
  -e http_proxy="http://webproxy:8080" \
  -e https_proxy="http://webproxy:8080" \
  -v /path/to/cope-a-9b-merged:/mnt/models \
  -v $(pwd):/srv/app \
  docker-registry.wikimedia.org/ml/amd-vllm014:gfx90agfx942rocm7.0.0pytorch2.10.0mori0.1flash-attn2.8.3aiter0.1.7vllm0.14-3 \
  bash
```

Inside the container:

```bash
cd /srv/app/src/models/policy_violation/cope_a_9b
pip install -r ../requirements.txt
python model.py
```

Query from another terminal:

```bash
curl -s localhost:8080/v1/models/cope-a-9b:predict -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "content": "CLICK HERE TO WIN $10000!!! Visit http://totallylegit.biz NOW!!!",
    "policy": "Content must not contain spam, phishing attempts, or deceptive links."
  }'
```

Notes:
- `--network host`: Needed so the container can resolve `webproxy` for pip and
  expose the KServe server (port 8080) to the host.
- `--group-add 105`: The render device group ID inside the container. Without it,
  PyTorch/HIP cannot initialize the GPU even though `rocm-smi` works.
- `HIP_VISIBLE_DEVICES`: Controls which GPU is visible to the container. Set to
  whichever device has free VRAM (check with `rocm-smi`).

## CoPE-B-A4B

* Paper: https://arxiv.org/abs/2512.18027
* Model Card: https://huggingface.co/zentropi-ai/cope-b-a4b
* Model: https://huggingface.co/zentropi-ai/cope-b-a4b
* Model license: Apache 2.0 License

A Gemma-4-26B-A4B Mixture-of-Experts fine-tune by Zentropi, the 2nd-generation
CoPE content policy evaluator. Like CoPE-A-9B it takes a natural language policy
definition and content and outputs a single-token binary classification
(0 = safe, 1 = violation), but with improved policy steerability and a much
larger context window (256K tokens).

Unlike CoPE-A-9B, this model is distributed under Apache 2.0 with the LoRA
adapter already merged into the base, so no merge step is required: it can be
run directly from the published weights.

> [!NOTE]
> This model-server uses HuggingFace transformers rather than vLLM. The `Gemma4ForCausalLM` architecture (model_type `gemma4_text`) [requires vLLM >= 0.19.0](https://github.com/vllm-project/vllm/pull/39291), which is newer than the vLLM 0.14 in the current `amd-vllm014` base image.
>
> On vLLM 0.14 the model falls back to the generic Transformers-MoE backend and fails to load (it cannot resolve the MoE `top_k` as shown in [P93623](https://phabricator.wikimedia.org/P93623)). Until a vLLM >= 0.19.0 base image is available, this server runs the model using HF transformers (see [P93624](https://phabricator.wikimedia.org/P93624)). This model-server has its own `requirements.txt` (kserve + transformers) rather than sharing `policy_violation/requirements.txt`, so that the vLLM-based CoPE-A-9B server above keeps its lean kserve-only dependency set.

### How to run locally

Requires at least [~52 GiB of free GPU VRAM](https://huggingface.co/zentropi-ai/cope-b-a4b#system-requirements) (BF16 weights). Preferred host: `ml-lab1002` (AMD Instinct MI210, 64 GiB VRAM).

```bash
docker run --rm -it \
  --network host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add 105 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e MODEL_NAME="cope-b-a4b" \
  -e MODEL_PATH="/mnt/models" \
  -e TRUST_REMOTE_CODE="True" \
  -e http_proxy="http://webproxy:8080" \
  -e https_proxy="http://webproxy:8080" \
  -v /path/to/cope-b-a4b:/mnt/models \
  -v $(pwd):/srv/app \
  docker-registry.wikimedia.org/ml/amd-vllm022:gfx90agfx942rocm7.2.0pytorch2.10.0flash-attn2.8.3aiter0.1.13vllm0.22.1-2 \
  bash
```

Inside the container:

```bash
cd /srv/app/src/models/policy_violation/cope_b_a4b
pip install -r requirements.txt
python model.py
```

Query from another terminal:

```bash
curl -s localhost:8080/v1/models/cope-b-a4b:predict -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "content": "CLICK HERE TO WIN $10000!!! Visit http://totallylegit.biz NOW!!!",
    "policy": "Content must not contain spam, phishing attempts, or deceptive links."
  }'
```

Expected response:

```json
{"violation": 1}
```
