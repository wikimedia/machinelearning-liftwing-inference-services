# Policy Violation Detection

The policy violation detection inference services evaluate user content against defined safety policies and classify whether the content violates those policies. Two models are available:

## gpt-oss-safeguard-20b

* Model Card: https://arxiv.org/pdf/2508.10925
* Source: https://github.com/roostorg/model-community/tree/142367c67cd8a1d83e293dd0985b50fff0f05b54/gpt
* Model: https://huggingface.co/openai/gpt-oss-safeguard-20b
* Model license: Apache 2.0 License

Uses OpenAI's Harmony encoding for structured input/output with reasoning and verdict channels.

### How to run locally

> [!NOTE]
> This model-server is designed to be hosted in a custom-built Docker image that supports vLLM 0.14 and can be found here: https://docker-registry.wikimedia.org/ml/amd-vllm014/tags/
>
> The software stack used in this vLLM image is: ROCm 7.0.0, Torch 2.10.0, MoRi 0.1, FlashAttention 2.8.3, Aiter 0.1.7, and vLLM 0.14. Since we use AMD GPUs on LiftWing, these software packages were built from source to target both MI210 (gfx90a) and MI300X (gfx942) GPUs.
>
> Because these heavy ML dependencies are pre-packaged within the Docker image, the `requirements.txt` file used below only contains the `kserve` dependency needed for the model-server.

If you are running outside the recommended Docker environment and your system already supports vLLM 0.14, create a virtual environment and install the dependencies using:
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/models/policy_violation/requirements.txt
```

Run the server:
```console
MODEL_NAME=policy-violation MODEL_PATH="openai/gpt-oss-safeguard-20b" TRUST_REMOTE_CODE="True" python3 src/models/policy_violation/gpt_oss_safeguard_20b/model.py
```

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
  docker-registry.wikimedia.org/ml/amd-vllm014:gfx90agfx942rocm7.0.0pytorch2.10.0mori0.1flash-attn2.8.3aiter0.1.7vllm0.14 \
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
