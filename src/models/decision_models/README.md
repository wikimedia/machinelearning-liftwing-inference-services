# Decision models

Decision models take a state and a schema of typed questions, and return a probability for every allowed option of every question in a single forward pass. There is no text generation and nothing to parse out of a reply.

## Clef

* Model cards:
  * https://huggingface.co/Cloudflare/clef
  * https://huggingface.co/Cloudflare/clef-flash
* Model license: Apache 2.0 License

Clef and Clef-Flash are Apache-2.0 multimodal decision models from Cloudflare, post-trained from Qwen3.8-27B and Qwen3.5-9B. Unlike the CoPE models, a Clef model does not generate text: it takes a state and a schema of typed questions (`noul`, `choice`, `score`) and returns a probability for every allowed option of every question in a single forward pass. Both read text, JSON or images.

One model-server serves either Clef or Clef-Flash since they have the same files, loader and API, and only the backbone differs. Point `MODEL_PATH` at the weights and set `MODEL_NAME` to match.

> [!NOTE]
> This model-server uses HuggingFace transformers rather than vLLM. A Clef release is a Qwen backbone plus a separate joint schema head (`joint_head.safetensors`), loaded by a vendor [joint_schema_model.py](https://huggingface.co/Cloudflare/clef-flash/blob/41318bf2106d135c7a92da745fec4b255279b4de/joint_schema_model.py) that ships with the weights. vLLM serves backbones and has no concept of a scoring head attached to one, so this is not a case of waiting for an architecture to be added upstream. The module is imported from the model directory at load time, so this server runs vendor code in the server process. This model-server has its own `requirements.txt` (kserve + transformers). For more details see [RFC for a decision endpoint in vLLM](https://github.com/vllm-project/vllm/issues/59365).

### How to run locally

Requires at least ~20 GiB of free GPU VRAM (BF16 weights) for clef-flash; the 27B clef backbone needs roughly 55 GiB. Preferred host: `ml-lab1002` (AMD Instinct MI210, 64 GiB VRAM).

```bash
docker run --rm -it \
  --network host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add 105 \
  -e HIP_VISIBLE_DEVICES=0 \
  -e MODEL_NAME="clef-flash" \
  -e MODEL_PATH="/mnt/models" \
  -e http_proxy="http://webproxy:8080" \
  -e https_proxy="http://webproxy:8080" \
  -v /path/to/clef-flash:/mnt/models \
  -v $(pwd):/srv/app \
  docker-registry.wikimedia.org/ml/amd-vllm022:gfx90agfx942rocm7.2.0pytorch2.10.0flash-attn2.8.3aiter0.1.13vllm0.22.1-3 \
  bash
```

Inside the container:

```bash
cd /srv/app/src/models/decision_models/clef
pip install -r requirements.txt
python3 model.py
```

Query from another terminal:

```bash
curl -s localhost:8080/v1/models/clef-flash:predict -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Edit replaced the article lead with promotional language.",
    "questions": {
      "revert": {"type": "noul", "instructions": "Should this edit be reverted?"},
      "severity": {"type": "score", "criteria": ["Minor", "Moderate", "Severe"]}
    }
  }'
```

Expected response:

```json
{
  "model": "clef-flash",
  "answers": {
    "revert": {"type":"noul", "noul":0.9325},
    "severity": {
      "type":"score", "score":1.4356, "confidence":0.5363,
      "legend": {"0":"Minor", "1":"Moderate", "2":"Severe"},
      "probabilities": {"0":0.1008, "1":0.3629, "2":0.5363}
    }
  },
  "usage": {"input_tokens":228, "output_tokens":0}
}
```
