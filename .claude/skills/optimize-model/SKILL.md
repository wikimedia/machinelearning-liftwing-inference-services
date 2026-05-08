---
name: optimize-model
description: Analyze a model server's Python code and its deployment chart to identify performance bottlenecks and optimization opportunities for CPU and GPU (AMD MI300X, ROCm, vLLM) inference services on KServe. Use when you want to improve inference throughput or latency for a model.
argument-hint: "[model-name]"
allowed-tools:
  - Read
  - Glob
  - Grep
  - WebFetch
---

# Optimize Model — Inference Services

Analyze a model server's code and deployment config for performance optimization opportunities. Covers the full stack: model code → inference backend tuning → GPU/CPU
settings → KServe config → K8s resources.

## Steps

1. Read `src/models/<model-name>/model_server/model.py` (or `src/models/<model-name>/model.py`) to understand the model class, env vars, and runtime configuration.
2. Read `src/models/<model-name>/README.md` for the model card, which documents the model architecture, training framework, hardware requirements, and expected
input/output schema. Use this to confirm whether the model is CPU or GPU, which inference backend it uses, and any documented performance expectations.
3. Read `src/models/<model-name>/requirements.txt` and `python/requirements.txt` to check dependency constraints.
4. Fetch the deployment chart from `operations/deployment-charts` at `helmfile.d/ml-services/<chart-name>/values.yaml` and environment overrides
(`values-ml-serve-eqiad.yaml` for prod).
5. Fetch upstream best practices to use as a baseline for the analysis:
   - If the model uses vLLM: identify the model family from the HuggingFace model ID in `model.py` or the README model card (e.g., GPT-Oss → `gpt-oss`, Qwen → `qwen`,
   Llama → `llama`) and fetch `https://recipes.vllm.ai/{family}/{model-name}` for the recommended vLLM configuration.
   - If the model runs on AMD GPUs (MI300X, MI250, MI210): fetch the ROCm vLLM optimization guide at
   `https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html` and the AITER README at
   `https://github.com/ROCm/aiter` for known issues and tuning recommendations.
   - If the model uses HF Transformers (no vLLM): fetch the HuggingFace model card for the model to check for recommended inference settings.
   - Cross-reference upstream recommendations against the current configuration found in the model code and deployment chart.
6. Determine the optimization scope based on the model's hardware target:
   - If the model is **CPU-only** (no GPU resources in chart, no CUDA/ROCm imports in code): ask the user whether they want to optimize for CPU only, or also assess
   GPU migration potential. If CPU-only, skip the vLLM and ROCm sections below and focus on KServe workers, external API optimization, K8s CPU/memory/HPA, and Python
   patterns.
   - If the model is **GPU** (has `amd.com/gpu` or `nvidia.com/gpu` in chart, imports vLLM/ROCm in code): run all sections below.
7. Analyze across the following dimensions (scope per step 6):

### Inference Backend
- Which inference backend is used (HF Transformers, vLLM, custom)? If using HF Transformers, flag vLLM+AITER migration as a potential improvement for throughput and
memory efficiency via PagedAttention.
- For vLLM models: is `AsyncLLMEngine` used for non-blocking generation? Falls back to synchronous generation?
- Model quantization: int4, int8, float16? Matches the hardware capability (MI300X for GPU, AVX/AMX for CPU)?
- **CPU models**: is ONNX Runtime, OpenVINO, or TorchServe used? These can offer better CPU throughput than vanilla HF Transformers.

### vLLM Configuration *(GPU only)*
- `tensor_parallel_size` — matches GPU count in K8s resources? For MI300X, TP=2 is typical for 20B+ models; check if it can be increased.
- `gpu_memory_utilization` — default 0.95 is aggressive; check for OOM risk vs underutilization.
- `max_num_seqs` / `max_num_batched_tokens` — balanced for the model's expected concurrency?
- `block_size` — 64 is common for AMD; 128 can be more efficient for long sequences.
- `max_model_len` — matches actual usage patterns? Overprovisioning wastes GPU memory.
- `enable_prefix_caching` — enabled in code? Reduces latency for repeated prompts.
- `attention_backend` — `ROCM_AITER_UNIFIED_ATTN` requires `VLLM_ROCM_USE_AITER=1`. Check consistency.
- `compilation_config` — `use_inductor_graph_partition` enabled? Missing tuning?
- `disable_custom_all_reduce` — should be `True` for >1 GPU on MI300X in some topologies.
- Inter-GPU communication topology: P2P vs SHM vs P2P+SHM? Check if `NCCL_P2P_LEVEL`, `/dev/shm` sizing, and `HSA_FORCE_FINE_GRAIN_PCIE` are correctly configured for
the chosen topology.

### ROCm / GPU Tuning *(GPU only)*
- `VLLM_ROCM_USE_AITER` — set to `1` for flash attention on MI300X?
- `TORCH_BLAS_PREFER_HIPBLASLT` — set to `1` for optimized GEMM kernels on MI300X.
- `HSA_FORCE_FINE_GRAIN_PCIE` — set to `1` when using P2P for fine-grained PCIe access.
- `HSA_ENABLE_IPC_MODE_LEGACY` — `0` for P2P mode, `1` for IPC/SHM fallback.
- NCCL/RCCL vars: `NCCL_P2P_DISABLE`, `NCCL_SOCKET_IFNAME`, `RCCL_MSCCL_ENABLE`.
- Shared memory (`/dev/shm`) — configured as `emptyDir` with adequate `sizeLimit` when TP>1?
- `TORCH_SYMM_MEM_DISABLE_MULTICAST` — set for P2P stability?
- Build-time vs runtime compilation: are ROCm device bitcode libraries included in the Docker image to avoid lazy runtime compilation on first inference?

### External Dependencies
- Does preprocessing make external API calls (MediaWiki, Wikidata, etc.)? Are they async or blocking?
- **Caching**: are external API responses cached to avoid redundant calls on repeated inputs? Check for TTL, max size, and eviction policy.
- **Retry logic**: is there exponential backoff or retry for transient failures in external calls? Does the code distinguish transient vs permanent errors?
- **Connection pooling**: are HTTP sessions reused or created per-request? Check for `mwapi.AsyncSession` or `aiohttp.ClientSession` reuse patterns.
- **Timeouts**: are timeouts configured for external calls? Missing timeouts can cause requests to hang indefinitely.
- **Error taxonomy**: are errors categorized (network timeout, API overload, bad entity ID, type error) and handled differently per category?

### KServe / Model Server Config
- `workers=` in `ModelServer(workers=N)` — for CPU I/O-bound models, increasing workers can significantly improve throughput (e.g., 8 workers per pod for
Wikidata-heavy models). For GPU models, more workers risk OOM; test incrementally.
- Model loading: does `load()` download or read local path? Cold start time matters.
- Preprocessing: async or sync? Blocking calls stall the event loop.
- Batching: is vLLM continuous batching configured? Check `max_num_seqs`. For CPU models, does KServe's built-in batching help?
- Timeouts: any request timeout settings that might cut off long inferences?

### Python / Code Patterns
- Model class structure: clean separation of `load`, `preprocess`, `predict`, `postprocess`?
- Caching: repeated computation in preprocessing (e.g., fetching article HTML every request)?
- Error handling: graceful degradation vs hard failures? Are there fallback paths for external API failures?
- Logging: excessive logging in hot path? KServe log level configuration?
- **CPU models**: is there GIL contention from threading? Would `NUM_OF_WORKERS` with multiple processes help?
- Memory management: are large objects (models, caches) properly scoped? Any risk of memory leaks under sustained load?

### Kubernetes Resources
- CPU/memory ratio: GPU models typically need 8-16 CPU and 16-80Gi memory per GPU. CPU models need CPU/memory proportional to worker count and external API concurrency.
- GPU count: matches `tensor_parallel_size` in vLLM config? *(GPU only)*
- Node affinity: GPU models pinned to MI300X nodes? Check `dedicated=mi300x-experiments` toleration. CPU models: any node constraints?
- Scaling: `maxReplicas` — GPU models often pinned to 1; is HPA configured? For CPU models, is autoscaling configured based on RPS or CPU utilization? Check
stabilization window, scale-down behavior, and pod disruption budget.
- Shared memory volume: present and correctly sized for TP>1? *(GPU only)*
- Resource quota: does the namespace have sufficient headroom for the configured replicas × resources per pod?

8. Produce a structured report with:
   - **Findings** organized by layer (Inference Backend, vLLM/ROCm [if GPU], External Dependencies, KServe, Python, K8s)
   - **Severity** for each finding: critical / major / suggestion
   - **Rationale** explaining why it matters for your stack
   - **Recommended change** with concrete code or config diff

## Notes

- Model code is in `src/models/<name>/` (directories use snake_case, e.g., `revert_risk_model`).
- Deployment charts are in `operations/deployment-charts` under `helmfile.d/ml-services/<name>/`.
- Chart names sometimes differ from model directory names.
- For GPU models: confirm the chart uses `amd.com/gpu` (not `nvidia.com/gpu`).
- For CPU models: check the README for the model card — WMF-trained models document architecture, training data, and expected performance.
- Model artifacts are stored in S3 at `s3://wmf-ml-models/<model-name>/`.
- vLLM settings can often be tuned without code changes via env vars in the chart.

## Input

$ARGUMENTS — the model name as it appears in `src/models/` (e.g., `policy_violation`, `articlequality`, `revert_risk_model`).
