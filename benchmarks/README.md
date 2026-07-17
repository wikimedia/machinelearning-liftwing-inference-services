# LiftWing LLM serving — benchmarking runbook

A repeatable procedure for benchmarking LiftWing LLM serving with **`vllm-bench`** and
recording a performance baseline. The baseline is the reference point for future
infrastructure / serving changes (MI300X tuning, vLLM upgrades, GPU repartitioning).

Tracking task: **T431851**. Metric definitions adapted from **T431554**.

---

## 1. What this measures (and what it doesn't)

`vllm-bench` is a **load generator**: it sends HTTP requests to an already-running
OpenAI-compatible endpoint and measures client-side latency and throughput. It does
**not** need the model weights or any server access — only:

- a reachable **endpoint**, and
- a **local copy of the model's tokenizer** (token counting is client-side, so the
  tokenizer must match the served model or every token metric is wrong).

The tokenizer is just the small text files (`tokenizer.json`, `tokenizer_config.json`,
vocab/merges — a few MB), **not** the multi-GB weights. Get them from the same source
as the model, e.g. `s3://wmf-ml-models/llm/Qwen3.6-27B-FP8/`, or copy from a running
pod's `/mnt/models`. If the FP8 artifact has no `tokenizer.json`, take it from the base
`Qwen3.6-27B` model — the tokenizer is identical.

---

## 2. The tool: `vllm-bench` (standalone Rust binary)

Use **[`vllm-bench`](https://github.com/vllm-project/vllm-bench)** — a standalone Rust
binary, *not* the Python `vllm bench serve`. It's a drop-in replacement with a matching
JSON output schema, and it has the two things we need for LiftWing: `--header` (custom
`Host` header) and `--insecure` (TLS). Because it's a single binary, it also sidesteps
Python entirely.

> Why not the Python `vllm bench serve`? It has no custom-header option in current
> builds, and installing `vllm` needs Python ≥3.10 (on 3.9 it fails at import with a
> `X | None` `TypeError`). `vllm-bench` avoids both problems.

**Install** (prebuilt binary; keep the WMF web proxy exported for the download):

```bash
export http_proxy=http://webproxy.eqiad.wmnet:8080
export https_proxy=http://webproxy.eqiad.wmnet:8080
# fetch the prebuilt binary per the repo README, then make sure it's on PATH:
chmod +x ./vllm-bench
export PATH="$PWD:$PATH"          # or move it into ~/.local/bin
vllm-bench --help | head          # confirm it runs. NOTE: options hang directly off
                                  # `vllm-bench` — there is no `serve` subcommand.
```

Ran from the **ML Lab host** (Python 3.11), which can reach the internal endpoint.

---

## 3. Endpoint: internal gateway (with a Host header)

- **Internal (used for baselines).** Not rate-limited; the internal Knative gateway.
- **Public (`api.wikimedia.org`).** Capped at **100 requests/hour** for
  anonymous/non-WMCS clients (LiftWingLLM policy) — a load test hits that wall in
  seconds. Benchmark the public path separately later, from a WMCS/known-client host.

The internal Knative gateway routes **by `Host` header**, and the OpenAI server is
mounted under **`/openai`**. So:

- `--base-url https://inference.svc.eqiad.wmnet:30443/openai`
- `--endpoint /v1/completions`
- `--header Host=llm-qwen36-27b.llm.wikimedia.org`  ← how the gateway routes to the model

Confirm reachability first (system `curl` trusts the WMF CA; note the `Host` header):

```bash
curl https://inference.svc.eqiad.wmnet:30443/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Host: llm-qwen36-27b.llm.wikimedia.org" \
  -d '{"model":"llm-qwen36-27b","messages":[{"role":"user","content":"hi"}]}'
```

*Alternative:* `kubectl -n llm port-forward <predictor-pod> 8080:8080`, then
`--base-url http://localhost:8080/openai` with no `Host` header. Simpler wiring, but the
tunnel gets flaky under high concurrency and bypasses the gateway/mesh — prefer the
gateway path above for representative numbers.

---

## 4. Two LiftWing-specific flags you will need

These are the non-obvious bits that a plain `vllm-bench` invocation trips over here:

- **`--insecure`** — `vllm-bench` validates TLS against its own bundled CA set, not the
  host's system trust store, so it can't verify the internal gateway cert
  (`invalid peer certificate: UnknownIssuer`). Skipping verification is fine for this
  known internal host. (Not needed when port-forwarding over plain http.)
- **`--prompt-token-ids`** + **`--backend openai --endpoint /v1/completions`** —
  LiftWing does **not** expose the vLLM `/tokenize` endpoint (returns 404). The `random`
  dataset otherwise verifies prompt lengths against `/tokenize` and hard-fails. Sending
  pre-tokenized IDs (built from the local tokenizer) skips that server call. Token-ID
  input requires a completions backend — the chat backend can't take token IDs.

---

## 5. Fixed configuration (keep these constant)

| Parameter | Value | Why |
|---|---|---|
| Model | `llm-qwen36-27b` | End-state config: FP8 weights + FP8 KV cache, TP=1, 32K ctx |
| Backend / endpoint | `openai` / `/v1/completions` | Required for `--prompt-token-ids` |
| Dataset | `random` | Synthetic, controlled, reproducible — no data file to prepare |
| `--prompt-token-ids` | on | Skips the unavailable server-side `/tokenize` verification |
| Input / output length | 1024 / 128 tokens | Fixed prompt/response shape |
| `--ignore-eos` | on | Forces the full output length (random prompts otherwise stop early) |
| Requests per run | 200 | Enough samples for stable percentiles |
| Seed | 42 | Reproducibility |
| Request rate | `inf` | Send as fast as `--max-concurrency` allows |
| Concurrency sweep | 1, 8, 32, 64 | From single-request latency to saturation |
| Percentiles | 50, 90, 99 | Report p50/p90/p99 per metric |
| Tokenizer | local dir + `--tokenizer-mode hf` | Client-side token counts |

If you change any of these, treat it as a **new** configuration and note it alongside
the results — don't compare across different configs.

---

## 6. Run it

The wrapper script `benchmark_llm.sh` bakes in the full recipe above (gateway URL, Host
header, `--insecure`, `--prompt-token-ids`, `openai`/`/v1/completions`, local tokenizer).
Set your tokenizer path and go:

```bash
TOKENIZER=/home/<you>/benchmarks/Qwen3.6-27B-FP8 ./benchmark_llm.sh
```

The script:
1. does a **warmup run (discarded)** — pays cold start / autoscale-from-zero / first-request compile;
2. runs the **concurrency sweep**, saving one JSON per run to `./results/<timestamp>_<model>/`;
3. records the **UTC start/end window** (`window.txt`);
4. prints a summary table and writes `summary.md` (paste-ready for Phabricator).

Single point, run manually (the working invocation):

```bash
vllm-bench \
  --backend openai \
  --base-url https://inference.svc.eqiad.wmnet:30443/openai --endpoint /v1/completions \
  --header Host=llm-qwen36-27b.llm.wikimedia.org \
  --insecure \
  --model llm-qwen36-27b \
  --tokenizer /path/to/Qwen3.6-27B-FP8 --tokenizer-mode hf \
  --prompt-token-ids \
  --dataset-name random --random-input-len 1024 --random-output-len 128 --ignore-eos \
  --num-prompts 200 --max-concurrency 32 --request-rate inf --seed 42 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --result-dir ./results --result-filename c32.json
```

> Loading the live single replica affects real traffic for the duration of the run —
> give the team a heads-up / pick a quiet window, and note the UTC window for Grafana.

---

## 7. Correlate with the dashboards

Take the UTC window printed at the end and view the **ml-infra** dashboard over that
range: <https://grafana.wikimedia.org/d/dpzzsnh/ml-infra>. Cross-check GPU utilization,
VRAM, and vLLM queue/running metrics against the load level.

---

## 8. Store & compare baselines

- Keep the whole `results/<timestamp>_<model>/` directory (raw JSON + `summary.md` + `window.txt`).
- Commit baselines to the repo (or attach to the task) so future runs can be diffed
  against them. Because the config is fixed, a later run is directly comparable — re-run
  the same script after an infra change (e.g. T431553) and compare the summary tables.
- `vllm-bench --compare a.json b.json` prints a side-by-side diff of two result files.

---

## 9. Comparing across GPU-partition changes

The MI300X cards can be split into partitions, and ml-serve nodes run different schemes
(e.g. the full 192 GB card, 24 GB × 8 partitions on some nodes, and a 96 GB × 2 scheme
under consideration). Moving a model between schemes changes performance, so **rerun this
benchmark whenever the deployment's partition size or replica layout changes** — that's
the main reason the config above is fixed.

**Why it changes (a partition is not just "less VRAM"):** each partition gets a *fraction
of the card's compute (XCDs), memory bandwidth, and VRAM*. That hits three things:
- prefill / **TTFT** (compute-bound) — slower on a smaller partition;
- decode / **TPOT, output tok/s** (memory-bandwidth-bound) — slower on a smaller partition;
- **peak throughput / saturation** — less VRAM means less KV-cache headroom, so fewer
  concurrent sequences before saturation. (And the model must *fit* at all: a 27B FP8
  model's ~27 GB of weights won't fit a 24 GB partition, before any KV cache.)

Smaller partitions → lower per-replica performance but higher density (more replicas per
card). The trade is capacity/density vs. per-request latency; the benchmark quantifies it.

### The unit that matters: per-partition-replica

Models run as N replicas, each on one partition, scheduled across a shared pool — replicas
may span cards, and a model may use more than one partition. So the useful metric is the
**per-partition-replica** curve (one replica on one partition), *not* a "per physical GPU"
figure (which assumes a fixed one-model-per-card layout we don't operate). Capacity
planning is then: total ≈ N × per-partition-replica at acceptable latency; autoscaling
just changes N.

### Procedure for a partition / layout change

1. Deploy on the target partition size, **pin 1 replica** (`minReplicas=maxReplicas=1`),
   rerun the sweep with the same fixed config, and `vllm-bench --compare old.json new.json`.
2. Record the **partition scheme** (e.g. `1×192 GB`, `2×96 GB`, `8×24 GB`) as the
   independent variable, and note the vLLM-reported `max_num_seqs` / KV-cache size — if a
   smaller partition forced it down (or the model no longer fits), that explains the deltas.
3. Watch the **amd-smi VRAM-reporting quirk** under some partition modes (the lead
   partition can report the full card's VRAM — see T429597) so dashboard VRAM isn't misread.

### One-time co-location ("noisy neighbor") check — a scheduling-policy input

Contention is inherent to partition-sharing: two busy partition-replicas on the *same*
physical card (same or different model) contend for that card's shared memory
bandwidth/fabric — independent of whether anything is pinned. Measure it **once** to inform
placement policy, not as a routine baseline:

- Run two saturated partition-replicas on the **same** card vs. on **different** cards, and
  compare to 2× a single replica.
- Small penalty (<~10%) → partitions can be packed freely. Large penalty (~30–40%) → the
  scheduler should **spread replicas across cards** (anti-affinity).

Elasticity (autoscaling scale-up latency, SLO under rising load) is a **separate** test —
use ramping/sustained load (the Locust stretch goal), not this fixed-config sweep.

---

## 10. How to read the metrics (from T431554)

Per-request timeline: **request sent → [TTFT] → first token → [ITL, ITL, …] → last token**.

- **Successful / Failed requests** — completed vs errored. Failed requests are excluded
  from latency/throughput math. `fail` in the table = `num_prompts − completed`.
- **Request throughput (req/s)** = successful requests / benchmark duration.
- **Output token throughput (tok/s)** = generated tokens / duration (decode speed).
- **Total token throughput (tok/s)** = (input + output tokens) / duration (prefill + decode).
- **TTFT (time to first token)** — request sent → first token arrives. Responsiveness /
  prefill cost. Watch **p99** for tail behavior under load.
- **TPOT (time per output token)** = (E2EL − TTFT) / (output tokens − 1) — per-request
  average decode gap. `Mean TPOT == Mean ITL` always; they differ only in percentiles.
- **ITL (inter-token latency)** — gap between each consecutive pair of streamed tokens;
  percentiles are over the pooled list of all gaps. Finer-grained view of decode smoothness.
- **E2EL (end-to-end latency)** — request sent → last token received. The single
  "how long did the user wait for the full answer?" number. Includes queueing + network.
  `E2EL ≈ TTFT + (output_tokens − 1) × TPOT`.

Reading the sweep: as concurrency rises, **total throughput** climbs until the single
replica saturates, while **TTFT/TPOT/E2EL** degrade — the knee is your practical
capacity per replica.

---

## 11. Deferred / follow-ups

- Public-endpoint run from WMCS/known-client (full user-facing path).
- Additional models: `llm-qwen3-14b` (and its future quantized build), internal models.
- More input/output shapes and a realistic dataset (`--dataset-name sharegpt`).
- Automated result storage + regression comparison.
- **(Stretch)** Locust-based end-to-end load tests.
