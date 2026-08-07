# Semantic highlighting

Extractive-QA highlighter for the OpenSearch `neural-search` plugin. Given a
query and a batch of passages, it returns the character-offset spans of each
passage to highlight — the span that answers the query, or nothing when the
passage does not answer it.

The model is [`timpal0l/mdeberta-v3-base-squad2`](https://huggingface.co/timpal0l/mdeberta-v3-base-squad2)
(1.11 GB), fine-tuned on SQuAD 2.0, so abstention is a first-class outcome:
an irrelevant passage yields an empty span list rather than a spurious highlight.
Being mDeBERTa-v3, it is multilingual (250k-token vocabulary), so one deployment
serves every wiki instead of English only.

Unlike the other GPU model servers this one uses `transformers` directly rather
than vLLM, because extractive QA needs the raw `start`/`end` logits of
`AutoModelForQuestionAnswering` plus the SQuAD 2.0 `[CLS]` abstention
probability — neither of which vLLM exposes. It builds on `amd-vllm022` purely
for that image's prebuilt ROCm + torch stack, as
`.pipeline/embeddings/blubber_transformers.yaml` does: the bundled vLLM is never
imported.

---

## Layout

| Module | Role |
| --- | --- |
| `model_server/model.py` | KServe shell: `preprocess`/`predict`/`postprocess` delegating to the highlighter |
| `model_server/highlighter.py` | Request body → spans → response envelope. No serving deps |
| `model_server/spans.py` | UTF-16 offset conversion and the span constraints OpenSearch enforces |
| `model_server/qa.py` | Tokenizer + model loading and batched answer-span decoding |
| `model_server/config.py` | Env-var settings (`pydantic-settings`) |
| `model_server/request_model.py` | Pydantic request schemas |

## Request and response

`POST /v1/models/semantic-highlighter:predict`

```json
{"inputs": [
  {"question": "who designed it", "context": "It was designed by Gustave Eiffel."},
  {"question": "who designed it", "context": "Bananas are a tropical fruit."}
]}
```

```json
{"highlights": [[{"start": 19, "end": 33}], []]}
```

One inner list per input, **in the same order**, with `[]` where the model
abstained. This batch form (OpenSearch 3.3+) is the only one accepted: a body
with no `inputs` — including the single-document shape 3.0–3.2 sent — answers
`{"highlights": []}` rather than a 4xx, so an older caller degrades to
unhighlighted hits instead of a failed `_search`.

That example shows the wire shape, not a live response: both contexts are well
under the default `SH_MIN_CONTEXT_TOKENS`, so a real server would skip them and
answer `{"highlights": [[], []]}`.

Two contract details that are easy to break:

- **The body must be an object with `highlights` at the top level.** ml-commons
  stores the response in `dataAsMap` and neural-search reads
  `dataAsMap.get("highlights")`. A wrapper key, or an array body, leaves every
  hit unhighlighted *silently* rather than failing. KServe's v1 REST path returns
  a custom `Model`'s `postprocess` dict verbatim, so this holds — but it is worth
  re-checking with curl after any kserve bump.
- **Offsets are UTF-16 code units**, not Python code points, because OpenSearch
  measures them in Java `String` units. They must also be in bounds, sorted by
  `start`, have unique starts and not overlap. A span that breaks any of these
  fails the *entire* `_search`, not just one hit, so every span goes through
  `spans.harden()`. Offsets were checked against Arabic, Japanese, Thai,
  Devanagari, emoji and NFKC traps; the one case that drifts is NFD-decomposed
  text, which MediaWiki's NFC-on-save keeps out of the index.

## Configuration

| Variable | Meaning | Default |
| --- | --- | --- |
| `MODEL_NAME` | Served model name — the URL path segment | `semantic-highlighter` |
| `MODEL_PATH` | Weights: a directory or a Hugging Face model id | `/mnt/models/` |
| `SH_DTYPE` | Weight precision: `float32`, `bfloat16` or `float16` | `float32` |
| `SH_MAX_ANSWER_LEN` | Max answer span length, in tokens | `30` |
| `SH_TOP_K` | Candidate starts/ends considered when decoding | `20` |
| `SH_BATCH_SIZE` | Pairs per batched forward pass | `16` |
| `SH_MIN_SCORE` | Drop answers below this confidence → no highlight | off |
| `SH_MIN_CONTEXT_TOKENS` | Passages shorter than this skip inference and return no highlight; `0` disables | `40` |

OpenSearch sends up to `max_inference_batch_size` (default 100) pairs per call
and calls serially, so `SH_BATCH_SIZE` re-chunks a call to bound peak memory
rather than to add parallelism.

### Why `SH_MIN_CONTEXT_TOKENS` defaults to 40

The answer span is ~7 tokens whatever the passage length, so the share of a
passage highlighted is roughly `7 / length`: about half at 57 characters, a
quarter at 98, a sixth at 150. Below ~40 tokens the highlight covers so much of a
one-line snippet that it narrows nothing and reads as noise, and the forward pass
is wasted.

**The cut is in tokens, not characters, so it means the same in every script.**
The same paragraph in ten languages:

| | en | fr | de | ru | ar | hi | th | ja | zh | ko |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| characters | 235 | 251 | 233 | 218 | 168 | 187 | 172 | 105 | 77 | 122 |
| tokens | 64 | 79 | 55 | 64 | 65 | 80 | 53 | 60 | 55 | 81 |

Characters vary 3.3× for identical content; tokens vary 1.5×. A 150-*character*
cut would skip the Chinese, Japanese and Korean paragraphs while highlighting the
English one that says the same thing — a silent quality regression on exactly the
wikis this checkpoint was chosen for.

Skipping saves more than passage length suggests, because
`answer_question_batch` pads every item in a chunk to the longest: a short
passage costs the same forward pass as the long one beside it. Skipped passages
answer `[]`, which is already the abstention outcome, so nothing downstream
changes. On real retrieval output 22% of passages fall below the default — see
[Load testing](#load-testing).

## Design decisions

Recorded so they are not undone by accident.

- **Logic split out of the KServe shell.** `highlighter.py` imports neither
  kserve nor torch, so the whole wire contract is unit testable without them.
  This is not incidental: no kserve release supports Python 3.13+ (0.20.0
  declares `Requires-Python <3.13`), so on a newer interpreter the shell cannot
  even be imported while the contract tests still run.
- **`preprocess`/`predict`/`postprocess` are `async`, and inference runs via
  `anyio.to_thread.run_sync` under an `asyncio.Lock`.** Other servers here define
  them sync, which makes KServe call them inline and block the event loop. Off
  the loop, a 100-pair batch cannot stall the readiness probe; under the lock,
  concurrent search requests cannot run several batches at once and exhaust GPU
  memory. `ModelServer(workers=1)` keeps one model copy in memory — scale with
  replicas, not in-process workers.
- **Answer decoding happens on the CPU on purpose.** `qa._decode_answer` reads
  `top_k * top_k` probabilities per item as Python scalars; on a GPU each read is
  a host-device sync costing more than the forward pass. The logits are moved
  across once per batch (`out.start_logits.float().cpu()`), which keeps the
  decoder identical on both devices.
- **Modules are copied path-preserved in the `production` variant**
  (`src/models/semantic_highlighting/model_server` → same path), not flattened
  into `/srv/app` the way `qwen36` and `embeddings` do. Those are single-module
  servers; this one is six modules that import each other, and path-preserving
  means the `src.models.semantic_highlighting.model_server.*` imports resolve
  identically at runtime and under `pytest test/unit`.
- **`production` copies only blubber's venv**, not the whole site-packages tree.
  The base keeps torch in the system site-packages and the venv is built with
  `use-system-site-packages`, so nothing is ever copied over the base's own
  packages — which is what keeps two versions of a package from ending up merged
  in one directory.
- **The `test` variant overrides `base:` to bookworm** and stubs torch and
  transformers via `sys.modules` (the `test/unit/qwen36` pattern), so CI lints and
  tests in seconds instead of pulling the multi-GB GPU base. Its requirements
  deliberately exclude the service `requirements.txt`.
- **Unknown request fields are ignored, not rejected**, and a body with neither
  shape returns `{"highlights": []}` rather than a 4xx. This sits in the search
  hot path: an empty result leaves hits unhighlighted, whereas an error fails the
  caller's entire `_search`.

## Local development

```bash
docker compose build semantic-highlighting && docker compose up semantic-highlighting
curl localhost:8080/v1/models/semantic-highlighter:predict -X POST \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"question":"who designed it","context":"The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889 as the centrepiece of the World Exposition."}]}'
```

Keep the passage long. A one-line context is under `SH_MIN_CONTEXT_TOKENS` and
answers `[]` without running the model, which looks like a broken service.

### Running it on a CPU-only machine

The image needs no GPU to run. `qa.load_model` selects `cuda:0` only when torch
reports a device, so with no `/dev/kfd` it loads on the CPU by itself, and the
server never imports the vLLM that would refuse to start. Two things make a
CPU run look broken when it is only slow:

- **Set `SH_BATCH_SIZE=1`.** `Highlighter.load` runs a warm-up pass of
  `SH_BATCH_SIZE` pairs padded to `qa.MAX_LENGTH` (384) tokens *before* the
  server reports ready. This base image's torch has no tuned CPU path -- one
  such pair measured ~45-50 s on 14 threads -- so the default 16 keeps the
  readiness probe waiting about ten minutes with no log line after
  `model loaded on device cpu`. At 1 the server is ready in under 90 s.
- **Expect tens of seconds per passage.** A four-passage request measured ~55 s.
  Use the CPU to check behaviour -- spans, abstention, the envelope -- and a GPU
  for anything about speed.

By default `MODEL_PATH` is the Hugging Face id, so the container downloads 1.1 GB
on every fresh start. To avoid that, download once and mount it.

### Serving the weights from disk

```bash
# 1. fetch the snapshot (pip install huggingface_hub)
hf download timpal0l/mdeberta-v3-base-squad2 \
  --exclude "pytorch_model.bin" \
  --local-dir ./models/semantic-highlighting

# 2a. plain docker
docker run --rm -p 8080:8080 \
  -e MODEL_NAME=semantic-highlighter \
  -e MODEL_PATH=/mnt/models/ \
  -v "$PWD/models/semantic-highlighting:/mnt/models/:ro" \
  semantic-highlighting:prod

# 2b. or compose, which mounts PATH_TO_SEMANTIC_HIGHLIGHTING_MODEL at /mnt/models/
PATH_TO_SEMANTIC_HIGHLIGHTING_MODEL="$PWD/models/semantic-highlighting" \
MODEL_PATH=/mnt/models/ \
  docker compose up semantic-highlighting
```

Three things that are easy to get wrong:

- **`--exclude "pytorch_model.bin"` halves the download.** The repo carries the
  same weights twice — 1.11 GB of `model.safetensors` and 1.11 GB of
  `pytorch_model.bin` — and transformers prefers the safetensors.
- **Do not also exclude `tokenizer.json`** (16 MB). This checkpoint ships no
  `spm.model`, so without it there is no fast tokenizer and `load_model` refuses
  to start. Same trap as the `model-upload` step above.
- **`--local-dir` is what makes it mountable.** It writes a flat directory, which
  `from_pretrained` accepts; the default cache layout
  (`models--org--name/snapshots/<sha>/`) is not a valid `MODEL_PATH`.

Without `huggingface_hub`, plain HTTP works — only six files are worth fetching:

```bash
BASE=https://huggingface.co/timpal0l/mdeberta-v3-base-squad2/resolve/main
DEST=./models/semantic-highlighting
mkdir -p "$DEST"
for f in config.json model.safetensors tokenizer.json \
         tokenizer_config.json special_tokens_map.json added_tokens.json; do
  curl -fL --retry 3 -o "$DEST/$f" "$BASE/$f"
done
```

`-L` is mandatory — the weights and tokenizer are LFS objects and `resolve/main`
redirects to a CDN. `-f` is what stops silent corruption: without it a 404 page
lands in `model.safetensors` and you find out at load time. Add `-C -` to resume
a partial download, or use `wget -c -P "$DEST"`, which follows redirects on its
own. Replace `main` in `BASE` with a commit sha to pin a revision.

### Running the tests

`pytest test/unit/semantic_highlighting` needs Python ≤3.12 because of kserve. If
the host Python is newer, run the CI stage in a container from the repo root:

```bash
docker run --rm -v "$PWD":/srv/app:ro -w /srv/app \
  -e PYTHONPATH=/srv/app -e RUFF_CACHE_DIR=/tmp/ruffcache \
  docker-registry.wikimedia.org/bookworm:20260510 bash -c '
    apt-get update >/dev/null 2>&1
    apt-get install -y -qq python3 python3-pip python3-venv git >/dev/null 2>&1
    python3 -m venv /venv && /venv/bin/pip install -q --upgrade pip
    /venv/bin/pip install -q ruff==0.11.10 -r requirements-test.txt \
      -r test/unit/semantic_highlighting/requirements.txt
    T="src/models/semantic_highlighting test/unit/semantic_highlighting"
    /venv/bin/ruff check $T && /venv/bin/ruff format --check $T
    /venv/bin/python -m pytest test/unit/semantic_highlighting -q -p no:cacheprovider'
```

Mount read-only (`:ro`) and pass `-p no:cacheprovider` / `RUFF_CACHE_DIR` so
neither tool tries to write a cache into the repo.

### Load testing

The load test lives with the others, in
[`test/locust/models/semantic_highlighting/`](../../../test/locust/models/semantic_highlighting/),
and replays **real retrieval output** instead of synthetic text:
`test/locust/data/pure_knn_10.json` holds 600 enwiki queries, each with the text
of the 10 best passages a kNN search returned for it, so one request carries a
whole search page — the shape OpenSearch actually sends, at the passage lengths
it actually produces. Ranks, page ids and titles are stripped: a load test
measures how fast the highlighter answers, not what it answers, and dropping
them halves the file.

Run it from `test/locust/` (see that directory's README for the shared
`HOST`/`NS`/`MODEL_NAME` variables and the `results/` comparison):

```bash
# against staging, the host in locust.conf
MODEL=semantic_highlighting locust

# or against a local compose service, 8 concurrent clients for 60s
MODEL=semantic_highlighting locust --host http://localhost:8080 \
  --headless -u 8 -r 8 -t 60s --only-summary
```

`--passages N` sets how many of the query's top 10 ride in one request, in rank
order. It is the one worth sweeping — the client half of `SH_BATCH_SIZE`:

```bash
for p in 1 3 5 10 25 50 100; do
  echo "== $p passages/request =="
  MODEL=semantic_highlighting locust --host http://localhost:8080 \
    --headless -u 8 -r 8 -t 60s --only-summary --passages "$p"
done
```

Above 10 the request is topped up with passages from other queries, so you can
reach the 100 OpenSearch may send in one call. Each batch size is recorded under
its own name (`…:predict [10 passages]`), so a single csv keeps a sweep apart.
`SH_BENCH_PASSAGES` works as an environment fallback, and `SH_BENCH_SEED` seeds
the sampling so two runs are comparable.

What to watch: per-request latency should climb roughly linearly with `N` once
the server is compute-bound, while *pairs* per second should rise and then
plateau — the plateau is where batching stops paying, and where `SH_BATCH_SIZE`
should be capped. Beyond that point the server re-chunks internally, so you are
measuring chunking, not batching.

#### What the corpus looks like

Measured over all 6000 passages with the served model's own tokenizer:

| | p10 | p50 | p90 | mean |
| --- | --- | --- | --- | --- |
| characters | 88 | 306 | 708 | 368 |
| tokens | 23 | 79 | 182 | 95 |

This matters for reading the results, because short passages skip inference
entirely:

| `SH_MIN_CONTEXT_TOKENS` | passages skipped |
| --- | --- |
| 20 | 7.3% |
| **40** (default) | **22.0%** |
| 60 | 36.0% |

So at the default roughly a fifth of each batch costs nothing, and throughput
figures already include that saving. Only 1 query in 600 skips outright, so
essentially every request still does real work. Set `SH_MIN_CONTEXT_TOKENS=0` to
measure the unoptimised path.

#### What it measures, and what it doesn't

It is a **client-side** measurement: request latency and throughput as seen over
HTTP, with no view of GPU utilisation or queueing inside the server. Pair it with
server-side metrics before drawing conclusions about saturation.

A 200 response is **not** counted as success on its own. The body must be an
object with a top-level `highlights` list of the same length as `inputs` —
otherwise the run is recorded as a failure. This is deliberate: ml-commons reads
`dataAsMap.get("highlights")`, so a wrapper key or a short list leaves every hit
unhighlighted *silently*, and a load test that called that a success would be
measuring a broken service at full speed.

The summary also prints what share of passages came back with a highlight. Watch
it next to latency — a configuration that highlights nothing is fast for the
wrong reason.

## OpenSearch side

The cluster reaches this service through an ml-commons **remote connector** whose
`url` points at `/v1/models/semantic-highlighter:predict` and whose
`request_body` is `{ "inputs": ${parameters.inputs:-null} }`. Batch highlighting
of `inner_hits` additionally needs OpenSearch 3.7+ (for the
`ext.semantic_highlighting_batch` flag) and the
[opensearch-ml-extra](https://gitlab.wikimedia.org/repos/search-platform/opensearch-ml-extra)
plugin, which provides the `semantic_highlighter_query_enricher` search-request
processor that defaults the highlighter model id at any query depth. All model-id
handling lives in one search pipeline attached as `index.search.default_pipeline`,
so queries never name a model.
