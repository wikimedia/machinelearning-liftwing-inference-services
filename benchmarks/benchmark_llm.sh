#!/usr/bin/env bash
# =============================================================================
# LiftWing LLM serving benchmark — vllm bench serve wrapper
#
# Runs a FIXED, documented configuration across a concurrency sweep, saves one
# JSON per run, and prints (and writes) a combined summary table.
#
# See README-benchmarking.md for the full procedure and the metric glossary.
# Keep the configuration below fixed so results stay comparable across
# infrastructure / vLLM / config changes (T431851).
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — EDIT THE TWO "REPLACE" VALUES, then keep the rest fixed.
# All are overridable via environment variables, e.g. NUM_PROMPTS=500 ./benchmark_llm.sh
# ---------------------------------------------------------------------------
# Internal endpoint. base-url ENDS IN /openai; the endpoint is appended to it.
# Two ways to reach it:
#  (A) Knative gateway (mirrors real internal clients): connect to the gateway and
#      route with a Host header. vllm-bench (rustls) does not trust the WMF CA, so
#      INSECURE=1 is needed here (see below).
#        BASE_URL=https://inference.svc.eqiad.wmnet:30443/openai
#        HOST_HEADER=llm-qwen36-27b.llm.wikimedia.org
#  (B) Port-forward the predictor pod (no Host header, plain http, INSECURE=0):
#        kubectl -n llm port-forward <predictor-pod> 8080:8080
#        BASE_URL=http://localhost:8080/openai   HOST_HEADER=   INSECURE=0
BASE_URL="${BASE_URL:-https://inference.svc.eqiad.wmnet:30443/openai}"
ENDPOINT="${ENDPOINT:-/v1/completions}"
HOST_HEADER="${HOST_HEADER:-llm-qwen36-27b.llm.wikimedia.org}"  # empty when port-forwarding

# Benchmark tool. Two options with a compatible JSON output schema:
#   vllm-bench         -> standalone Rust binary; supports --header KEY=VALUE and --insecure (recommended: lets you
#                         hit the gateway directly with a Host header). NOTE: no "serve" subcommand.
#   vllm bench serve   -> the Python CLI (no custom-header support in some builds)
BENCH_CMD="${BENCH_CMD:-vllm-bench}"
HEADER_FMT="${HEADER_FMT:-key=value}"   # vllm-bench: key=value ; python vllm bench serve: "key: value"
# vllm-bench (rustls) does NOT trust the system CA store, so it rejects the gateway's
# WMF-CA cert -> keep 1 for the internal gateway. Set 0 only if the tool trusts the CA
# (e.g. when port-forwarding over plain http, INSECURE is irrelevant).
INSECURE="${INSECURE:-1}"
# LiftWing does NOT expose the server /tokenize endpoint, and the random dataset otherwise
# verifies prompt lengths against it. Sending pre-tokenized IDs (from the local tokenizer)
# skips that verification. Requires a completions backend (openai + /v1/completions).
PROMPT_TOKEN_IDS="${PROMPT_TOKEN_IDS:-1}"   # 1 -> add --prompt-token-ids
TOKENIZER_MODE="${TOKENIZER_MODE:-hf}"      # local HF tokenizer

# openai + /v1/completions pairs with --prompt-token-ids (chat backend can't take token IDs).
BACKEND="${BACKEND:-openai}"
MODEL="${MODEL:-llm-qwen36-27b}"          # the "model" field sent in each request (served-model-name)
TOKENIZER="${TOKENIZER:-/REPLACE/path/to/Qwen3.6-27B-FP8}"  # LOCAL tokenizer dir (small files, NOT weights)

INPUT_LEN="${INPUT_LEN:-1024}"            # random-dataset prompt length (tokens)
OUTPUT_LEN="${OUTPUT_LEN:-128}"           # forced output length (with --ignore-eos)
NUM_PROMPTS="${NUM_PROMPTS:-200}"         # total requests per run
SEED="${SEED:-42}"                        # fixed for reproducibility
REQUEST_RATE="${REQUEST_RATE:-inf}"       # inf = send as fast as --max-concurrency allows
CONCURRENCIES="${CONCURRENCIES:-1 8 32 64}"   # the sweep
PERCENTILES="${PERCENTILES:-50,90,99}"    # percentiles to report per metric

RESULT_DIR="${RESULT_DIR:-./results/$(TZ=UTC date +%Y%m%dT%H%M%SZ)_${MODEL}}"
# ---------------------------------------------------------------------------

mkdir -p "$RESULT_DIR"

run_bench () {
  local concurrency="$1" filename="$2"
  local cmd=($BENCH_CMD)          # word-split "vllm-bench serve" / "vllm bench serve" into argv
  local extra=()
  # Optional Host header (gateway routing), in the format the chosen tool expects.
  if [ -n "$HOST_HEADER" ]; then
    if [ "$HEADER_FMT" = "key=value" ]; then
      extra+=(--header "Host=$HOST_HEADER")
    else
      extra+=(--header "Host: $HOST_HEADER")
    fi
  fi
  [ "$INSECURE" = "1" ] && extra+=(--insecure)
  [ "$PROMPT_TOKEN_IDS" = "1" ] && extra+=(--prompt-token-ids)
  "${cmd[@]}" \
    --backend "$BACKEND" \
    --base-url "$BASE_URL" \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --tokenizer-mode "$TOKENIZER_MODE" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --ignore-eos \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$concurrency" \
    --request-rate "$REQUEST_RATE" \
    --seed "$SEED" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles "$PERCENTILES" \
    ${extra[@]+"${extra[@]}"} \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "$filename"
}

echo "==> Results dir : $RESULT_DIR"
echo "==> Model       : $MODEL   in/out=${INPUT_LEN}/${OUTPUT_LEN}  num_prompts=$NUM_PROMPTS  seed=$SEED  rate=$REQUEST_RATE"
echo "==> Endpoint    : $BASE_URL$ENDPOINT"

# 1) Warmup (discarded): pays cold-start / autoscale-from-zero / first-request compile.
echo "==> Warmup run (discarded)…"
run_bench 8 "warmup.json" || true
rm -f "$RESULT_DIR/warmup.json"

# 2) Measured sweep. Record the UTC window so it can be lined up with Grafana.
START_UTC="$(TZ=UTC date +%FT%TZ)"
echo "==> Sweep start (UTC): $START_UTC"
for c in $CONCURRENCIES; do
  echo "==> Concurrency $c …"
  run_bench "$c" "c${c}.json"
done
END_UTC="$(TZ=UTC date +%FT%TZ)"
echo "==> Sweep end   (UTC): $END_UTC"
printf 'start_utc=%s\nend_utc=%s\n' "$START_UTC" "$END_UTC" > "$RESULT_DIR/window.txt"

# 3) Build a combined summary table (stdout + summary.md for pasting into Phabricator).
python3 - "$RESULT_DIR" <<'PY'
import json, glob, os, sys

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "c*.json")),
               key=lambda p: int(os.path.basename(p)[1:-5]))

HDR = ["conc", "req/s", "out tok/s", "tot tok/s",
       "TTFT p50", "TTFT p99", "TPOT p50", "TPOT p99",
       "E2EL p50", "E2EL p99", "fail"]

def fmt(v):
    if v is None: return "-"
    if isinstance(v, float): return f"{v:,.2f}"
    return f"{v:,}"

rows = []
for f in files:
    with open(f) as fh:
        r = json.load(fh)
    g = r.get
    completed = g("completed") or 0
    total = g("num_prompts") or completed
    rows.append([
        g("max_concurrency"),
        g("request_throughput"),
        g("output_throughput"),
        g("total_token_throughput"),
        g("p50_ttft_ms") or g("median_ttft_ms"),
        g("p99_ttft_ms"),
        g("p50_tpot_ms") or g("median_tpot_ms"),
        g("p99_tpot_ms"),
        g("p50_e2el_ms") or g("median_e2el_ms"),
        g("p99_e2el_ms"),
        total - completed,
    ])

# Console table
print("\n" + " | ".join(f"{h:>10}" for h in HDR))
print("-" * (len(HDR) * 13))
for row in rows:
    print(" | ".join(f"{fmt(c):>10}" for c in row))

# Markdown table (Phabricator / runbook)
md = os.path.join(d, "summary.md")
with open(md, "w") as out:
    out.write("| " + " | ".join(HDR) + " |\n")
    out.write("|" + "|".join(["---"] * len(HDR)) + "|\n")
    for row in rows:
        out.write("| " + " | ".join(fmt(c) for c in row) + " |\n")
print(f"\nMarkdown summary written to {md}")
PY

echo
echo "==> Done."
echo "    Grafana window (UTC): $START_UTC -> $END_UTC"
echo "    ml-infra dashboard:  https://grafana.wikimedia.org/d/dpzzsnh/ml-infra"
