# wrk load tests

wrk is a tool that can generate a significant load for benchmarking HTTP performance. It can optionally execute a Lua script for generating HTTP requests, processing responses, and custom reporting. For more information, visit https://github.com/wg/wrk/tree/master.

This directory contains Lua scripts that are used to perform wrk load tests for our inference services.

## Usage

In order to run the load tests through wrk, one has to run the command from the deployment server:

```bash
wrk <options> -s <script.lua> <url> --latency -- <input_file>
```

Options:
```bash
-c, --connections <N>  Connections to keep open
-d, --duration    <T>  Duration of test
-t, --threads     <N>  Number of threads to use
-s, --script      <S>  Load Lua script file
-H, --header      <H>  Add header to request
    --latency          Print latency statistics
    --timeout     <T>  Socket/request timeout
```

## Examples

Revscoring models:
```bash
wrk -c 4 -t 2 --timeout 5s -s revscoring.lua https://inference.svc.eqiad.wmnet:30443/v1/models/enwiki-goodfaith:predict --header "Host: enwiki-goodfaith.revscoring-editquality-goodfaith.wikimedia.org" --latency -- enwiki.input
```

Revert-risk LA:
```bash
wrk -c 8 -t 4 --timeout 3s -s revertrisk.lua https://inference-staging.svc.codfw.wmnet:30443/v1/models/revertrisk-language-agnostic:predict --header "Host: revertrisk-language-agnostic.revertrisk.wikimedia.org" --latency -- revertrisk.input
```

ORES legacy:
```bash
wrk -c 1 -t 1 -d 10s -s ores_legacy.lua https://ores-legacy.wikimedia.org --latency -- ores_legacy.input
```

langid:
```bash
wrk -c 8 -t 1 --timeout 50s -s langid.lua https://inference.svc.codfw.wmnet:30443/v1/models/langid:predict --latency  -- langid.input
```

The scripts also generate log files named `wrk_N.log` to record the returned status codes and responses.
