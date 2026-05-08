# Prometheus reference

WMF Grafana (`grafana.wikimedia.org`) accepts anonymous **GET** to the datasource proxy. POST is blocked, which is why the official `grafana/mcp-grafana` MCP doesn't work today. Skip the MCP and call the proxy directly.

A Viewer-scope service account would unblock the MCP — service-account creation requires org-admin rights we don't have, so SRE Observability would need to provision one. Until then, `curl -G`.

## Curl pattern

Default to range queries (`/api/v1/query_range`); a live "is it broken right now" check is just a range with `end = now`.

```bash
curl -sG "https://grafana.wikimedia.org/api/datasources/proxy/uid/<ds>/api/v1/query_range" \
     --data-urlencode 'query=<PromQL>' \
     --data-urlencode 'start=<unix-seconds>' \
     --data-urlencode 'end=<unix-seconds>' \
     --data-urlencode 'step=60'
```

The instant endpoint `/api/v1/query` (no `start`/`end`/`step`) is available for cheap one-shot "current value" lookups, but isn't required — a 5-minute range with `end = now` does the same job and gives you a slope.

Both return JSON. Pipe through `python3 -m json.tool` for readability or extract specific fields with a small inline Python.

For range results, compress the output by emitting only **transitions** — lines where the value changed from the previous step. A 90-minute "queue-proxy not-ready" stretch should appear as two timestamps, not 90.

## Datasource UIDs

- `000000026` — **Thanos** (default; federates everything; filter by the `prometheus` label). Use this for cross-site queries and historical data (longer retention).
- `D-2kXvZnk` — codfw `k8s-mlserve` direct
- `PEB5F43D0C34E78B3` — codfw `k8s-mlstaging` direct
- `aWotKxQMz` — eqiad `k8s-mlserve` direct
- `000000017` / `000000018` — eqiad / codfw `k8s` (general clusters, not ML)

## Label vocabulary

- `site`: `eqiad`, `codfw`, `drmrs`, `eqsin`, `esams`, `magru`, `ulsfo`.
- `prometheus`: `k8s`, `k8s-mlserve`, `k8s-mlstaging`, `k8s-staging`, `k8s-aux`, `k8s-dse`, `services`, `analytics`, `cloud`, `ext`, `ops`.
- On `kube_*` (kube-state-metrics) series: `namespace`, `deployment`, `pod`, `container`. The `kubernetes_namespace` / `kubernetes_pod_name` labels point at the kube-state-metrics pod itself — ignore those.
- Container names in inference-services pods: `kserve-container` (the model), `queue-proxy` (Knative sidecar), `istio-proxy` (mesh sidecar).

## Canned PromQL

Substitute `<ns>`, `<deploy>`, and the right `prometheus` / `site` labels.

### Replica & rollout state

```promql
# desired vs available — the scheduling-vs-readiness split.
# spec > available + pod count = available  ->  scheduling failure (pod never created)
# spec = pod count, but available < spec     ->  readiness failure (pod exists, container not ready)
kube_deployment_spec_replicas{prometheus="k8s-mlserve",site="codfw",namespace="<ns>",deployment="<deploy>"}
kube_deployment_status_replicas_available{prometheus="k8s-mlserve",site="codfw",namespace="<ns>",deployment="<deploy>"}

# unavailable replicas (the alert metric — sums both failure modes)
kube_deployment_status_replicas_unavailable{prometheus="k8s-mlserve",site="codfw",namespace="<ns>",deployment="<deploy>"}

# pod phase (Pending/Running/Failed/Succeeded/Unknown)
kube_pod_status_phase{prometheus="k8s-mlserve",namespace="<ns>",pod=~"<deploy>.*"} == 1

# unschedulable pods (5th replica never got a node, etc.)
kube_pod_status_unschedulable{prometheus="k8s-mlserve",namespace="<ns>"} == 1
```

### Container health (the smoking-gun layer)

```promql
# per-container readiness — 0 means not ready. The single most useful metric for diagnosis:
# tells us *which* container failed (kserve / queue-proxy / istio-proxy) and for how long.
kube_pod_container_status_ready{prometheus="k8s-mlserve",namespace="<ns>",pod=~"<deploy>.*"} == 0

# containers stuck waiting and why (ImagePullBackOff, CrashLoopBackOff,
# CreateContainerConfigError, ContainerCreating)
kube_pod_container_status_waiting_reason{prometheus="k8s-mlserve",namespace="<ns>"} == 1

# last termination reason (OOMKilled, Error, Completed)
kube_pod_container_status_last_terminated_reason{prometheus="k8s-mlserve",namespace="<ns>"} == 1

# restarts in last hour, per pod/container
sum by (pod,container) (
  increase(kube_pod_container_status_restarts_total{prometheus="k8s-mlserve",namespace="<ns>"}[1h])
)

# image tags currently running (catches version skew across clusters)
kube_pod_container_info{prometheus="k8s-mlserve",namespace="<ns>",pod=~"<deploy>.*"}
```

### Resources

```promql
# OOM kill events in last hour
increase(container_oom_events_total{namespace="<ns>"}[1h])

# memory usage as fraction of limit
container_memory_working_set_bytes{namespace="<ns>"}
  / on(pod,container)
  kube_pod_container_resource_limits{namespace="<ns>",resource="memory"}

# CPU throttling — high values mean the container hits its CPU limit
rate(container_cpu_cfs_throttled_seconds_total{namespace="<ns>"}[5m])
```

### Traffic & errors (Istio)

`response_flags` is the fastest split between model-bug and infra-bug. `-` = upstream returned the error itself; `UH` no healthy upstream, `UF` connect failure, `NR` no route, `DC` downstream disconnect — the rest mean the mesh tripped the request.

```promql
# 5xx rate to the service, by code
sum by (response_code) (
  rate(istio_requests_total{destination_service_namespace="<ns>",response_code=~"5.."}[5m])
)

# 5xx split by Istio response_flags (which side failed)
sum by (response_code, response_flags) (
  rate(istio_requests_total{destination_service_namespace="<ns>",response_code=~"5.."}[5m])
)

# request latency p99
histogram_quantile(0.99,
  sum by (le, destination_service_name) (
    rate(istio_request_duration_milliseconds_bucket{destination_service_namespace="<ns>"}[5m])
  )
)
```

Add to this list as new failure modes show up. Do **not** pre-engineer queries for hypotheticals.
