---
name: incident-triage
description: Triage Wikimedia ML inference-services incidents at a time T (now or past). TRIGGER on "investigate / triage / debug / post-mortem / what happened" when the subject is a LiftWing alert, ML-serve namespace, error spike, latency regression, or pod crash — including generic phrasing ("investigate this issue") when context (Phabricator task, pasted alert, log snippet) points at LiftWing, even with no alert string. Always trigger on: LiftWingServiceErrorRate, KubernetesDeploymentUnavailableReplicas, KubernetesPodCrashLooping; any namespace under revscoring*, revertrisk*, articletopic-outlink, or other ML-serve; clusters ml-serve-eqiad, ml-serve-codfw, ml-staging-codfw. SKIP only when modifying code without diagnosing a runtime incident. Pulls Prometheus via Grafana's anonymous datasource proxy; correlates with Gerrit merges in `inference-services` and `deployment-charts` (Phab IDs from `Bug:` footers). Read-only.
---

# Incident triage for ML inference-services

Read-only external triage. No SSH, no kubectl. Sources:
- Prometheus via Grafana's anonymous datasource proxy ([queries.md](queries.md)).
- Gerrit MCP for `machinelearning/liftwing/inference-services` and `operations/deployment-charts`.
- Phabricator MCP — only on task IDs from CL `Bug: T...` footers (don't search Phab; tags are unreliable).

## Cluster mapping

`prometheus` alone is ambiguous (`k8s-mlserve` covers eqiad and codfw). Always pair with `site`.

| Cluster | `site` | `prometheus` |
| --- | --- | --- |
| ml-serve-eqiad | `eqiad` | `k8s-mlserve` |
| ml-serve-codfw | `codfw` | `k8s-mlserve` |
| ml-staging-codfw | `codfw` | `k8s-mlstaging` |

For health-check sweeps with no specific target, fan out to all three in parallel.

## Time window

Every investigation centres on a single timestamp **T** (`now` or past). Always use range queries (`/api/v1/query_range`, `step=60`); a live check is a range with `T = now`. Default window: `[T - 2h, T + 2h]`, clamped at `now`. Widen if the change-point isn't visible — alert `for:` durations mean the cause often starts 15–60 min earlier.

### Pin T before any query — blocking step

Before the first Prometheus call, T must be fully specified: **year, month, day, hour, minute, timezone**. If any one is missing or guessable from context but not stated, **stop and ask the user with `AskUserQuestion`** — do not assume, do not pick the "most plausible" interpretation, do not anchor on task creation time as a substitute. A wrong day or missed UTC↔CEST shift returns empty results that look like "metric missing"; the cost of one clarifying question is far lower than triaging the wrong window.

Sources of T, in order of trust:
1. Explicit user statement ("alert at 14:32 UTC on 2026-04-21").
2. Alertmanager `startsAt` or Grafana URL `from=` param — always UTC, unambiguous.
3. Bot-rendered alert text (`<jinxer-wm>` etc.) — **timezone not guaranteed**; bare "1:01 PM" is not enough.
4. Phabricator task `dateCreated` / comment timestamps — these are UTC epoch from the API, but they timestamp the *report*, not the *incident*; only use as T when the user explicitly says so.

Wikimedia alert timestamps are UTC **when sourced from Alertmanager/Grafana**. Bot-rendered chat text may be local. CET = UTC+1, CEST = UTC+2 (DST flips last Sun of Mar / Oct).

If a query returns empty data, recheck T before assuming the metric is gone — wrong-day is the most common cause.

Compute timestamps in Python from the actual T — don't eyeball, don't reuse a number from a previous session, watch the year:

```bash
python3 -c "
import datetime as dt
T = dt.datetime(YYYY, MM, DD, HH, MM, tzinfo=dt.timezone.utc)  # or dt.datetime.now(dt.timezone.utc)
print(int((T - dt.timedelta(hours=2)).timestamp()), int((T + dt.timedelta(hours=2)).timestamp()))
"
```

## Workflow

1. **Parse**: alert name, namespace, deployment, cluster(s), and **T**. If T is not fully specified (year/month/day/hour/minute/timezone), ask before querying — see "Pin T before any query".
2. **Pick metrics path** — different alerts point at different layers. Run the right set first; cross over only if it doesn't explain the signal. Fan independent queries out in parallel. Full PromQL in [queries.md](queries.md).

   | Alert | First-pass metrics |
   |---|---|
   | `LiftWingServiceErrorRate` | Istio: 5xx by `response_code` and `response_flags`, p99 latency, request rate. Pods often look healthy — failure is per-request. |
   | `KubernetesDeploymentUnavailableReplicas` | `kube_deployment_spec_replicas` vs `_available`, then per-container readiness. |
   | `KubernetesPodCrashLooping` | `restarts_total` increase, `last_terminated_reason`, `container_oom_events_total`. |
   | Health sweep | Replicas + readiness first; add Istio if 2xx rate looks normal but errors are reported. |

   `response_flags` is the fastest split between "model bug" and "infra bug": `-` = upstream (the model server) returned the error itself; `UH`/`UF`/`NR`/`DC` = mesh-side (no healthy upstream / connect failure / no route / disconnect).

   **Spec-vs-available is the first split when replicas are unavailable**: if `spec > available` and pod count = `available`, it's **scheduling** (pod never created); if pod count = `spec` but readiness is 0, it's **container readiness**. Different failure modes — don't conflate. From there, `kube_pod_container_status_ready == 0` per `(pod, container)` names which container failed.
3. **Gerrit** — `mcp__gerrit__query_changes_by_date_and_filters`, ±1–2 days (widen to 3 if empty), both repos. Filter `message_substring` on the service name. Multi-CL incidents (change → hotfix → rollback) are common — surface all.

   | Project | Likely culprit when... |
   |---|---|
   | `machinelearning/liftwing/inference-services` | container crashes, model load errors, exceptions in predict |
   | `operations/deployment-charts` | readiness fails without crashing, pods Pending, replica/limit changes, ImagePull errors |

4. **Upstream check** — if metrics show a runtime failure with no in-window CL (e.g. articlequality / revertrisk hitting MW API timeouts), don't stop at "no deploy correlation." Note the upstream dependency and recommend checking its health for the same window (MW API SLOs, mw-api-* alerts).
5. **Phabricator** — for each relevant CL, parse `Bug: T<NNNN>` from the commit footer; pull tasks via `mcp__phabricator__phabricator_get_task`.
6. **Current state** — always end by querying the same namespace at `t = now`: `kube_deployment_status_replicas_unavailable` and `kube_pod_container_status_ready == 0`. For past T, this is the "did it recover, regress, or roll forward" coda. For T = now, this **is** the headline. Fold into the report.

## Confidence

No container logs, no K8s events — we see *which* container failed and *which* CLs landed nearby, not the actual error. Calibrate, name the gap, feed the rating into the report.

- **High** — single container failed readiness while peers were healthy, and a CL touching that container's config/code/network merged in window. Name the CL.
- **Medium** — multiple plausible CLs, or none clean.
- **Low** — clear metric signal but no in-window CL → likely upstream (MW API, Swift) or transient runtime; **or** intermittent 5xx with no readiness/restart drops → needs Logstash (LDAP `logstash-access`). Distinguish the two in the report and recommend the next data source.

## Safety

Read-only external calls only: GETs to `grafana.wikimedia.org`; Gerrit MCP reads (`query_changes*`, `get_commit_message`, `get_change_details`, `get_file_diff`); Phab MCP reads (`phabricator_get_task`, `phabricator_get_task_comments`). No SSH, no kubectl, no mutating MCP calls. Propose remediation; human executes.

## Output

Four sections, in order. Hard data, no preamble, no filler. Tables as real Markdown (pipes outside a fence); keep columns short — drop a column rather than wrap.

1. **Date** — T and the UTC window queried. If T = now, write "now (YYYY-MM-DD HH:MM UTC)".
2. **Issue description** — one-line summary table, then bullet anomalies (one line each, hard facts).
3. **Root cause** — one line + confidence rating. Name CL(s): number, subject, Phab task. If multi-CL, label causal vs reactive.
4. **Potential solutions** — actionable bullets, scoped for a human SRE. Concrete first, investigation second. If already resolved, say so on one line and skip remediation.

If everything is clean: one sentence, stop.

## Example

**Date**: 2026-04-30, window 14:00–17:30 UTC (alert fired 15:12 UTC).

**Issue description**:

| Cluster | Rev | First unavail | Recovered | Failing | Restarts |
| --- | --- | --- | --- | --- | --- |
| ml-serve-codfw | 00005 | 14:42 rollout; 16:27 qp flip | 14:46 / past window | queue-proxy | 0 |

- 14:42–14:46: rollout churn, normal.
- 14:46 onward: spec wants 5 replicas, only 4 pods exist — 5th never scheduled. Drives the alert (`for: 25m`).
- 16:27: queue-proxy flipped not-ready on all 4 pods, did not recover. kserve-container and istio-proxy stayed ready.

**Root cause**: **High** for the 16:27 queue-proxy flip — CL 1280202 `kserve-inference: allow ingress on queue-proxy port 8013` (T424049) modified queue-proxy ingress; only queue-proxy went not-ready across all pods. **Medium** for the 5th-replica scheduling — no scheduling reason in metrics, needs kube events.

**Potential solutions**:
- Revert or amend CL 1280202 (likely NetworkPolicy/probe port mismatch on queue-proxy 8013).
- Pull queue-proxy logs from Logstash to confirm the probe failure.
- `kubectl describe pod` on the missing 5th replica for the `FailedScheduling` reason.
- If urgent, scale spec.replicas 5→4 to clear the alert while the queue-proxy fix lands.

**Current state**: rolled forward to rev 00007, all 4 replicas healthy. Already resolved.
