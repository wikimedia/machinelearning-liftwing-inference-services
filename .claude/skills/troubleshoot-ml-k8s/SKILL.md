---
name: troubleshoot-ml-k8s
description: Help pinpoint why a Wikimedia ML KServe/Knative InferenceService deployment is not working on ml-serve or ml-staging Kubernetes clusters.
disable-model-invocation: false
argument-hint: "<cluster> <namespace> <inferenceservice>"
context: ""
agent: ""
---

# Pinpoint Broken ML InferenceService Deployments

Use this skill when an InferenceService deployment is failing, stuck, unreachable, not scaling, or returning bad responses. The goal is not to collect every possible Kubernetes fact; it is to narrow the failure to the responsible layer and state the most likely cause with the evidence that supports it.

**Important constraints**

- Do not use Kubernetes MCP tools.
- Do not run cluster commands yourself. Ask the user to run the exact commands and paste the output.
- All commands must be run from `deployment.eqiad.wmnet` after entering the cluster context (sudo users: `sudo -i` + `kube-env admin <cluster>`; non-sudo users: `kube-env <your-username> <cluster>`).
- Prefer focused commands for the named InferenceService over broad `kubectl get pods -A` sweeps.
- Do not suggest to run test commands like curl if you're not aware of how to test an endpoint

## Interactive Setup

Collect information one prompt at a time. Do not dump all questions at once.

**Step 0 — Access check**

Prompt: "Do you have sudo access on `deployment.eqiad.wmnet`? If yes, you will run `sudo -i` and `kube-env admin <cluster>`. If not, you will run `kube-env <your-username> <cluster>` instead."

Wait for the user's reply before proceeding.

**Step 1 — Cluster**

Prompt: "Which cluster is the InferenceService deployed to? (`ml-serve-eqiad`, `ml-serve-codfw`, or `ml-staging-codfw`)"

Wait for the user's reply before proceeding.

**Step 2 — Namespace**

Prompt: "What namespace is it in?"

Wait for the user's reply before proceeding.

**Step 3 — InferenceService name**

Prompt: "What is the InferenceService name?"

After collecting all three inputs, have the user enter the cluster context:

```bash
ssh deployment.eqiad.wmnet
# If you have sudo access:
sudo -i
KUBE_USER=admin
kube-env "$KUBE_USER" <cluster>

# If you do not have sudo access:
KUBE_USER=<your-username>
kube-env "$KUBE_USER" <cluster>
```

Then ask them to run the initial diagnostic bundle:

```bash
NS=<namespace>
ISVC=<inferenceservice>

kubectl get inferenceservice "$ISVC" -n "$NS" -o wide
kubectl get inferenceservice "$ISVC" -n "$NS" -o yaml
kubectl describe inferenceservice "$ISVC" -n "$NS"
kubectl get events -n "$NS" --sort-by='.lastTimestamp' | tail -80
kubectl get ksvc,revision,route,configuration -n "$NS" -l serving.kserve.io/inferenceservice="$ISVC"
kubectl get pods -n "$NS" -l serving.kserve.io/inferenceservice="$ISVC" -o wide
```

Read the output in this order:

1. InferenceService `status.conditions`: identify the first `False` or `Unknown` condition and its `reason`/`message`.
2. Events: use the newest warning/event that mentions the InferenceService, revision, pod, image, volume, scheduler, or webhook.
3. Knative resources: check whether the failure is before revision creation, during revision readiness, route/ingress, or scaling.
4. Pods: check whether there is no pod, a Pending pod, a crashing pod, or a Running pod with bad readiness.

## Symptom Branches

### InferenceService Missing or Spec Rejected

Use when `kubectl get inferenceservice` returns NotFound, or `describe`/events show admission, webhook, validation, or reconciliation errors.

```bash
kubectl get inferenceservices -n "$NS"
kubectl get pods -n kserve
kubectl logs -n kserve deploy/kserve-controller-manager --tail=200
```

Likely causes:

- deployed to the wrong namespace or cluster
- invalid predictor spec, storage URI, runtime, resource request, or annotations
- KServe controller or webhook problem

Conclude with the exact rejected field or controller error if present.

### No Revision or Revision Not Ready

Use when the InferenceService exists but no ready Knative revision is created.

```bash
kubectl describe ksvc "$ISVC-predictor" -n "$NS"
kubectl get revisions -n "$NS" -l serving.kserve.io/inferenceservice="$ISVC" -o wide
kubectl describe revision -n "$NS" -l serving.kserve.io/inferenceservice="$ISVC"
kubectl logs -n knative-serving deploy/controller --tail=200
```

Likely causes:

- container image cannot be resolved or pulled
- invalid command, args, env, port, or resource shape
- Knative configuration/revision failed before pods became healthy

Name the failing revision and condition.

### Predictor Pod Pending

Use when pod status is `Pending`, `Unschedulable`, or stuck creating.

```bash
POD=<pod>

kubectl describe pod "$POD" -n "$NS"
kubectl get nodes -o wide
kubectl get pvc -n "$NS"
kubectl get nodes -o json | jq -r '.items[] | {name: .metadata.name, capacity_gpu: .status.capacity."amd.com/gpu", allocatable_gpu: .status.allocatable."amd.com/gpu"}'
```

Likely causes:

- insufficient CPU, memory, or `amd.com/gpu`
- node selector, affinity, taint, or toleration mismatch
- PVC/storage mount problem
- image pull secret or registry access issue

Use the pod events as the source of truth. Quote the scheduler or kubelet reason in the final diagnosis.

### Predictor Pod Crashing

Use when pod status is `CrashLoopBackOff`, `Error`, `RunContainerError`, or restarts keep increasing.

```bash
POD=<pod>

kubectl describe pod "$POD" -n "$NS"
kubectl logs "$POD" -n "$NS" --all-containers --tail=200
kubectl logs "$POD" -n "$NS" --all-containers --previous --tail=200
```

Likely causes:

- model server process exits on startup
- missing model files, bad model path, unsupported format, or bad `storageUri`
- missing environment variable, secret, config, or runtime dependency
- out-of-memory or GPU initialization failure

Differentiate app failure from platform failure:

- App failure: stack trace or explicit model/runtime error in predictor logs.
- Platform failure: kubelet/container events, image pull, mount, OOMKilled, permission, or device errors.

### Pod Running but Not Ready

Use when the predictor pod is Running but the InferenceService/Revision remains not ready.

```bash
POD=<pod>

kubectl describe pod "$POD" -n "$NS"
kubectl logs "$POD" -n "$NS" --all-containers --tail=200
kubectl get revision -n "$NS" -l serving.kserve.io/inferenceservice="$ISVC" -o yaml
kubectl describe ksvc "$ISVC-predictor" -n "$NS"
```

Likely causes:

- readiness probe failing
- wrong container port or model server not listening
- queue-proxy cannot reach the predictor container
- slow model load exceeding probe or progress deadline

Look for readiness probe failures, queue-proxy errors, and the actual listening port in logs.

### Endpoint Unreachable or Returning 404/503

Use when resources are ready but HTTP requests fail.

```bash
kubectl get inferenceservice "$ISVC" -n "$NS" -o jsonpath='{.status.url}{"\n"}'
kubectl get route "$ISVC-predictor" -n "$NS" -o yaml
kubectl describe route "$ISVC-predictor" -n "$NS"
kubectl get virtualservice -A | grep "$ISVC"
kubectl get pods -n istio-system
kubectl logs -n knative-serving deploy/activator --tail=200
```

Likely causes:

- request is using the wrong host/path/protocol
- Knative route or Istio VirtualService not programmed
- activator/ingress path is failing
- service scaled to zero and activation is failing

If the service is ready, make the diagnosis from route status, ingress/activator logs, and the exact HTTP status.

### Slow, Timing Out, or Not Scaling

Use when requests eventually work but latency, timeouts, or scale behavior is wrong.

```bash
kubectl get revision -n "$NS" -l serving.kserve.io/inferenceservice="$ISVC" -o yaml
kubectl describe ksvc "$ISVC-predictor" -n "$NS"
kubectl logs -n knative-serving deploy/autoscaler --tail=200
kubectl logs -n knative-serving deploy/activator --tail=200
kubectl top pods -n "$NS"
kubectl top nodes
```

Likely causes:

- model startup or first inference is too slow
- autoscaling annotations do not match workload behavior
- concurrency, target utilization, min scale, or max scale mismatch
- resource saturation on CPU, memory, GPU, or network

Separate cold-start problems from steady-state performance problems.

### Bad Prediction Response

Use when the service is reachable but returns 4xx, 5xx, malformed output, or wrong predictions.

```bash
POD=<pod>

kubectl logs "$POD" -n "$NS" --all-containers --tail=300
kubectl get inferenceservice "$ISVC" -n "$NS" -o yaml
kubectl describe pod "$POD" -n "$NS"
```

Likely causes:

- request schema does not match model server expectations
- model artifact/version mismatch
- runtime dependency or tokenizer/config mismatch
- application-level exception after request reaches the predictor

Treat this as an application/model issue unless Kubernetes events or readiness conditions show platform failure.

## Output Format

When responding to the user, be concise and diagnostic:

1. **Likely failing layer**: KServe reconciliation, Knative revision, scheduler, container runtime, model server, storage/model artifact, ingress/route, autoscaling, or request/application.
2. **Evidence**: the condition, event, pod status, log line, or HTTP behavior that points there.
3. **Most likely cause**: a concrete hypothesis, not a generic category.
4. **Next check or fix**: one or two targeted commands or changes that would confirm or resolve it.

If evidence is contradictory, say what conflicts and ask for the smallest missing command output. Avoid asking for full cluster dumps unless the current branch gives no signal.

## Cluster Reference

- `ml-serve-eqiad`: production eqiad, workers `ml-serve1001` through `ml-serve1015`
- `ml-serve-codfw`: production codfw, workers `ml-serve2001` through `ml-serve2011`
- `ml-staging-codfw`: staging codfw, workers `ml-staging2001` through `ml-staging2003`
- Kubernetes: v1.31
- CNI: Calico v3.29
- GPU resource: AMD GPUs via `amd.com/gpu`
