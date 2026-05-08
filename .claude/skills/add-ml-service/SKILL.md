---
name: add-ml-service
description: Scaffold a new LiftWing ML service in operations/deployment-charts. Handles both adding to an existing namespace (append inference_services entry) and creating a brand-new namespace (full helmfile scaffold). Use when an engineer wants to deploy a new inference service for the first time.
argument-hint: "<service-name>"
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash(cp *)
  - Bash(mkdir *)
  - Bash(ls *)
  - Bash(cat *)
---

# Add a New ML Service — deployment-charts

Add a new LiftWing inference service to `operations/deployment-charts/helmfile.d/ml-services/`.

There are two paths depending on whether the service belongs to an existing namespace or
needs a new one:

- **Existing namespace** (simpler): append a new `inference_services` entry to the existing
  `values.yaml` and `values-ml-staging-codfw.yaml`.
- **New namespace** (full scaffold): create `helmfile.yaml`, `values.yaml`, and
  `values-ml-staging-codfw.yaml` from scratch.

## Input

`$ARGUMENTS` — the new inference service name in kebab-case (e.g. `edit-check`,
`revertrisk-multilingual`). This is the service name, not necessarily the namespace name.

---

## Steps

### 1. Locate the deployment-charts repo

Check the `DEPLOYMENT_CHARTS_PATH` environment variable. If it is set and points to a valid
directory, use it as the repo root.

If it is not set, ask the user:

> "What is the path to your local `operations/deployment-charts` checkout?"

Then tell them they can avoid this prompt in future by adding the path to
`.claude/settings.local.json` (gitignored, so safe for local paths):

```json
{
  "env": {
    "DEPLOYMENT_CHARTS_PATH": "/path/to/operations/deployment-charts"
  }
}
```

---

### 2. Ask: existing namespace or new namespace?

List the existing namespaces by reading `helmfile.d/ml-services/` (exclude `_example_` and
`CLAUDE.md`). Show the list to the user, then ask:

> "Should `<service-name>` be added to an existing namespace, or does it need a new one?"
>
> Existing namespaces: article-descriptions, article-models, revertrisk, edit-check, ...

If **existing namespace**: ask which namespace to use, then jump to **Path A**.

If **new namespace**: ask for the namespace name (may differ from the service name — e.g.
service `revertrisk-multilingual` lives in namespace `revertrisk`), then follow **Path B**.

---

## Path A — Existing Namespace (append only)

### A1. Inspect the namespace structure

Read **all** values files in the chosen namespace directory (use `ls` to list them first).
Namespaces vary in how they organise `inference_services` entries:

- **Simple namespaces** (e.g. `edit-check`, `revertrisk`): entries live in `values.yaml`,
  with a separate `values-ml-staging-codfw.yaml` for staging overrides.
- **Env-split namespaces** (e.g. `experimental`): `values.yaml` holds only shared config
  (docker, networkpolicy, monitoring); entries live in env-specific files such as
  `values-ml-serve-eqiad.yaml` and/or `values-ml-staging-codfw.yaml`.

Read the existing files before collecting config so you know which file(s) to append to,
and can mirror the indentation, field order, and style of existing entries.

Ask the user which environment file(s) the new service should be added to if it is not
obvious from the namespace structure.

### A2. Collect service config

Collect from the user:

| Field | Default / hint |
|---|---|
| Docker image name | `machinelearning-liftwing-inference-services-<service-name>` |
| `MODEL_NAME` env var | same as `<service-name>` |
| `STORAGE_URI` | `s3://wmf-ml-models/<service-name>/` (user must supply full path) |
| Extra `custom_env` entries | none |
| CPU request & limit | ask |
| Memory request & limit | ask |
| GPU needed? | 0 or 1 (same for prod and staging) |
| Prod `minReplicas` / `maxReplicas` | ask |
| Enable request batching? | if yes: ask `maxBatchSize` and `maxLatency` |
| Environments | both prod DCs (default) or eqiad-only? |

### A3. Append to the target file(s)

Append a new entry under `inference_services:` in each target file identified in A1.

**How to append safely:** Use `Edit` to replace the last unique block at the end of the
file with that block plus the new entry. If `Edit` reports multiple matches (ambiguous
context — e.g. two services share identical node-affinity blocks), fall back to Bash
append:

```bash
cat >> <file> << 'EOF'
  <new-entry-yaml>
EOF
```

Be careful not to accidentally extend an existing entry's YAML structure — always verify
the result with `Read` after appending.

The new entry to append:

```yaml
  <service-name>:
    predictor:
      config:
        minReplicas: <prod-minReplicas>
        maxReplicas: <prod-maxReplicas>
        <batcher block if enabled:>
        batcher:
          maxBatchSize: <maxBatchSize>
          maxLatency: <maxLatency>
      image: "<docker-image-name>"
      image_version: "TODO-image-version"
      custom_env:
        - name: MODEL_NAME
          value: "<MODEL_NAME>"
        - name: STORAGE_URI
          value: "<STORAGE_URI>"
        <any extra custom_env entries>
      container:
        resources:
          limits:
            cpu: "<cpu-limit>"
            memory: <memory-limit>
            amd.com/gpu: "<gpu>"
          requests:
            cpu: "<cpu-request>"
            memory: <memory-request>
            amd.com/gpu: "<gpu>"
```

### A4. Skip to Step 4 (summary, commit message, next steps)

---

## Path B — New Namespace (full scaffold)

### B1. Collect namespace and service config

Collect from the user:

| Field | Default / hint |
|---|---|
| Namespace name | ask (e.g. `revertrisk` for a new revertrisk variant) |
| Docker image name | `machinelearning-liftwing-inference-services-<service-name>` |
| `MODEL_NAME` env var | same as `<service-name>` |
| `STORAGE_URI` | `s3://wmf-ml-models/<service-name>/` (user must supply full path) |
| Extra `custom_env` entries | none |
| CPU request & limit | ask |
| Memory request & limit | ask |
| GPU needed? | 0 or 1 (same for prod and staging) |
| Prod `minReplicas` / `maxReplicas` | ask |
| Enable request batching? | if yes: ask `maxBatchSize` and `maxLatency` |
| Environments | both prod DCs (default) or eqiad-only? |
| Network egress endpoints | checklist — see below |

**Egress checklist** — ask which of these the service needs to reach:

| Label | codfw CIDR | eqiad CIDR | Port |
|---|---|---|---|
| `thanos-swift` | 10.2.1.54/32 | 10.2.2.54/32 | 443/tcp |
| `api-ro` | 10.2.1.22/32 | 10.2.2.22/32 | 443/tcp |
| `mw-api-int-ro` | 10.2.1.81/32 | 10.2.2.81/32 | 4446/tcp |
| `eventgate-main` | 10.2.1.45/32 | 10.2.2.45/32 | 4492/tcp |
| `rest-gateway` | 10.2.1.82/32 | 10.2.2.82/32 | 4113/tcp |

Collect all inputs before writing any files.

### B2. Create the namespace directory

```
mkdir -p <deployment-charts-root>/helmfile.d/ml-services/<namespace>/
```

### B3. Create helmfile.yaml

Read `helmfile.d/ml-services/_example_/helmfile.yaml` as the template. Replace every
occurrence of `SERVICE_NAME` with `<namespace>`.

If eqiad-only was selected, remove the `ml-serve-codfw` block from the `environments:`
section.

Write to `helmfile.d/ml-services/<namespace>/helmfile.yaml`.

### B4. Create values.yaml

Write `helmfile.d/ml-services/<namespace>/values.yaml`:

```yaml
docker:
  registry: docker-registry.discovery.wmnet/wikimedia
  imagePullPolicy: IfNotPresent

networkpolicy:
  egress:
    enabled: true
    # These endpoints should be reachable by Istio proxy sidecars.
    dst_nets:
      <only the CIDRs the user selected — both codfw and eqiad entries per endpoint,
       formatted as in the example below>
      # - cidr: 10.2.1.54/32 # thanos-swift.svc.codfw.wmnet
      #   ports:
      #   - port: 443
      #     protocol: tcp

monitoring:
  enabled: true

inference:
  annotations:
    sidecar.istio.io/inject: "true"
    autoscaling.knative.dev/metric: "rps"
    autoscaling.knative.dev/target: "15"
    prometheus.kserve.io/scrape: "true"
    prometheus.kserve.io/port: "8080"
    prometheus.kserve.io/path: "/metrics"

inference_services:
  <service-name>:
    predictor:
      config:
        minReplicas: <prod-minReplicas>
        maxReplicas: <prod-maxReplicas>
        <batcher block if enabled>
      image: "<docker-image-name>"
      image_version: "TODO-image-version"
      custom_env:
        - name: MODEL_NAME
          value: "<MODEL_NAME>"
        - name: STORAGE_URI
          value: "<STORAGE_URI>"
        <any extra custom_env entries>
      container:
        resources:
          limits:
            cpu: "<cpu-limit>"
            memory: <memory-limit>
            amd.com/gpu: "<gpu>"
          requests:
            cpu: "<cpu-request>"
            memory: <memory-request>
            amd.com/gpu: "<gpu>"
```

### B5. Create values-ml-staging-codfw.yaml

Write `helmfile.d/ml-services/<namespace>/values-ml-staging-codfw.yaml` — same
`inference_services` block as values.yaml, with staging differences:
- `minReplicas: 1`
- `maxReplicas: 1`
- CPU, memory, GPU: same as prod
- Image version: `TODO-image-version`

### B6. Create values-ml-serve-eqiad.yaml (eqiad-only only)

If eqiad-only was selected, create
`helmfile.d/ml-services/<namespace>/values-ml-serve-eqiad.yaml` with:

```yaml
# eqiad-only service — no codfw deployment
```

---

## Step 4 — Summary of changes

List the files created or modified with their full paths. Remind the user that
`image_version: "TODO-image-version"` must be replaced before deploying — use the
`/bump-image` skill once Jenkins publishes the image after the inference-services patch merges.

---

## Step 5 — Draft the commit message

Display for the user to copy — do **not** run `git commit`.

**New namespace:**
```
ml-services: add <namespace> namespace for <service-name>

Why:

- New LiftWing inference service scaffolded in deployment-charts

What:

- Add helmfile.yaml, values.yaml, values-ml-staging-codfw.yaml
  for <namespace>/<service-name>

Assisted-by: Claude Sonnet 4.6
```

**Existing namespace (append):**
```
ml-services: add <service-name> to <namespace>

Why:

- New LiftWing inference service added to existing namespace

What:

- Append <service-name> inference_services entry to
  <namespace>/<file(s) actually modified>

Assisted-by: Claude Sonnet 4.6
```

---

## Step 6 — Next steps

```
Next steps:
  1. Replace TODO-image-version once Jenkins publishes the image
     (or run /bump-image after the inference-services patch merges).

  2. cd <deployment-charts-root>/helmfile.d/ml-services/<namespace>/
     git add .
     git commit    # paste the message above
     git review

  3. After merge — on deployment.eqiad.wmnet:
     cd /srv/deployment-charts/helmfile.d/ml-services/<namespace>

     # Staging first:
     helmfile -e ml-staging-codfw diff
     helmfile -e ml-staging-codfw sync

     # Then production:
     helmfile -e ml-serve-eqiad diff
     helmfile -e ml-serve-eqiad sync
     helmfile -e ml-serve-codfw diff   # skip if eqiad-only
     helmfile -e ml-serve-codfw sync   # skip if eqiad-only
```
