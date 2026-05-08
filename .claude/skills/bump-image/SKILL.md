---
description: Bump the Docker image tag for one or more LiftWing inference services in operations/deployment-charts after a new image is published by Jenkins. Use when a patch has merged in inference-services, PipelineBot has posted a new image tag on the Gerrit CL, and you need to update the corresponding values.yaml file(s) to deploy it.
disable-model-invocation: false
name: bump-image
argument-hint: "<inference-services-cl-number> [isvc-name ...]"
allowed-tools:
  - Read
  - Edit
  - Glob
  - Grep
  - Bash(git diff*)
  - Bash(git log*)
  - Bash(git status*)
  # TODO: replace WebFetch with mcp__gerrit__list_change_comments once the gerrit-mcp-server
  # is updated to return full message bodies (currently truncates PipelineBot messages)
  - WebFetch
  - mcp__gerrit__get_commit_message
  - mcp__gerrit__list_change_files
  - mcp__phabricator__phabricator_get_task
---

# Bump Image Tag in deployment-charts

Update the Docker image tag for one or more LiftWing inference services (isvcs) in
`operations/deployment-charts` after Jenkins publishes a new image.

> **Scope note:** This skill is intentionally limited to bumping image tags. If it ever
> expands to handle other `values.yaml` changes (env vars, replica counts, etc.), it
> should be renamed to reflect that broader scope.

## Input

`$ARGUMENTS` — `<cl-number> [isvc-name ...]`

- `cl-number`: the inference-services Gerrit CL number where the patch merged and
  PipelineBot posted the published image (e.g. `1279278`). **Required.**
- `isvc-name`: zero or more isvc names to pre-filter candidates
  (e.g. `revertrisk-language-agnostic revertrisk-multilingual`). **Optional.**
  When omitted, all isvcs whose image matches the published image are shown for selection.

> **Note:** isvc names and image names do not map 1:1. For example, both
> `revertrisk-language-agnostic` and `revertrisk-language-agnostic-pre-save` share the
> image `machinelearning-liftwing-inference-services-revertrisk`. Always confirm the
> exact set of isvcs to update in the interactive step below.

## Steps

### 1 + 1b + 2. Parallel data fetch (do all three in one batch)

**Make the following three calls in parallel** — they are independent and should never be
sequential:

**1. PipelineBot images** — WebFetch the Gerrit messages endpoint:
```
URL:    https://gerrit.wikimedia.org/r/changes/<cl-number>/messages
Prompt: Extract only Docker image references published by PipelineBot matching
        machinelearning-liftwing-inference-services-*:*-publish.
        Return as a compact list, one entry per line: <image-name>:<tag>.
        Omit all other message content, timestamps, and metadata.
```
> **Note:** `mcp__gerrit__list_change_comments` cannot be used here — it does not return
> full message bodies. Replace with that tool once the gerrit-mcp-server exposes
> `/changes/<id>/messages`.

**1b. Changed files for auto-pre-filter** — call `mcp__gerrit__list_change_files` on the
same CL. Look for paths under `src/models/<model-name>/`. The model directory name is a
strong signal for which image was the primary target. Cross-reference against the image
list from step 1 — images whose name contains the model directory name are **pre-selected**.
If `isvc-name` args were provided in `$ARGUMENTS`, use those instead and skip inference.

**2. Phabricator task** — call `mcp__gerrit__get_commit_message` on the same CL and
extract the `Bug: TXXXXX` trailer. Reuse it in the deployment-charts commit message.

If no PipelineBot messages are found, ask the user to paste the image reference(s) directly.

### 3. Locate the deployment-charts repo

Check the `DEPLOYMENT_CHARTS_PATH` environment variable. If it is set and points to a
valid directory, use it as the repo root.

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

### 4. Find candidate isvcs

Use **Grep** (not Read) to search `helmfile.d/ml-services/*/values.yaml` and
`helmfile.d/ml-services/*/values-ml-staging-codfw.yaml` for lines containing the image
name from step 1. Grep returns only matching lines — do not read entire files at this
stage. Also grep for the surrounding context lines (use `-C 3` or similar) to capture
the isvc name and current tag alongside the image reference.

If `isvc-name` args were provided, further filter candidates to only those matching the
given names.

Show the user a preview table. Mark pre-selected entries with `*`:

```
  Namespace / file                                  isvc name                       Current tag                New tag
* outlink/values.yaml                               outlink-topic-model             2024-04-01-100000-publish  2024-05-07-143022-publish
  revertrisk/values.yaml                            revertrisk-language-agnostic    2024-04-01-100000-publish  2024-05-07-143022-publish
  ...
```

Ask the user to confirm the selection (pre-selected entries are the default) or adjust it,
then choose scope: **both prod + staging**, **staging only**, or **prod only**.

### 5. Edit the values files

For each confirmed isvc + file combination, use `Edit` to replace the old tag string with
the new tag. Only the tag string is replaced — no other YAML keys are touched.

### 6. Draft the commit message

Do **not** run `git commit`. Instead, draft the commit message and display it for the user
to review, modify if needed, and commit themselves.

Commit message format (Wikimedia conventions):

- Subject: `ml-services: update <primary-isvc-name> image tag to <new-tag>`
  - Use the first/most significant updated isvc name as the primary one.
  - If multiple isvcs were updated, list the rest in the `What:` body.
- Body:

```
ml-services: update <primary-isvc-name> image tag to <new-tag>

Why:

- New Docker image published after patch merge in inference-services

What:

- Bump <primary-isvc-name> version to <new-tag>
- Bump <other-isvc-name> version to <new-tag>  (if applicable)

Assisted-by: Claude Sonnet 4.6
Bug: TXXXXX
```

### 7. Remind about next steps

```
Next steps:
  git add <modified files>
  git commit            # paste the drafted message above
  git review            # submit for review

  # After merge — on deployment.eqiad.wmnet, cd to the namespace dir:
  cd /srv/deployment-charts/helmfile.d/ml-services/<namespace>

  # Staging:
  helmfile -e ml-staging-codfw diff
  helmfile -e ml-staging-codfw sync

  # Production (both DCs):
  helmfile -e ml-serve-eqiad diff
  helmfile -e ml-serve-eqiad sync
  helmfile -e ml-serve-codfw diff
  helmfile -e ml-serve-codfw sync
```
