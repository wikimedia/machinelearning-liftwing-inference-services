# Revertrisk Wikidata

The revertrisk-wikidata inference service uses the metadata and content of a Wikidata article revision ID to predict the risk of this revision being reverted.

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Production/RevertRisk_Wikidata
* Source: https://github.com/trokhymovych/wikidata-vandalism-detection
* Model: https://drive.google.com/drive/folders/1czw8zFRfkZKyxRFcwFB5PWgBIf_8QeK7?usp=sharing and https://analytics.wikimedia.org/published/wmf-ml-models/revertrisk/wikidata/20251104121312/
* Model license: Apache 2.0 License


## Inference Protocols

The service supports KServe v1 (REST) and v2 (REST and gRPC) protocols. The v2
protocol is what the [Linked Artifacts Cache](https://wikitech.wikimedia.org/wiki/Prep_Pantry/Architectures#Linked_Artifacts_Cache)
uses to call this service as a lambda (see T433699).

All three transports accept the same payloads and return the same prediction body.
Under v2 that body is JSON-encoded in `outputs[0]`. Three input shapes are accepted:

| Payload | Sent by |
|---------|---------|
| `{"rev_id": <int>}` | direct callers |
| `{"event": { ...mediawiki.page_change... }}` | change-prop / the prediction stream |
| `{"wiki_id": "wikidatawiki", "page_id": <int>, "revision_id": <int>}` | the Linked Artifacts Cache (hoarde) |

The third shape is hoarde's fixed `(wiki, page, revision)` artifact key. It has no
per-table field mapping, so the service translates it: `revision_id` becomes
`rev_id`, `page_id` is ignored (the page is derived from the revision via the MW
API), and a `wiki_id` other than `wikidatawiki` is rejected with a 400.

### V1 REST

```console
curl localhost:8080/v1/models/revertrisk-wikidata:predict \
  -H "Content-Type: application/json" \
  -d '{"rev_id": 1892513445}'
```

### V2 REST

```console
curl localhost:8080/v2/models/revertrisk-wikidata/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "input", "shape": [1], "datatype": "BYTES", "data": ["{\"rev_id\": 1892513445}"]}]}'
```

### V2 gRPC

Using grpcurl (download proto first):
```bash
curl -sL "https://raw.githubusercontent.com/kserve/kserve/v0.16.0/docs/predict-api/v2/grpc_predict_v2.proto" -o /tmp/grpc_predict_v2.proto

grpcurl \
  -plaintext \
  -import-path /tmp \
  -proto grpc_predict_v2.proto \
  -d '{
    "model_name": "revertrisk-wikidata",
    "inputs": [{
      "name": "input",
      "shape": [1],
      "datatype": "BYTES",
      "contents": {"bytes_contents": ["eyJyZXZfaWQiOiAxODkyNTEzNDQ1fQ=="]}
    }]
  }' \
  localhost:8081 \
  inference.GRPCInferenceService/ModelInfer
```

Using Python:
```python
import grpc
import json
from kserve.protocol.grpc import grpc_predict_v2_pb2, grpc_predict_v2_pb2_grpc

channel = grpc.insecure_channel('localhost:8081')
request = grpc_predict_v2_pb2.ModelInferRequest()
request.model_name = "revertrisk-wikidata"
request.id = "test-123"

input_data = json.dumps({"rev_id": 1892513445}).encode("utf-8")
tensor = request.inputs.add()
tensor.name = "input"
tensor.shape.extend([1])
tensor.datatype = "BYTES"
tensor.contents.bytes_contents.append(input_data)

stub = grpc_predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
response = stub.ModelInfer(request)
result = json.loads(response.outputs[0].contents.bytes_contents[0].decode("utf-8"))
print(json.dumps(result, indent=2))
```

Note: input validation errors are currently raised as HTTP 400, which gRPC
surfaces as `UNKNOWN` rather than `INVALID_ARGUMENT`. Switching to
`kserve.errors.InvalidInput` would change the REST error body key from `detail`
to `error`, so it is deferred until the current REST clients move to LAC.

## How to run locally

In order to run the revertrisk-wikidata model-server locally, please choose one of the two options below:

<details>
<summary>1. Automated setup using the Makefile</summary>

### 1.1. Build
In the first terminal run:
```console
make revertrisk-wikidata
```
This build process will set up: a Python venv, install dependencies, download the model, and run the server.

### 1.2. Query
On the second terminal query the isvc using:
```console
curl -s localhost:8080/v1/models/revertrisk-wikidata:predict -X POST -d '{"rev_id": 1892513445}' -i -H "Content-type: application/json"
```

### 1.3. Remove
If you would like to remove the setup run:
```console
MODEL_TYPE=revertrisk-wikidata make clean
```
</details>
<details>
<summary>2. Manual setup</summary>

### 2.1. Build Python venv and install dependencies
First add the top level directory of the repo to the PYTHONPATH:
```console
export PYTHONPATH=$PYTHONPATH:.
```

Create a virtual environment and install the dependencies using:
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/models/revertrisk_wikidata/requirements.txt torch pandas
```

`torch` and `pandas` are imported by the model server but are not listed in the requirements
file — in production they come from the ROCm base image, so they have to be installed
explicitly when running outside the container.

Use a dedicated virtualenv rather than an existing one: the model pickle contains a
`transformers.Pipeline`, so the pinned `transformers==4.25.1` matters and unpickling under a
different version can fail or change results.

### 2.2. Download data file(s)
Download the model from the link below and place it in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/revertrisk/wikidata/20251104121312/

Now our PATH_TO_MODEL_DIR directory contains the model with the following structure:
```console
PATH_TO_MODEL_DIR
└── wikidata_revertrisk_graph2text_v2.pkl
```

### 2.3. Run the server
We can run the server locally with:
```console
MODEL_NAME=revertrisk-wikidata MODEL_PATH=PATH_TO_MODEL_DIR/wikidata_revertrisk_graph2text_v2.pkl python3 src/models/revertrisk_wikidata/model_server/model.py
```

On a separate terminal we can make a request to the server with:
```console
curl -s localhost:8080/v1/models/revertrisk-wikidata:predict -X POST -d '{"rev_id": 1892513445}' -i -H "Content-type: application/json"
```
</details>

## How to test cache integration

End-to-end test of [hoarde](https://gitlab.wikimedia.org/repos/sre/hoarde) (the Linked
Artifacts Cache) talking to revertrisk-wikidata over the KServe v2 gRPC protocol, backed by
Cassandra. This is the architecture proposed in
[T433699](https://phabricator.wikimedia.org/T433699).

Requires Docker, Go, `curl`, plus `protoc` and `grpcurl`:

```console
# macOS
brew install protobuf grpcurl

# Debian/Ubuntu
sudo apt install protobuf-compiler
# grpcurl: see https://github.com/fullstorydev/grpcurl/releases
```

This walkthrough uses two repos, cloned side by side. Each step says which one to run from —
`hoarde` for steps 3-6, `inference-services` for the rest:

```
~/some/dir/
├── hoarde/               # cloned in step 3
└── inference-services/   # this repo
```

### 1. Download the model

From the root of `inference-services`:

```console
mkdir -p ./models/revertrisk/wikidata/20251104121312
curl -L --retry 5 --retry-all-errors \
  -o ./models/revertrisk/wikidata/20251104121312/wikidata_revertrisk_graph2text_v2.pkl \
  https://analytics.wikimedia.org/published/wmf-ml-models/revertrisk/wikidata/20251104121312/wikidata_revertrisk_graph2text_v2.pkl
```

The file is ~662 MB (exactly 662,319,231 bytes — worth checking, a truncated download fails
later with a confusing unpickling error).

### 2. Start revertrisk-wikidata

The service image is built on an **amd64-only ROCm base**, so `docker compose` on an ARM Mac
runs it under emulation (the base layer alone is 3.48 GB). The model itself is *not*
GPU-bound — it falls back to CPU — so on a Mac prefer running it natively.

First create the virtualenv (as in [2.1](#21-build-python-venv-and-install-dependencies) above);
skip this if you already have it:

```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/models/revertrisk_wikidata/requirements.txt torch pandas
```

Do not reuse an unrelated virtualenv: a missing dependency shows up as
`ModuleNotFoundError: No module named 'catboost'`, and a `transformers` other than the pinned
4.25.1 risks failing to unpickle the model's `transformers.Pipeline`. Then start the server:

```console
MODEL_NAME=revertrisk-wikidata \
MODEL_PATH=$(pwd)/models/revertrisk/wikidata/20251104121312/wikidata_revertrisk_graph2text_v2.pkl \
CUSTOM_UA="revertrisk-wikidata-local/1.0 (https://phabricator.wikimedia.org/T433699; you@wikimedia.org)" \
PYTHONPATH=.:src/models/revertrisk_wikidata/model_server \
  python3 src/models/revertrisk_wikidata/model_server/model.py
```

`CUSTOM_UA` is required outside LiftWing. Wikimedia's edge rejects requests whose User-Agent
carries no contact details with `HTTP 403 … "Please set a user-agent and respect our robot
policy"` (see [T400119](https://phabricator.wikimedia.org/T400119)); LiftWing's own network is
allowlisted, your laptop is not. Put a real contact address in it.

On amd64 hardware you can use the container instead:

```console
PATH_TO_REVERTRISK_WIKIDATA_MODEL=$(pwd)/models/revertrisk/wikidata/20251104121312 \
  docker compose up --build revertrisk-wikidata
```

Either way this exposes:
- `localhost:8080` — KServe v1/v2 REST
- `localhost:8081` — KServe v2 gRPC (used by hoarde)

Wait for `Application startup complete` in the logs. The server runs in the foreground, so
**leave it in this terminal and open a new one for the remaining steps**, then confirm:

```console
curl -s localhost:8080/v1/models/revertrisk-wikidata
# {"name":"revertrisk-wikidata","ready":true}
```

### 3. Clone hoarde

hoarde is a separate repo, maintained by SRE. Clone it next to `inference-services` — the next
step needs its `schema.cql`:

```console
cd /path/to/wiki_repos       # the directory that contains inference-services
git clone https://gitlab.wikimedia.org/repos/sre/hoarde.git
cd hoarde
```

Steps 4-6 all run from this directory; it is what "from the root of `hoarde`" means below.

### 4. Start Cassandra

```console
docker run -d --name cassandra -p 9042:9042 cassandra:4.1
```

This pulls the upstream Cassandra image from Docker Hub — there is nothing to build.

Wait until it accepts connections. The loop below is silent while it waits, so on a first run it
can sit for several minutes pulling the ~521 MB image; once the image is local, Cassandra is
ready in ~30 s:

```console
until docker exec cassandra cqlsh -e "DESCRIBE KEYSPACES" >/dev/null 2>&1; do sleep 2; done
```

From the root of `hoarde`, create the keyspace and the table that hoarde expects:

```console
docker exec cassandra cqlsh -e "
CREATE KEYSPACE IF NOT EXISTS hoarde
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};"

sed "s/{keyspace_name}/hoarde/g; s/{table_name}/revertrisk_wikidata_scores/g" schema.cql \
  | docker exec -i cassandra cqlsh
```

Use hoarde's own `schema.cql` — the table needs a `last_modified` column that older copies of
this DDL omit.

### 5. Build hoarde

From the root of `hoarde`:

```console
# Install protoc plugins (only needed once)
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
export PATH="$(go env GOPATH)/bin:$PATH"

make

## Output ##
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VERSION ......: v1.5.0
# BUILD HOST ...: wmf3620
# BUILD DATE ...: 2026-09-09T13:37:38Z
# GO VERSION ...: go1.24.9
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# go build  -ldflags "-X main.version=v1.5.0 -X main.buildDate=2026-09-09T13:37:38Z -X main.buildHost=wmf3620" -o hoarde .
# go build -o examples/cmd/server/server examples/cmd/server/server.go
# go build -o examples/cmd/client/client examples/cmd/client/client.go
# go build -o cmd/bench/bench            cmd/bench/main.go
# go build -o cmd/inference/client       cmd/inference/client.go
```

`make` regenerates the protobuf bindings and produces a `./hoarde` binary. hoarde needs Go
1.24+ and one plugin needs 1.25+; with the default `GOTOOLCHAIN=auto` Go downloads what it
needs, so an older local Go is fine.

Expect `make` to leave the hoarde checkout dirty: the `build` target runs `protoc`, which
rewrites the four generated `proto/*.pb.go` files. Unless your `protoc` matches the version
last used upstream, `git status` will show them as modified — the diff is only the generated
`// protoc-gen-go vX.Y.Z` header comments, with no functional change. Discard it with
`git checkout -- proto/` if you want a clean tree. The next step also adds an untracked
`local-config-kserve.yaml`.

### 6. Configure hoarde

Create `local-config-kserve.yaml` at the root of the hoarde repo:

```yaml
service_name: linked-artifact-cache
log_level: debug

tables:
  revertrisk_wikidata_scores:
    lambda:
      type: kserve_v2
      hostname: localhost
      port: 8081
      model_name: revertrisk-wikidata
      timeout: 120000ms
      content_type: application/json

listen_port: 8181

cassandra:
  keyspace: hoarde
  hosts:
    - localhost:9042
  consistency: one
```

The generous timeout matters: the model loads lazily on the first request, and a cold
prediction on CPU takes a few seconds.

Start hoarde:

```console
./hoarde -config local-config-kserve.yaml
```

You should see something like this:

```console
## Output ##
{"@timestamp":"2026-09-09T16:39:53+03:00","message":"Initializing linked-artifact-cache (Version: v1.5.0, Go: go1.24.9, Build host: wmf3620, Timestamp: 2026-09-09T13:37:38Z)","log":{"level":"INFO"},"service":{"name":"linked-artifact-cache"},"ecs":{"version":"1.11.0"}}
{"@timestamp":"2026-09-09T16:39:53+03:00","message":"Listening on localhost:8181","log":{"level":"INFO"},"service":{"name":"linked-artifact-cache"},"ecs":{"version":"1.11.0"}}
{"@timestamp":"2026-09-09T16:39:53+03:00","message":"Cassandra: Opened new connection to 127.0.0.1","log":{"level":"DEBUG"},"service":{"name":"linked-artifact-cache"},"ecs":{"version":"1.11.0"}}
{"@timestamp":"2026-09-09T16:39:53+03:00","message":"Cassandra: Opened new connection to ::1","log":{"level":"DEBUG"},"service":{"name":"linked-artifact-cache"},"ecs":{"version":"1.11.0"}}
{"@timestamp":"2026-09-09T16:39:53+03:00","message":"Cassandra: Opened new connection to ::1","log":{"level":"DEBUG"},"service":{"name":"linked-artifact-cache"},"ecs":{"version":"1.11.0"}}
{"@timestamp":"2026-09-09T16:39:53+03:00","message":"Cassandra: Opened new connection to ::1","log":{"level":"DEBUG"},"service":{"name":"linked-artifact-cache"},"ecs":{"version":"1.11.0"}}

```

hoarde runs in the foreground — leave it in this terminal and use another for the next step.

Readiness is signalled by the **HTTP status code**: 200 once hoarde is serving, 503 while it is
still initialising. The `/healthz` body is hoarde's build info (version, build date, Go version)
and is byte-identical in both cases, so check the status, not the payload:

```console
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8181/healthz
# 200
```

### 7. Exercise the cache

These are all `localhost` requests, so they can be run from any directory (hoarde is still
running in its own terminal). The hoarde URL pattern is
`/revisions/v1/{table}/{wiki}/{page}/{revision}`.

Cache miss (forces a call to the lambda):

```console
curl -s -H "Cache-Control: no-cache" \
  "http://localhost:8181/revisions/v1/revertrisk_wikidata_scores/wikidatawiki/108601362/1892513445"
```

Expected output:

```json
{"model_name": "revertrisk-wikidata", "model_version": "2", "revision_id": 1892513445,
 "output": {"prediction": false, "probabilities": {"true": 0.10437076243299818, "false": 0.8956292375670019}}}
```

Cache hit (served from Cassandra; observe the `Last-Modified` header and note that the model
server logs no second request):

```console
curl -i "http://localhost:8181/revisions/v1/revertrisk_wikidata_scores/wikidatawiki/108601362/1892513445"
```

Expected output:

```json
{"model_name": "revertrisk-wikidata", "model_version": "2", "revision_id": 1892513445, "output": {"prediction": false, "probabilities": {"true": 0.10437076243299818, "false": 0.8956292375670019}}}%
```

Latest known revision for a page (returns an `X-Hoarde-Revision-Id` header):

```console
curl -i "http://localhost:8181/revisions/v1/revertrisk_wikidata_scores/wikidatawiki/108601362"
```

Expected output:

```json
{"model_name": "revertrisk-wikidata", "model_version": "2", "revision_id": 1892513445, "output": {"prediction": false, "probabilities": {"true": 0.10437076243299818, "false": 0.8956292375670019}}}%
```

Confirm the artifact was persisted:

```console
docker exec cassandra cqlsh -e \
  "SELECT wiki,page,revision,last_modified FROM hoarde.revertrisk_wikidata_scores;"
```

Expected output:

```console
 wiki         | page      | revision   | last_modified
--------------+-----------+------------+---------------------------------
 wikidatawiki | 108601362 | 1892513445 | 2026-09-09 13:47:10.790000+0000

(1 rows)
```

hoarde keys artifacts on a fixed `(wiki, page, revision)` triple and sends it to the lambda as
`{"wiki_id": ..., "page_id": ..., "revision_id": ...}`. The model server accepts that shape
directly (see [Inference Protocols](#inference-protocols)); `page_id` is ignored because the
page is derived from the revision via the MW API.

### 8. Tear down

```console
pkill -f "hoarde -config"
pkill -f model_server/model.py          # or: docker compose down
docker stop cassandra && docker rm cassandra
```

The model caches MW API responses under `/tmp/revertrisk_wikidata_disk_cache`; remove it if you
want the next run to start cold.
