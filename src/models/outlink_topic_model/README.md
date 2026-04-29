# Article Topic Outlink

The articletopic-outlink inference service takes a Wikipedia article title and the language it was written in then uses links in the article to predict a set of topics that the article might be relevant to.

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Production/Language_agnostic_link-based_article_topic
* Model: https://analytics.wikimedia.org/published/wmf-ml-models/articletopic/outlink/
* Model license: CC0 License

## Inference Protocols

The service supports KServe v1 (REST) and v2 (REST and gRPC) protocols.

### V1 REST

```console
curl localhost:8080/v1/models/outlink-topic-model:predict \
  -H "Content-Type: application/json" \
  -d '{"page_id": 5355, "lang": "en"}'
```

### V2 REST

```console
curl localhost:8080/v2/models/outlink-topic-model/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "input", "shape": [1], "datatype": "BYTES", "data": ["{\"page_id\": 5355, \"lang\": \"en\"}"]}]}'
```

### V2 gRPC

Using grpcurl (download proto first):
```bash
curl -sL "https://raw.githubusercontent.com/kserve/kserve/v0.14.1/docs/predict-api/v2/grpc_predict_v2.proto" -o /tmp/grpc_predict_v2.proto

grpcurl \
  -plaintext \
  -import-path /tmp \
  -proto grpc_predict_v2.proto \
  -d '{
    "model_name": "outlink-topic-model",
    "inputs": [{
      "name": "input",
      "shape": [1],
      "datatype": "BYTES",
      "contents": {"bytes_contents": ["eyJwYWdlX2lkIjogNTM1NSwgImxhbmciOiAiZW4ifQ=="]}
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
request.model_name = "outlink-topic-model"
request.id = "test-123"

input_data = json.dumps({"page_id": 5355, "lang": "en"}).encode("utf-8")
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

## Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `page_title` | string | One of `page_title` or `page_id` | Wikipedia article title |
| `page_id` | integer | One of `page_title` or `page_id` | Wikipedia page ID |
| `lang` | string | One of `lang` or `wiki_id` | Language code (e.g., "en") |
| `wiki_id` | string | One of `lang` or `wiki_id` | Wiki database code (e.g., "enwiki"); resolved to `lang` via the bundled `data/wikis.tsv` mapping |
| `revision_id` | integer | No | Specific revision ID to analyze |
| `threshold` | float | No | Confidence threshold for results (default: 0.5) |
| `debug` | boolean | No | Enable debug mode (default: false) |

When both `lang` and `wiki_id` are provided, `lang` takes precedence. Example using `wiki_id`:

```console
curl localhost:8080/v1/models/outlink-topic-model:predict \
  -H "Content-Type: application/json" \
  -d '{"page_id": 5355, "wiki_id": "enwiki"}'
```

### Using `revision_id`

By default, the model fetches outlinks from the current state of the article using a fast single-query approach. When `revision_id` is provided, the model uses a 2-query approach to fetch outlinks from that specific revision:

1. Parse API call to get links from the specific revision
2. Batch queries to resolve Wikidata QIDs for those links

This is useful for:
- Reproducible predictions on historical revisions
- Analyzing how topic predictions change over time
- Processing events where you need predictions for a specific revision

Example with `revision_id`:
```console
curl localhost:8080/v1/models/outlink-topic-model:predict -i -X POST \
  -d '{"page_title": "Douglas_Adams", "lang": "en", "revision_id": 1264030954}'
```

**Performance Note** The revision-based approach is slower (~3-4x) than the default approach due to multiple API calls.

**Template rendering accuracy limitation** This is not purely the state of the page at that moment in time. Any templates are rendered as they currently exist, so if e.g., the navbox at the end of the article has changed substantially since the revision, those changes would be reflected in the results (as opposed to whatever state it was in when the revision was made).

## How to run locally
In order to run the articletopic-outlink model server locally, please choose one of the two options below:

<details>
<summary>1. Automated setup using the Makefile</summary>

### 1.1. Build
In the first terminal run:
```console
make articletopic-outlink
```

### 1.2. Query
In the second terminal query the isvc using:
```console
curl localhost:8080/v1/models/outlink-topic-model:predict -i -X POST -d '{"page_title": "Douglas_Adams", "lang": "en"}'
```

### 1.3. Remove
If you would like to remove the setup run:
```console
MODEL_TYPE=articletopic make clean
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
pip install -r src/models/outlink_topic_model/model_server/requirements.txt
pip install -r python/requirements.txt
```

### 2.2. Download the model
Download the `model.bin` from the link below and place it in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/articletopic/outlink/20221111111111/

### 2.3. Run the server
Start the predictor:
```console
MODEL_PATH=PATH_TO_MODEL_DIR MODEL_NAME="outlink-topic-model" python3 src/models/outlink_topic_model/model_server/model.py
```

In the second terminal make a request to the server:
```console
curl localhost:8080/v1/models/outlink-topic-model:predict -i -X POST -d '{"page_title": "Douglas_Adams", "lang": "en"}'
```
</details>

## How to test cache integration

End-to-end test of [hoarde](https://gitlab.wikimedia.org/repos/sre/hoarde) talking to
outlink-topic-model over the KServe v2 gRPC protocol, backed by Cassandra. Assumes macOS
or Linux with Docker, Go 1.24+, `protoc` and `curl`. The hoarde and inference-services
repos are assumed to be checked out side-by-side; commands are run from each repo root.

### 1. Download the model

From the root of `inference-services`:

```console
mkdir -p ./models/outlink/20221111111111
curl -L -o ./models/outlink/20221111111111/model.bin \
  https://analytics.wikimedia.org/published/wmf-ml-models/articletopic/outlink/20221111111111/model.bin
```

The file is ~990 MB.

### 2. Start outlink-topic-model

Still from the root of `inference-services`:

```console
PATH_TO_OUTLINK_TOPIC_MODEL=$(pwd)/models/outlink/20221111111111 \
  docker compose up --build outlink-topic-model
```

(The compose file mounts an absolute host path into the container, so `$(pwd)/...` is
required; the model files themselves stay under the repo.)

This exposes the model on:
- `localhost:8080` — KServe v1/v2 REST
- `localhost:8081` — KServe v2 gRPC (used by hoarde)

Wait for `Application startup complete` in the logs.

### 3. Start Cassandra

```console
docker run -d --name cassandra -p 9042:9042 cassandra:4.1
```

Wait until it accepts connections (~30 s):

```console
until docker exec cassandra cqlsh -e "DESCRIBE KEYSPACES" >/dev/null 2>&1; do sleep 2; done
```

From the root of `hoarde`, create the keyspace and the table that hoarde expects:

```console
docker exec cassandra cqlsh -e "
CREATE KEYSPACE IF NOT EXISTS hoarde
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};"

sed "s/{keyspace_name}/hoarde/g; s/{table_name}/outlink_topic_scores/g" schema.cql \
  | docker exec -i cassandra cqlsh
```

### 4. Build hoarde

From the root of `hoarde`:

```console
# Install protoc plugins (only needed once)
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
export PATH="$(go env GOPATH)/bin:$PATH"

make
```

`make` regenerates the protobuf bindings and produces a `./hoarde` binary.

### 5. Configure hoarde

Create `local-config-kserve.yaml` at the root of the hoarde repo:

```yaml
service_name: linked-artifact-cache
log_level: debug

tables:
  outlink_topic_scores:
    lambda:
      type: kserve_v2
      hostname: localhost
      port: 8081
      model_name: outlink-topic-model
      timeout: 10000ms

listen_port: 8181

cassandra:
  keyspace: hoarde
  hosts:
    - localhost:9042
  consistency: one
```

Start hoarde:

```console
./hoarde -config local-config-kserve.yaml
```

It is ready when `GET /healthz` returns 200:

```console
until curl -sf http://localhost:8181/healthz >/dev/null; do sleep 1; done
```

### 6. Exercise the cache

The hoarde URL pattern is `/v1/{table}/{wiki}/{page}/{revision}`.

Cache miss (forces a call to the lambda):

```console
curl -s -H "Cache-Control: no-cache" \
  "http://localhost:8181/v1/outlink_topic_scores/enwiki/5355/1264030954"
```

Expected output:

```json
{"topics": [["Culture.Media.Media*", 0.81...], ["Culture.Media.Music", 0.58...]],
 "lang": "en", "page_id": 5355, "page_title": null}
```

Cache hit (served from Cassandra; observe the `Last-Modified` header):

```console
curl -i "http://localhost:8181/v1/outlink_topic_scores/enwiki/5355/1264030954"
```

Latest known revision for a page:

```console
curl "http://localhost:8181/v1/outlink_topic_scores/enwiki/5355"
```

### 7. Tear down

```console
pkill -f "./hoarde -config"
docker compose down                       # outlink-topic-model
docker stop cassandra && docker rm cassandra
```
