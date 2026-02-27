# Outlink Cache Adapter

A gRPC adapter that bridges [Hoarde](https://gitlab.wikimedia.org/repos/sre/hoarde) (a revision-based caching service) with the outlink-topic-model KServe inference endpoint. On a cache miss, Hoarde calls this adapter via gRPC, which translates the request into an HTTP inference call to the outlink-topic-model and returns the result for caching in Cassandra.

* Hoarde: https://gitlab.wikimedia.org/repos/sre/hoarde
* Outlink topic model: [../README.md](../README.md)

## How to run locally

The full stack requires Cassandra, the outlink-topic-model, this adapter, and Hoarde. Follow the steps below to start everything.

<details>
<summary>1. Docker Compose (adapter + model)</summary>

### 1.1. Start Cassandra
```console
docker run -d --name cassandra -p 9042:9042 cassandra:4.1
```

Wait for it to be ready (~30 seconds):
```console
docker exec cassandra cqlsh -e "SELECT now() FROM system.local"
```

### 1.2. Create the schema
```console
docker exec cassandra cqlsh -e "CREATE KEYSPACE IF NOT EXISTS hoarde WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};"
docker exec cassandra cqlsh -e "CREATE TABLE IF NOT EXISTS hoarde.outlink_topic_scores (wiki text, page bigint, revision bigint, content blob, metadata map<text,text>, PRIMARY KEY((wiki, page), revision)) WITH CLUSTERING ORDER BY (revision DESC);"
```

### 1.3. Build and start the adapter and model
From the `inference-services` root:
```console
PATH_TO_OUTLINK_TOPIC_MODEL=/path/to/models/articletopic/outlink/20221111111111 \
  docker compose up -d outlink-topic-model outlink-cache-adapter
```

Wait for the model to load:
```console
curl http://localhost:8080/v1/models/outlink-topic-model
# {"name":"outlink-topic-model","ready":true}
```

### 1.4. Start Hoarde
Create a config file for Hoarde that points to the adapter and Cassandra:
```yaml
# hoarde-local.yaml
service_name: linked-artifact-cache
log_level: debug

tables:
  outlink_topic_scores:
    lambda:
      type: grpc
      hostname: localhost
      port: 50051
      timeout: 10000ms

listen_port: 8181

cassandra:
  keyspace: hoarde
  hosts:
    - localhost:9042
  consistency: one
```

Build and run (requires Go and the Go protobuf plugins):
```console
brew install go
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

cd /path/to/hoarde
make build
./hoarde -config hoarde-local.yaml
```

Verify Hoarde is running:
```console
curl http://localhost:8181/healthz
```

### 1.5. Query
Request article topic predictions through Hoarde:
```console
curl http://localhost:8181/v1/outlink_topic_scores/enwiki/12/1306462
```

Expected response:
```json
{
  "prediction": {
    "article": "https://en.wikipedia.org/wiki?curid=12",
    "results": [
      {"topic": "Culture.Philosophy_and_religion", "score": 0.78},
      {"topic": "STEM.STEM*", "score": 0.64}
    ]
  }
}
```

To verify caching, send the same request again and check the adapter logs:
```console
docker logs inference-services-outlink-cache-adapter-1
```
You should see only one `GetArtifact` entry for that page. To force a cache miss, use `Cache-Control: no-cache`:
```console
curl -H "Cache-Control: no-cache" http://localhost:8181/v1/outlink_topic_scores/enwiki/12/1306462
```

### 1.6. Remove
```console
docker compose down outlink-topic-model outlink-cache-adapter
docker stop cassandra && docker rm cassandra
```

</details>

## Configuration

Environment variables for the adapter:

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPC_PORT` | `50051` | Port for the gRPC server |
| `KSERVE_URL` | `http://outlink-topic-model.articletopic-outlink/v1/models/outlink-topic-model:predict` | Outlink-topic-model endpoint |
| `KSERVE_HOST_HEADER` | `outlink-topic-model-predictor.articletopic-outlink.svc.cluster.local` | Host header for KServe routing (empty for local dev) |
| `KSERVE_TIMEOUT` | `5` | HTTP request timeout in seconds |

In docker compose, `KSERVE_URL` is overridden to `http://outlink-topic-model:8080/v1/models/outlink-topic-model:predict` and `KSERVE_HOST_HEADER` is set to empty.

## Regenerating proto stubs

If `proto/lambda.proto` is updated, regenerate the Python stubs:
```console
python3 -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto proto/lambda.proto
```

This requires `grpcio-tools` (`pip install grpcio-tools==1.60.0`).
