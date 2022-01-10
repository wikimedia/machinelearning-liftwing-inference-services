# Outlinks Topic Model KServer

Custom KServe model running the Outlinks (fastText) topic model

## Deploy a custom image InferenceService using the command line

### Setup

1. Your ~/.kube/config should point to a cluster with [KServe installed](https://github.com/kserve/kserve#installation).
2. Your cluster's Istio Ingress gateway must be [network accessible](https://istio.io/latest/docs/tasks/traffic-management/ingress/ingress-control/).

### Build and push the Docker Image

```
# Build the container on your local machine
docker build -t {username}/outlinks-topic-model ./model-server

# Push the container to docker registry
docker push {username}/outlinks-topic-model
```

### Create the InferenceService

Apply the CRD

```
kubectl apply -f service.yaml
```

Expected Output

```
$ inferenceservice.serving.kubeflow.org/outlink-topic-model created
```

### Run a prediction
The first step is to [determine the ingress IP and ports](../../../../README.md#determine-the-ingress-ip-and-ports) and set `INGRESS_HOST` and `INGRESS_PORT`

```
MODEL_NAME=outlink-topic-model
INPUT_PATH=@./input.json
SERVICE_HOSTNAME=$(kubectl get inferenceservice ${MODEL_NAME} -o jsonpath='{.status.url}' | cut -d "/" -f 3)

curl -v -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict -d $INPUT_PATH
```

Expected Output:

```
*   Trying 10.100.34.202...
* Connected to 10.100.34.202 (10.100.34.202) port 80 (#0)
> POST /v1/models/outlink-topic-model:predict HTTP/1.1
> Host: outlink-topic-model.kubeflow-user.example.com
> User-Agent: curl/7.47.0
> Accept: */*
> Content-Length: 18
> Content-Type: application/x-www-form-urlencoded
>
* upload completely sent off: 18 out of 18 bytes
< HTTP/1.1 200 OK
< content-length: 112
< content-type: application/json; charset=UTF-8
< date: Fri, 07 May 2021 20:09:10 GMT
< server: istio-envoy
< x-envoy-upstream-service-time: 192
<
* Connection #0 to host 10.100.34.202 left intact
{"prediction": {"article": "https://en.wikipedia.org/wiki/Toni Morrison", "results": [{"topic": "Culture.Biography.Biography*", "score": 0.9626831412315369}, {"topic": "Culture.Literature", "score": 0.6654205918312073}, {"topic": "Geography.Regions.Americas.North_America", "score": 0.607673168182373}]}}
```

### Delete the InferenceService

```
kubectl delete -f service.yaml
```

Expected Output

```
$ inferenceservice.serving.kubeflow.org "outlink-topic-model" deleted
```

