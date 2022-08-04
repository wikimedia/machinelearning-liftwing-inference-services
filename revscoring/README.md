# Revscoring InferenceServices that use a KServe Model Server

This is how we serve [revscoring](https://github.com/wikimedia/revscoring) ( [articlequality](https://github.com/wikimedia/articlequality), [editquality](https://github.com/wikimedia/editquality), [draftquality](https://github.com/wikimedia/draftquality), and topic ( [drafttopic , articletopic](https://github.com/wikimedia/drafttopic) ) ) models using [KServe](https://github.com/kserve/kserve).

## Deploy a revscoring InferenceService using the command line

### Setup

1. Your ~/.kube/config should point to a cluster with [KServe installed](https://github.com/kserve/kserve#installation).
2. Your cluster's Istio Ingress gateway must be [network accessible](https://istio.io/latest/docs/tasks/traffic-management/ingress/ingress-control/).

### Build and push the Docker Image

To build the Docker image, [Blubber](https://wikitech.wikimedia.org/wiki/Blubber) is needed:

```
blubber .pipeline/editquality/blubber.yaml production | docker build --tag SOME-DOCKER-TAG-THAT-YOU-LIKE --file - .
```

See also the Wikitech [Kserve](https://wikitech.wikimedia.org/wiki/Machine_Learning/LiftWing/KServe) documentation for more info about local testing with Docker.

### Create the InferenceService

Apply the CRD

```
$ kubectl apply -f ./editquality/enwiki-goodfaith/service.yaml
```

Expected Output

```
$ inferenceservice.serving.kubeflow.org/enwiki-goodfaith created
```

### Check InferenceServices that exist to confirm yours was created

```
$ kubectl get inferenceservice
```

Expected Output

```
NAME               URL                                                 READY   DEFAULT TRAFFIC   CANARY TRAFFIC   AGE
enwiki-goodfaith   http://enwiki-goodfaith.kubeflow-user.example.com   True    100                                5d
```

### Run a prediction
The first step is to [determine the ingress IP and ports](https://kserve.github.io/website/get_started/first_isvc/#3-determine-the-ingress-ip-and-ports) and set `INGRESS_HOST` and `INGRESS_PORT`

```
MODEL_NAME=enwiki-goodfaith
INPUT_PATH=@./editquality/enwiki-goodfaith/input.json
SERVICE_HOSTNAME=$(kubectl get inferenceservice ${MODEL_NAME} -o jsonpath='{.status.url}' | cut -d "/" -f 3)

$ curl -v -H "Host: ${SERVICE_HOSTNAME}" http://${INGRESS_HOST}:${INGRESS_PORT}/v1/models/${MODEL_NAME}:predict -d ${INPUT_PATH}
```

Expected Output:

```
*   Trying 10.100.34.202...
* Connected to 10.100.34.202 (10.100.34.202) port 80 (#0)
> POST /v1/models/enwiki-goodfaith:predict HTTP/1.1
> Host: enwiki-goodfaith.kubeflow-user.example.com
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
{"predictions": {"prediction": true, "probability": {"false": 0.03387957196040836, "true": 0.9661204280395916}}}
```

### Delete the InferenceService

```
$ kubectl delete -f ./editquality/enwiki-goodfaith/service.yaml
```

Expected Output

```
$ inferenceservice.serving.kubeflow.org "enwiki-goodfaith" deleted
```