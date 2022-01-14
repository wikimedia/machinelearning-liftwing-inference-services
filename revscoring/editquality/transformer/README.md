# Predict on an InferenceService using custom revscoring model and transformer

The revscoring models call the mediawiki API in order to extract various features needed to score different types of models. We can move the API calls into a pre-processing step with a Transformer.

## Setup
1. Your ~/.kube/config should point to a cluster with [KServe installed](https://github.com/kserve/kserve#installation).
2. Your cluster's Istio Ingress gateway must be [network accessible](https://istio.io/latest/docs/tasks/traffic-management/ingress/ingress-control/).

##  Build Transformer image
This step can be part of your CI/CD pipeline to continuously build the transformer image version.
```shell
docker build -t {{username}}/revscoring-editquality-transformer:latest -f transformer.Dockerfile .
```
