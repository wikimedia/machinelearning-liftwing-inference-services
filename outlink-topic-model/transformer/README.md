# Predict on a InferenceService using custom fastText Server and Transformer

The outlink topic model requires a feature string containing all the outlinks
of a given article, so a pre-processing step is needed before making the
prediction call if the user is sending in raw input format (i.e. language and
article title). Additionally, we need to only return predictions based on
a confidence threshold, so a post-processing step is needed after the
prediction is made.

## Setup
1. Your ~/.kube/config should point to a cluster with [KFServing installed](https://github.com/kubeflow/kfserving/#install-kfserving).
2. Your cluster's Istio Ingress gateway must be [network accessible](https://istio.io/latest/docs/tasks/traffic-management/ingress/ingress-control/).

##  Build Transformer image
This step can be part of your CI/CD pipeline to continuously build the transformer image version.
```shell
docker build -t {{username}}/outlink-transformer:latest -f transformer.Dockerfile .
```
