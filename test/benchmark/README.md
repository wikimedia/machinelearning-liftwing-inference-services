## Load Test Benchmarking

### KFServing revscoring enwiki-goodfaith
- Create `InferenceService`
```bash
kubectl apply -f ../../enwiki-goodfaith/service.yaml -n kubeflow-user
```
- Create the input vegeta configmap
```bash
kubectl apply -f ./enwiki-goodfaith_vegeta_cfg.yaml -n kubeflow-user
```
- Create the benchmark job using [vegeta](https://github.com/tsenart/vegeta)
Note that you can configure pod anti-affinity to run vegeta on a different node on which the inference pod is running.
```bash
kubectl create -f ./enwiki-goodfaith_benchmark.yaml -n kubeflow-user
```

#### Viewing job logs
The load testing job takes ~10 minutes to complete. You can see current running
jobs:
```
kubectl get jobs -n kubeflow-user
```

Once the job is completed you can see the results in the logs.
First get the job's pod name:
```
kubectl get pods -n kubeflow-user
```

Copy the job pod's name and then use it here:
```
kubectl logs <job-pod-name> -n kubeflow-user
```

#### Delete a job from cluster
You can remove a job from the cluster like this:
```
kubectl delete job <job-name> -n kubeflow-user
```


