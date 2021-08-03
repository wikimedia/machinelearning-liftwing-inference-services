CLUSTER_IP=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.clusterIP}')
SERVICE_HOSTNAME=$(kubectl -n kubeflow-user get inferenceservice ${MODEL_NAME} -o jsonpath='{.status.url}' | cut -d "/" -f 3)
curl -v -H "Host: ${SERVICE_HOSTNAME}" -H "Cookie: authservice_session=${SESSION}" http://${CLUSTER_IP}/v1/models/${MODEL_NAME}:predict -d ${INPUT_PATH}
