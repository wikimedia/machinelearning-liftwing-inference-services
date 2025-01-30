# Huggingface Serving Runtime
###### This readme has been based on the original [huggingface runtime README](https://github.com/kserve/kserve/tree/master/python/huggingfaceserver) and has been modified

The Huggingface serving runtime implements a runtime that can serve huggingface transformer based model out of the box.
The preprocess and post-process handlers are implemented based on different ML tasks, for example text classification,
token-classification, text-generation, text2text generation. Based on the performance requirement, you can choose to perform
the inference on a more optimized inference engine like triton inference server and vLLM for text generation.


## Run Huggingface Server Locally

1. Build the docker image
```bash
docker build --target production -f .pipeline/huggingface/blubber.yaml --platform=linux/amd64 -t hf:kserve .
```

2a. Downloading the model from huggingface

By defining the MODEL_ID env variable, the model will be downloaded from huggingface and saved in the models directory. The model name
is the name of the model that will be used to serve the model (part of the url).
```bash
docker run -p 8080:8080 -e MODEL_ID=bert-base-uncased -e MODEL_NAME=bert hf:kserve


WARNING: The requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64/v8) and no specific platform was requested
INFO:kserve:successfully loaded tokenizer for task: 5
Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMaskedLM: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
INFO:kserve:successfully loaded huggingface model from path bert-base-uncased
INFO:kserve:Registering model: bert
INFO:kserve:Setting max asyncio worker threads as 12
INFO:kserve:Starting uvicorn with 1 workers
2024-03-13 15:39:59.203 uvicorn.error INFO:     Started server process [1]
2024-03-13 15:39:59.206 uvicorn.error INFO:     Waiting for application startup.
2024-03-13 15:39:59.277 1 kserve INFO [start():62] Starting gRPC server on [::]:8081
2024-03-13 15:39:59.282 uvicorn.error INFO:     Application startup complete.
2024-03-13 15:39:59.284 uvicorn.error INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

2b. Run the server with a local model

``` bash
docker run -p 8080:8080 -e MODEL_NAME=bloom --rm -v /local/path/to/bloom-560m:/mnt/models/ hf:kserve

INFO:root:Copying contents of /mnt/models to local
INFO:kserve:successfully loaded tokenizer for task: 6
/opt/lib/python/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
INFO:kserve:successfully loaded huggingface model from path /mnt/models
INFO:kserve:Registering model: bloom
INFO:kserve:Setting max asyncio worker threads as 12
INFO:kserve:Starting uvicorn with 1 workers
2024-03-14 11:48:17.909 uvicorn.error INFO:     Started server process [1]
2024-03-14 11:48:17.911 uvicorn.error INFO:     Waiting for application startup.
2024-03-14 11:48:17.987 1 kserve INFO [start():62] Starting gRPC server on [::]:8081
2024-03-14 11:48:17.991 uvicorn.error INFO:     Application startup complete.
2024-03-14 11:48:17.994 uvicorn.error INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)

```

3. Test the server
```bash
curl -H "content-type:application/json" -v localhost:8080/v1/models/bert:predict -d '{"instances": ["The capital of france is [MASK]."] }'

{"predictions":["paris"]}
```

## Updating kserve and dependencies

The huggingfaceserver is one of the latest additions in kserve and is still under development. In order for us to be able
to incorporate the latest changes in kserve but also create a standard build we are using a [wikimedia fork of kserve](https://github.com/wikimedia/kserve)
and a specific branch named `liftwing`.

Starting from blubber version v.0.23.0 python runs inside a virtual environment. Because we are using both the
system packages from the base pytorch image as well as the requirements we install in the virtualenv we need to copy
them to the production variant. In order for pip to detect them and avoid reinstalling any packages we append both paths
to the PYTHONPATH environment variable.
In order to update to the latest version of huggingfaceserver and/or kserve, you can do the following:
- sync the `main` branch in our fork wikimedia/kserve repository to the kserve/kserve one via the `Update branch` button under the sync fork menu in the github UI.
  - If we plan to use a custom commit (in development phase) we can sync also the `liftwing` branch to the latest version of the `main` branch.
  - If we plan to use a stable release then we should also sync all the tags manually via cli.
    ```
    git remote update upstream https://github.com/kserve/kserve.git
    git fetch upstream --tags
    git push origin --tags
    ```


## Build for Apple Silicon processors

If you want to run things locally using an Apple Silicon processor you can build the image for the arm64 architecture using the dedicated blubberfile.
This uses a different base image and the `linux/arm64` platform without vllm as it is not yet supported.
```bash
docker build --target production -f .pipeline/huggingface/blubber_m1.yaml -t hf:kserve-m1 .
```


```bash
docker run -p 8080:8080 --rm -v /local/path/to/model/:/mnt/models/ -e MODEL_NAME=name hf:kserve-m1
```


### What to do when a new model is released on Huggingface

When a new model is released on Huggingface, we need to update the `huggingface` server to support it. The following steps should be taken:
If the architecture used by the model is already included in the `transformers` package there shouldn't be any steps needed to be taken and the model should be available for use.
If the architecture is not included in `transformers` the following sequence of steps should be taken:
1. an update is included in `transformers` and the model is available for use. Look for a specific mention of the model in the release notes (similar to [this for Llama 3.1](https://github.com/huggingface/transformers/releases/tag/v4.43.0))
2. We should make sure that this version of `transformers` is supported by the [huggingfaceserver in kserve](https://github.com/kserve/kserve/blob/master/python/huggingfaceserver/pyproject.toml#L15).
   If not we should update the [wikimedia/kserve fork](https://github.com/wikimedia/kserve) to allow the installation of this version by updating/relaxing the constraints in the `pyproject.toml` file.
3. Update our huggingface image in the inference-services repo to include the new version of the `transformers` package.
   To be sure we use a specific version of the package we can include an entry in the `requirements.txt` file in the `src/models/huggingface_modelserver` directory.
4. Upload the model on swift.
5. File a change in deployment-charts to either switch an existing service to the new model or add a new service with it. Things required to change for an existing deploymetn would be:
    - the name of the service
    - the model name
    - the model path in swift
