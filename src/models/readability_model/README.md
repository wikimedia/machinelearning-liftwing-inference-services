# Readability

The readability inference service takes Wikipedia article text from a revision ID and predicts how hard it is for a reader to understand the article. It supports 100 languages.

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Proposed/Multilingual_readability_model_card
* Source: https://gitlab.wikimedia.org/repos/research/readability
* Model: https://analytics.wikimedia.org/published/wmf-ml-models/readability/multilingual/
* Model license: Apache 2.0 License


## How to run locally
In order to run the readability model server locally, please choose one of the two options below:

<details>
<summary>1. Automated setup using the Makefile</summary>

### 1.1. Build
In the first terminal run:
```console
make readability
```
This build process will set up: a Python venv, install dependencies, download the model(s), and run the server.

### 1.2. Query
On the second terminal query the isvc using:
```console
curl localhost:8080/v1/models/readability:predict -X POST -d '{"rev_id": 123456, "lang": "en"}' -H "Content-type: application/json"
```

### 1.3. Remove
If you would like to remove the setup run:
```console
MODEL_TYPE=readability make clean
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
pip install -r src/models/readability_model/model_server/requirements.txt
python3 -m nltk.downloader punkt
```

### 2.2. Download the model
Download the `model.pkl` from the link below and place it in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/readability/multilingual/

### 2.3. Run the server
We can run the server locally with:
```console
MODEL_PATH=PATH_TO_MODEL_DIR MODEL_NAME=readability python3 src/models/readability_model/model_server/model.py
```

On a separate terminal we can make a request to the server with:
```console
curl localhost:8080/v1/models/readability:predict -X POST -d '{"rev_id": 123456, "lang": "en"}' -H "Content-type: application/json"
```
</details>
