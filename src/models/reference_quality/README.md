# Reference Need

The reference-need model computes a reference need score of a Wikipedia article's revision.

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Proposed/Multilingual_reference_need

## How to run locally

In order to run the reference-need model server locally, please choose one of the two options below:

<details>
<summary>1. Automated setup using the Makefile</summary>

### 1.1. Build
In the first terminal run:
```console
make reference-need
```
This build process will: set up a Python venv, install dependencies, download the model, and run the server.

### 1.2. Query
On the second terminal query the isvc using:
```console
curl -s localhost:8080/v1/models/reference-need:predict -X POST -d '{"rev_id": 1242378206, "lang": "en"}' -i -H "Content-type: application/json"
```

### 1.3. Remove
If you would like to remove the setup run:
```console
MODEL_TYPE=reference-need make clean
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
python -m venv .venv
source .venv/bin/activate
pip install -r src/models/reference_quality/model_server/requirements.txt
```

### 2.2. Download the model
Download the `model.pkl` from the link below and place it in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/reference-quality/reference-need/

### 2.3. Run the server
We can run the server locally with:
```console
MODEL_PATH=<PATH_TO_MODEL_DIR/model.pkl> MODEL_NAME=reference-need python src/models/reference_quality/model_server/model.py
```

On a separate terminal we can make a request to the server with:
```console
curl localhost:8080/v1/models/reference-need:predict -X POST -d '{"rev_id": 1242378206, "lang": "en"}' -H "Content-type: application/json"
```
</details>
