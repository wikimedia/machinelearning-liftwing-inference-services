# Reference Quality

The reference quality inference service contains two models:
1. Reference-need model uses revision content to predict the reference need score of that revision.

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Proposed/Multilingual_reference_need

2. Reference-risk model uses edit history metadata to predict the likelihood of a reference to survive on a Wikipedia article.

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Proposed/Language-agnostic_reference_risk

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
First add the top level directory of the repo to the `PYTHONPATH`:
```console
export PYTHONPATH=$PYTHONPATH:.
```

Create a virtual environment and install the dependencies using:
```console
python -m venv .venv
source .venv/bin/activate
pip install -r src/models/reference_quality/model_server/requirements.txt
```

### 2.2. Download the model and the features.db
Download the `model.pkl` and `features.db` from the link below and place it in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/reference-quality/


### 2.3. Run the server
We can run the server locally with:
```console
MODEL_PATH=<PATH_TO_MODEL_DIR/model.pkl> FEATURES_DB_PATH=<PATH_TO_MODEL_DIR/features.db> python src/models/reference_quality/model_server/model.py
```

On a separate terminal we can test making a request to the reference-need model with:
```console
curl localhost:8080/v1/models/reference-need:predict -X POST -d '{"rev_id": 1242378206, "lang": "en"}' -H "Content-type: application/json"
```
and the reference-risk model with:
```console
curl localhost:8080/v1/models/reference-risk:predict -X POST -d '{"rev_id": 1242378206, "lang": "en"}' -H "Content-type: application/json"
```
</details>
