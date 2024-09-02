# Reference Need

The reference-need model computes a reference need score of a Wikipedia article's revision.

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Proposed/Multilingual_reference_need

## How to run locally

### 1. Build Python venv and install dependencies
First add the top level directory of the repo to the PYTHONPATH:
```console
export PYTHONPATH=$PYTHONPATH:.
```

Create a virtual environment and install the dependencies using:
```console
python -m venv .venv
source .venv/bin/activate
pip install -r src/models/reference_need/model_server/requirements.txt
```

### 2. Download the model
Download the `model.pkl` from the link below and place it in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/reference-quality/reference-need/

### 3. Run the server
We can run the server locally with:
```console
MODEL_PATH=<PATH_TO_MODEL_DIR/model.pkl> MODEL_NAME=reference-need python src/models/reference_need/model_server/model.py
```

On a separate terminal we can make a request to the server with:
```console
curl localhost:8080/v1/models/reference-need:predict -X POST -d '{"rev_id": 1242378206, "lang": "en"}' -H "Content-type: application/json"
```
