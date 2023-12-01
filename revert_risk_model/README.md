# How to run locally

Let's say we want to run revertrisk-language-agnostic model server locally. First we need to create a virtual environment and install the dependencies:

This can be done with:
```console
python -m venv .venv
source .venv/bin/activate
pip install -r revert_risk_model/model_server/revertrisk/requirements.txt
```

Download the model.pkl from the public repo (https://analytics.wikimedia.org/published/wmf-ml-models/revertrisk/) and place them in the directory named PATH_TO_MODEL_DIR.

To be able to use preprocess_utils.py in the python directory, we need to add the top level directory to the PYTHONPATH
> export PYTHONPATH=$PYTHONPATH:<PATH_TO_INFERENCE_SERVICES_REPO>

We can run the server locally with:
> MODEL_PATH=<PATH_TO_MODEL_DIR/model.pkl> MODEL_NAME=revertrisk-language-agnostic python revert_risk_model/model_server/model.py

On a separate terminal we can make a request to the server with:
> curl -s localhost:8080/v1/models/revertrisk-language-agnostic:predict -X POST -d '{"lang": "en", "rev_id": 12345}' -i --header "Content-type: application/json"
