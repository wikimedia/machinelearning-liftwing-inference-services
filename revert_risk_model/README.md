# Revert Risk Language-agnostic

The revert-risk-language-agnostic inference service uses a Wikipedia article revision ID and the language it was written in to predict the risk of this revision being reverted.

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Proposed/Language-agnostic_revert_risk
* Source: https://gitlab.wikimedia.org/repos/research/knowledge_integrity/-/blob/research_notebooks/RRR/RevisionRevertsRisk_LanguageAgnostic.ipynb
* Model: https://analytics.wikimedia.org/published/wmf-ml-models/revertrisk/language-agnostic/
* Model license: Apache 2.0 License


## How to run locally
In order to run the revert-risk-language-agnostic model server locally, please choose one of the two options below:

<details>
<summary>1. Automated setup using the Makefile</summary>

### 1.1. Build
In the first terminal run:
```console
make revertrisk-language-agnostic
```
This build process will set up: a Python venv, install dependencies, download the model(s), and run the server.

### 1.2. Query
On the second terminal query the isvc using:
```console
curl localhost:8080/v1/models/revertrisk-language-agnostic:predict -i -X POST -d '{"lang": "en", "rev_id": 12345}'
```

### 1.3. Remove
If you would like to remove the setup run:
```console
MODEL_TYPE=revertrisk make clean
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
pip install -r revert_risk_model/model_server/revertrisk/requirements.txt
```

Clone the descartes repository by running:
```console
git clone https://github.com/wikimedia/descartes.git --branch 1.0.1 article_descriptions/model_server/descartes
```

### 2.2. Download the model
Download the `model.pkl` from the link below and place it in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/revertrisk/language-agnostic/

### 2.3. Run the server
We can run the server locally with:
```console
MODEL_PATH=<PATH_TO_MODEL_DIR/model.pkl> MODEL_NAME=revertrisk-language-agnostic python revert_risk_model/model_server/model.py
```

On a separate terminal we can make a request to the server with:
```console
curl -s localhost:8080/v1/models/revertrisk-language-agnostic:predict -X POST -d '{"lang": "en", "rev_id": 12345}' -i --header "Content-type: application/json"
```

### 2.4. Batch inference
If you want to use batch inference to request multiple predictions using a single request, set the `USE_BATCHER` environment variable to True when launching the model server:
```console
MODEL_PATH=<PATH_TO_MODEL_DIR/model.pkl> MODEL_NAME=revertrisk-language-agnostic USE_BATCHER=True python revert_risk_model/model_server/model.py
```

We have a different input schema for batch inference. For example, the input should be in JSON format as shown below:
```json
{
    "instances": [
      {
        "lang": "en",
        "rev_id": 123456
      },
      {
        "lang": "en",
        "rev_id": 23456
      },
      {
        "lang": "en",
        "rev_id": 12345
      }
    ]
}
```
</details>