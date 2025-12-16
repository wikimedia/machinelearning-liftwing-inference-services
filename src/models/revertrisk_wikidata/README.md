# Revertrisk Wikidata

The revertrisk-wikidata inference service uses the metadata and content of a Wikidata article revision ID to predict the risk of this revision being reverted.

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Production/RevertRisk_Wikidata
* Source: https://github.com/trokhymovych/wikidata-vandalism-detection
* Model: https://drive.google.com/drive/folders/1czw8zFRfkZKyxRFcwFB5PWgBIf_8QeK7?usp=sharing and https://analytics.wikimedia.org/published/wmf-ml-models/revertrisk/wikidata/20251104121312/
* Model license: Apache 2.0 License


## How to run locally

In order to run the revertrisk-wikidata model-server locally, please choose one of the two options below:

<details>
<summary>1. Automated setup using the Makefile</summary>

### 1.1. Build
In the first terminal run:
```console
make revertrisk-wikidata
```
This build process will set up: a Python venv, install dependencies, download the model, and run the server.

### 1.2. Query
On the second terminal query the isvc using:
```console
curl -s localhost:8080/v1/models/revertrisk-wikidata:predict -X POST -d '{"rev_id": 1892513445}' -i -H "Content-type: application/json"
```

### 1.3. Remove
If you would like to remove the setup run:
```console
MODEL_TYPE=revertrisk-wikidata make clean
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
pip install -r src/models/revertrisk_wikidata/requirements.txt
```

### 2.2. Download data file(s)
Download the model from the link below and place it in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/revertrisk/wikidata/20251104121312/

Now our PATH_TO_MODEL_DIR directory contains the model with the following structure:
```console
PATH_TO_MODEL_DIR
└── wikidata_revertrisk_graph2text_v2.pkl
```

### 2.3. Run the server
We can run the server locally with:
```console
MODEL_NAME=revertrisk-wikidata MODEL_PATH=PATH_TO_MODEL_DIR/wikidata_revertrisk_graph2text_v2.pkl python3 src/models/revertrisk_wikidata/model_server/model.py
```

On a separate terminal we can make a request to the server with:
```console
curl -s localhost:8080/v1/models/revertrisk-wikidata:predict -X POST -d '{"rev_id": 1892513445}' -i -H "Content-type: application/json"
```
</details>
