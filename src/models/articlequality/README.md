# Language-agnostic Article Quality

The language-agnostic articlequality inference service takes a Wikipedia article revision ID and the language it was written in, then returns a score between 0 and 1 (0 = bad; 1 = good quality).

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Proposed/Language-agnostic_Wikipedia_article_quality
* Source: https://github.com/wikimedia/research-api-endpoint-template/blob/quality-article/model/wsgi.py
* Model: https://analytics.wikimedia.org/published/wmf-ml-models/articlequality/language-agnostic/
* Model license: CC0 License


## How to run locally
In order to run the articlequality model server locally, please choose one of the two options below:

<details>
<summary>1. Automated setup using the Makefile</summary>

### 1.1. Build
In the first terminal run:
```console
make articlequality
```
This build process will set up: a Python venv, install dependencies, download the model(s), and run the server.

### 1.2. Query
On the second terminal query the isvc using:
```console
curl -s localhost:8080/v1/models/articlequality:predict -X POST -d '{"rev_id": 12345, "lang": "en"}' -i -H "Content-type: application/json"
```

### 1.3. Remove
If you would like to remove the setup run:
```console
MODEL_TYPE=articlequality make clean
```
</details>
<details>
<summary>2. Manual setup</summary>

### 2.1 Build Python venv and install dependencies
First add the top level directory of the repo to the PYTHONPATH:
```console
export PYTHONPATH=$PYTHONPATH:.
```

Create a virtual environment and install the dependencies using:
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/models/articlequality/requirements.txt
pip install -r python/requirements.txt
```

### 2.2. Download the model
Download the `model.pkl` from the link below and place it in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/articlequality/language-agnostic/

### 2.3. Run the server
We can run the server locally with:
```console
MODEL_PATH=PATH_TO_MODEL_DIR MAX_FEATURE_VALS=PATH_TO_MAX_FEATURE_VALS MODEL_NAME=articlequality python3 src/models/articlequality/model_server/model.py
```
PATH_TO_MAX_FEATURE_VALS is the absolute path to `data/max-vals-html-dumps-ar-en-fr-hu-tr-zh.tsv`.

On a separate terminal we can make a request to the server with:
```console
curl -s localhost:8080/v1/models/articlequality:predict -X POST -d '{"rev_id": 12345, "lang": "en"}' -i -H "Content-type: application/json"
```
</details>
