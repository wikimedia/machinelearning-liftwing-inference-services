# Article Topic Outlink

The articletopic-outlink inference service takes a Wikipedia article title and the language it was written in then uses links in the article to predict a set of topics that the article might be relevant to.

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Production/Language_agnostic_link-based_article_topic
* Model: https://analytics.wikimedia.org/published/wmf-ml-models/articletopic/outlink/
* Model license: CC0 License


## How to run locally
In order to run the articletopic-outlink model server locally, please choose one of the two options below:

<details>
<summary>1. Automated setup using the Makefile</summary>

### 1.1. Build
In the first terminal run:
```console
make articletopic-outlink
```

### 1.2. Query
In the second terminal query the isvc using:
```console
curl localhost:8080/v1/models/outlink-topic-model:predict -i -X POST -d '{"page_title": "Douglas_Adams", "lang": "en"}'
```

### 1.3. Remove
If you would like to remove the setup run:
```console
MODEL_TYPE=articletopic make clean
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
pip install -r src/models/outlink_topic_model/model_server/requirements.txt
pip install -r python/requirements.txt
```

### 2.2. Download the model
Download the `model.bin` from the link below and place it in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/articletopic/outlink/20221111111111/

### 2.3. Run the server
Start the predictor:
```console
MODEL_PATH=PATH_TO_MODEL_DIR MODEL_NAME="outlink-topic-model" python3 src/models/outlink_topic_model/model_server/model.py
```

In the second terminal make a request to the server:
```console
curl localhost:8080/v1/models/outlink-topic-model:predict -i -X POST -d '{"page_title": "Douglas_Adams", "lang": "en"}'
```
</details>
