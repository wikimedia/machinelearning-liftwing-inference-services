# Article Descriptions

The article-descriptions inference service uses a Wikipedia article title and the language it was written in to recommend potential Wikidata article descriptions for the Wikipedia article. It supports 25 languages.

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Proposed/Article_descriptions
* Source: https://github.com/wikimedia/descartes/tree/transformers-wrapper
* Paper: https://arxiv.org/abs/2205.10012
* Model: https://drive.google.com/file/d/1bhn5O2WW6uXo4UvKDFoHqQnc0ozCCXmi/view?usp=sharing and
 https://analytics.wikimedia.org/published/wmf-ml-models/article-descriptions/
* Model license: MIT


## How to run locally
In order to run the article-descriptions model server locally, please choose one of the two options below:

<details>
<summary>1. Automated setup using the Makefile</summary>

### 1.1. Build
In the first terminal run:
```console
$ make article-descriptions
```
This build process will set up: a Python venv, install dependencies, download the model(s), and run the server.

### 1.2. Query
On the second terminal query the isvc using:
```console
$ curl localhost:8080/v1/models/article-descriptions:predict -X POST -d '{"lang": "en", "title": "Clandonald", "num_beams": 3}' -H "Content-Type: application/json" --http1.1
```

### 1.3. Remove
If you would like to remove the setup run:
```console
$ MODEL_TYPE=article-descriptions make clean
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
pip install -r article_descriptions/model_server/requirements.txt
```

Clone the descartes repository by running:
```console
git clone https://github.com/wikimedia/descartes.git --branch 1.0.1 article_descriptions/model_server/descartes
```

### 2.2. Download the model(s)
Download the models from the links below and place them in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/article-descriptions/mbart-large-cc25/
https://analytics.wikimedia.org/published/wmf-ml-models/article-descriptions/bert-base-multilingual-uncased/

Now our PATH_TO_MODEL_DIR directory contains the models and has the following structure:
```console
PATH_TO_MODEL_DIR
├── bert-base-multilingual-uncased
├── mbart-large-cc25
```

### 2.3. Run the server
We can run the server locally with:
> MODEL_PATH=PATH_TO_MODEL_DIR MODEL_NAME=article-descriptions python article_descriptions/model_server/model.py

 On a separate terminal we can make a request to the server with:
> curl -s localhost:8080/v1/models/article-descriptions:predict -X POST -d '{"lang": "en", "title": "Clandonald", "num_beams": 2}' -i --header "Content-type: application/json"
</details>