# Article Country

The article-country inference service uses a Wikipedia article title and the language it was written in to determine which countries are relevant to this article.

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Proposed/Article_country
* Source: https://github.com/wikimedia/research-api-endpoint-template/blob/region-api/model/wsgi.py
* Model: currently rule-based with no model binary
* Model license: CC0 License


## How to run locally

In order to run the article-country model server locally, please choose one of the two options below:

<details>
<summary>1. Automated setup using the Makefile</summary>

### 1.1. Build
In the first terminal run:
```console
make article-country
```
This build process will set up: a Python venv, install dependencies, download data file(s), and run the server.

### 1.2. Query
On the second terminal query the isvc using:
```console
curl -s localhost:8080/v1/models/article-country:predict -X POST -d '{"lang": "en", "title": "Toni_Morrison"}' -i -H "Content-type: application/json"
```

### 1.3. Remove
If you would like to remove the setup run:
```console
MODEL_TYPE=article-country make clean
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
pip install -r src/models/article_country/requirements.txt
```

### 2.2. Download data file(s)
Download the data files from the link below and place them in the same directory named PATH_TO_DATA_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/article-country/

Now our PATH_TO_DATA_DIR directory contains the data files and has the following structure:
```console
PATH_TO_DATA_DIR
├── category-countries.tsv.gz
└── ne_10m_admin_0_map_units.geojson
```

### 2.3. Run the server
We can run the server locally with:
```console
MODEL_NAME=article-country DATA_PATH=PATH_TO_DATA_DIR python3 src/models/article_country/model_server/model.py
```

On a separate terminal we can make a request to the server with:
```console
curl -s localhost:8080/v1/models/article-country:predict -X POST -d '{"lang": "en", "title": "Toni_Morrison"}' -i -H "Content-type: application/json"
```
</details>
