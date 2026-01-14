# Article Topic Outlink

The articletopic-outlink inference service takes a Wikipedia article title and the language it was written in then uses links in the article to predict a set of topics that the article might be relevant to.

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Production/Language_agnostic_link-based_article_topic
* Model: https://analytics.wikimedia.org/published/wmf-ml-models/articletopic/outlink/
* Model license: CC0 License

## Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `page_title` | string | One of `page_title` or `page_id` | Wikipedia article title |
| `page_id` | integer | One of `page_title` or `page_id` | Wikipedia page ID |
| `lang` | string | Yes | Language code (e.g., "en") |
| `revision_id` | integer | No | Specific revision ID to analyze |
| `threshold` | float | No | Confidence threshold for results (default: 0.5) |
| `debug` | boolean | No | Enable debug mode (default: false) |

### Using `revision_id`

By default, the model fetches outlinks from the current state of the article using a fast single-query approach. When `revision_id` is provided, the model uses a 2-query approach to fetch outlinks from that specific revision:

1. Parse API call to get links from the specific revision
2. Batch queries to resolve Wikidata QIDs for those links

This is useful for:
- Reproducible predictions on historical revisions
- Analyzing how topic predictions change over time
- Processing events where you need predictions for a specific revision

Example with `revision_id`:
```console
curl localhost:8080/v1/models/outlink-topic-model:predict -i -X POST \
  -d '{"page_title": "Douglas_Adams", "lang": "en", "revision_id": 1264030954}'
```

**Performance Note** The revision-based approach is slower (~3-4x) than the default approach due to multiple API calls.

**Template rendering accuracy limitation** This is not purely the state of the page at that moment in time. Any templates are rendered as they currently exist, so if e.g., the navbox at the end of the article has changed substantially since the revision, those changes would be reflected in the results (as opposed to whatever state it was in when the revision was made).

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
