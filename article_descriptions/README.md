# How to run locally

First we need to create a virtual environment and install the dependencies:
This can be done with:
```console
python -m venv .venv
source .venv/bin/activate
pip install -r article_descriptions/model_server/requirements.txt
```


Download the models from huggingface and place them in the same directory named PATH_TO_MODEL_DIR.
https://huggingface.co/facebook/mbart-large-cc25
https://huggingface.co/bert-base-multilingual-uncased

We clone the descartes repository by running:
> git clone https://github.com/wikimedia/descartes.git --branch 1.0.0 article_descriptions/model_server/descartes

Now our PATH_TO_MODEL_DIR directory contains the models and has the following structure:
```
PATH_TO_MODEL_DIR
├── bert-base-multilingual-uncased
├── mbart-large-cc25
```

We can run the server locally with:
> MODEL_PATH=PATH_TO_MODEL_DIR MODEL_NAME=article-descriptions python article_descriptions/model_server/model.py

 On a separate terminal we can make a request to the server with:
> curl -s localhost:8080/v1/models/article-model:predict -X POST -d '{"lang": "en", "title": "Clandonald", "num_beams": 2}' -i --header "Content-type: application/json"
