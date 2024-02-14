# Language Identification

The language-identification (langid) inference service takes a text snippet and predicts the language it was written in. It supports 201 languages.

* Model Card: https://meta.wikimedia.org/wiki/Machine_learning_models/Proposed/Language_Identification
* Source: https://github.com/laurieburchell/open-lid-dataset
* Paper: https://arxiv.org/pdf/2305.13820.pdf
* Model: https://data.statmt.org/lid/lid201-model.bin.gz and
 https://analytics.wikimedia.org/published/wmf-ml-models/langid/
* Model license: the GNU General Public License v3.0.


## How to run locally
In order to run the langid model server locally, please choose one of the two options below:

<details>
<summary>1. Automated setup using the Makefile</summary>

### 1.1. Build
In the first terminal run:
```console
make language-identification
```
This build process will set up: a Python venv, install dependencies, download the model(s), and run the server.

### 1.2. Query
On the second terminal query the isvc using:
```console
curl localhost:8080/v1/models/langid:predict -i -X POST -d '{"text": "Some random text in any language"}'
```

### 1.3. Remove
If you would like to remove the setup run:
```console
MODEL_TYPE=langid make clean
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
pip install -r langid/requirements.txt
```

### 2.2. Download the model
Download the `lid201-model.bin` from the link below and place it in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/langid/

### 2.3. Run the server
We can run the server locally with:
```console
MODEL_PATH=PATH_TO_MODEL_DIR MODEL_NAME=langid python langid/model.py
```

On a separate terminal we can make a request to the server with:
```console
curl localhost:8080/v1/models/langid:predict -i -X POST -d '{"text": "Some random text in any language"}'
```
</details>
