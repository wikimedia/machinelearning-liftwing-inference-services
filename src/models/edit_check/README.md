# Edit Check

The edit-check inference service compares the original and modified text (from a Wikipedia revision diff) and the language in which they were written to determine whether the new content uses peacock language.

* Model Card: TBD
* Source: TBD
* Model: https://analytics.wikimedia.org/published/wmf-ml-models/edit-check/
* Model license: TBD


## How to run locally

In order to run the edit-check model server locally, please choose one of the three options below:

<details>
<summary>1. Docker Compose</summary>

### 1.1. Download model
Download the model files from the link below and place them in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/edit-check/

Now our PATH_TO_MODEL_DIR directory contains the data files and has the following structure:
```console
PATH_TO_MODEL_DIR
└── edit-check
    └── peacock
        ├── config.json
        ├── model.safetensors
        ├── optimizer.pt
        ├── rng_state.pth
        ├── scheduler.pt
        ├── special_tokens_map.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── trainer_state.json
        ├── training_args.bin
        └── vocab.txt
```

Add PATH_TO_MODEL_DIR to a `.env` file, as shown below:
```console
echo 'PATH_TO_EDIT_CHECK_MODEL=/full/path/to/model/dir' > .env
```

### 1.2. Build image

<details>
<summary>1.2a GPU image</summary>

```console
docker compose build edit-check
```

This process will build an edit-check image with all dependencies installed.
</details>
<details>
<summary>1.2b CPU image</summary>

```console
docker compose build edit-check-cpu
```

This will build the cpu image for local testing in your localhost.
</details>

### 1.3. Run container
```console
docker compose up edit-check
```
This will run the container that hosts the model-server.

### 1.4. Query
On the second terminal query the isvc using:
```console
curl -s localhost:8080/v1/models/edit-check-staging:predict -X POST -d '{"instances": [{"lang": "en", "check_type": "tone", "original_text": "original text example original", "modified_text": "modified text example with hype"}]}' -i -H "Content-type: application/json"
```

Query locally on the cpu-version

```console
curl -s localhost:8080/v1/models/edit-check:predict -X POST -d '{"instances": [{"lang": "en", "check_type": "tone", "original_text": "original text example original", "modified_text": "modified text example with hype"}]}' -i -H "Content-type: application/json"
```

</details>
<details>
<summary>2. Makefile</summary>

### 2.1. Build
In the first terminal run:
```console
make edit-check
```
This build process will set up: a Python venv, install dependencies, download data file(s), and run the server.

### 2.2. Query
On the second terminal query the isvc using:
```console
curl -s localhost:8080/v1/models/edit-check:predict -X POST -d '{"instances": [{"lang": "en", "check_type": "tone", "original_text": "original text example original", "modified_text": "modified text example with hype"}]}' -i -H "Content-type: application/json"
```

### 2.3. Remove
If you would like to remove the setup run:
```console
MODEL_TYPE=edit-check make clean
```
</details>
<details>
<summary>3. Manual setup</summary>

### 3.1. Build Python venv and install dependencies
First add the top level directory of the repo to the PYTHONPATH:
```console
export PYTHONPATH=$PYTHONPATH:.
```

Create a virtual environment and install the dependencies using:
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/models/edit_check/model_server/requirements.txt
pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

### 3.2. Download model file(s)
Download the model files from the link below and place them in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/edit-check/

Now our PATH_TO_MODEL_DIR directory contains the data files and has the following structure:
```console
PATH_TO_MODEL_DIR
└── edit-check
    └── peacock
        ├── config.json
        ├── model.safetensors
        ├── optimizer.pt
        ├── rng_state.pth
        ├── scheduler.pt
        ├── special_tokens_map.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── trainer_state.json
        ├── training_args.bin
        └── vocab.txt
```

### 3.3. Run the server
We can run the server locally with:
```console
MODEL_NAME=edit-check MODEL_PATH=PATH_TO_MODEL_DIR python3 src/models/edit_check/model_server/model.py
```

On a separate terminal we can make a request to the server with:
```console
curl -s localhost:8080/v1/models/edit-check:predict -X POST -d '{"instances": [{"lang": "en", "check_type": "tone", "original_text": "original text example original", "modified_text": "modified text example with hype"}]}' -i -H "Content-type: application/json"
```
</details>
