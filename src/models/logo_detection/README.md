# Logo Detection

The logo-detection inference service takes Wikimedia Commons image(s) and returns a prediction on whether they are a logo or not.

* Model Card: TBD
* Source: https://gitlab.wikimedia.org/mfossati/scriptz/-/blob/main/liftwing_prototype.py
* Paper: https://arxiv.org/abs/2104.00298
* Model: https://analytics.wikimedia.org/published/wmf-ml-models/logo-detection/
* Model license: Apache 2.0


## How to run locally
> [!NOTE]
> Unfortunately, [tensorflow-cpu](https://pypi.org/project/tensorflow-cpu/) is not available for apple silicon. If you are a Mac user, please replace `tensorflow-cpu` with `tensorflow` in `src/models/logo_detection/model_server/requirements.txt`.

In order to run the logo-detection model server locally, please choose one of the two options below:

<details>
<summary>1. Automated setup using the Makefile</summary>

### 1.1. Build
In the first terminal run:
```console
make logo-detection
```
This build process will set up: a Python venv, install dependencies, download the model(s), and run the server.

### 1.2. Query
On the second terminal query the isvc using:
```console
curl -s localhost:8080/v1/models/logo-detection:predict -X POST -d '{"instances": [ { "filename": "Cambia_logo.png", "image": "'$(base64 -w0 < src/models/logo_detection/data/Cambia_logo.png)'", "target": "logo" }, { "filename": "Blooming_bush_(14248894271).jpg", "image": "'$(base64 -w0 < src/models/logo_detection/data/Blooming_bush_\(14248894271\).jpg)'", "target": "logo" }, { "filename": "12_rue_de_Condé_-_detail.jpg", "image": "'$(base64 -w0 < src/models/logo_detection/data/12_rue_de_Condé_-_detail.jpg)'", "target": "logo" } ] }' -i -H "Content-type: application/json"
```

### 1.3. Remove
If you would like to remove the setup run:
```console
MODEL_TYPE=logo-detection make clean
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
pip install -r src/models/logo_detection/model_server/requirements.txt
```

### 2.2. Download the model
Download the `logo_max_all.keras` model from the link below and place it in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/logo-detection/

### 2.3. Run the server
We can run the server locally with:
```console
MODEL_PATH=PATH_TO_MODEL_DIR MODEL_NAME=logo-detection python3 src/models/logo_detection/model_server/model.py
```

On a separate terminal we can make a request to the server with:
```console
curl -s localhost:8080/v1/models/logo-detection:predict -X POST -d '{"instances": [ { "filename": "Cambia_logo.png", "image": "'$(base64 -w0 < src/models/logo_detection/data/Cambia_logo.png)'", "target": "logo" }, { "filename": "Blooming_bush_(14248894271).jpg", "image": "'$(base64 -w0 < src/models/logo_detection/data/Blooming_bush_\(14248894271\).jpg)'", "target": "logo" }, { "filename": "12_rue_de_Condé_-_detail.jpg", "image": "'$(base64 -w0 < src/models/logo_detection/data/12_rue_de_Condé_-_detail.jpg)'", "target": "logo" } ] }' -i -H "Content-type: application/json"
```
</details>
