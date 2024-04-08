# Logo Detection

The logo-detection inference service takes Wikimedia Commons image(s) and returns a prediction on whether they are a logo or not.

* Model Card: TBD
* Source: https://gitlab.wikimedia.org/mfossati/scriptz/-/blob/main/liftwing_prototype.py
* Paper: https://arxiv.org/abs/2104.00298
* Model: https://analytics.wikimedia.org/published/wmf-ml-models/logo-detection/
* Model license: Apache 2.0


## How to run locally
In order to run the logo-detection model server locally, please follow the instructions below:

<details>
<summary>1. Manual setup</summary>

### 1. Build Python venv and install dependencies
First add the top level directory of the repo to the PYTHONPATH:
```console
export PYTHONPATH=$PYTHONPATH:.
```

Create a virtual environment and install the dependencies using:
```console
python3 -m venv .venv
source .venv/bin/activate
pip install -r logo_detection/model_server/requirements.txt
```

### 1.2. Download the model
Download the `logo_max_all.keras` model from the link below and place it in the same directory named PATH_TO_MODEL_DIR.
https://analytics.wikimedia.org/published/wmf-ml-models/logo-detection/

### 1.3. Run the server
We can run the server locally with:
> MODEL_PATH=PATH_TO_MODEL_DIR MODEL_NAME=logo-detection python3 logo_detection/model_server/model.py

 On a separate terminal we can make a request to the server with:
> curl -s localhost:8080/v1/models/logo-detection:predict -X POST -d '{"instances": [ { "filename": "Cambia_logo.png", "url": "https://phab.wmfusercontent.org/file/data/mb6wynlvf3bdfw5e443f/PHID-FILE-wc27fvtkl6yv4rjdlqzn/Cambia_logo.png", "target": "logo" }, { "filename": "Blooming_bush_(14248894271).jpg", "url": "https://phab.wmfusercontent.org/file/data/46i23voto2a4aqwo6iyb/PHID-FILE-eldmzjv4p3vwsiwsuxya/Blooming_bush_%2814248894271%29.jpg", "target": "logo" }, { "filename": "12_rue_de_Condé_-_detail.jpg", "url": "https://phab.wmfusercontent.org/file/data/wxtr7be45udzyjzrojr6/PHID-FILE-tnu6mrji2smn2hpm6nhv/12_rue_de_Cond%C3%A9_-_detail.jpg", "target": "logo" } ] }' -i --header "Content-type: application/json"
</details>
