MODEL_NAME = revertrisk-language-agnostic
# TODO: assign the url to the latest timestamp in the public model dir
MODEL_URL = revertrisk/language-agnostic/20231117132654
MODEL_DIR = revert_risk_model
DEP_DIR = revertrisk

VENV = my_venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTHON_PATH = .
export PYTHONPATH := $(PYTHON_PATH):$(PYTHONPATH)

.PHONY: run clean

run: $(VENV)/bin/activate $(MODEL_DIR)/models/model.pkl
	MODEL_PATH=$(MODEL_DIR)/models/model.pkl MODEL_NAME=$(MODEL_NAME) $(PYTHON) $(MODEL_DIR)/model_server/model.py

$(VENV)/bin/activate: $(MODEL_DIR)/model_server/$(DEP_DIR)/requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(MODEL_DIR)/model_server/$(DEP_DIR)/requirements.txt

$(MODEL_DIR)/models/model.pkl:
	mkdir -p $(MODEL_DIR)/models
	curl --output-dir $(MODEL_DIR)/models -O https://analytics.wikimedia.org/published/wmf-ml-models/$(MODEL_URL)/model.pkl

clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	rm -rf $(MODEL_DIR)/models
