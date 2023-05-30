VENV = venv
PIP = $(VENV)/bin/pip
LIFTWING_URL ?= https://inference.discovery.wmnet:30443
UVICORN ?= $(VENV)/bin/uvicorn app.main:app --reload --port 8000

.PHONY: create-venv clean test-server

create-venv:
	python3 -m venv $(VENV)
	source $(VENV)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -rf venv

test-server: create-venv
	$(UVICORN)
