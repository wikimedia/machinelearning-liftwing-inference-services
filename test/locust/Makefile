VENV = my_locust_venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
LOCUST = $(VENV)/bin/locust
PYTHON_PATH = .

export https_proxy := http://webproxy:8080
export PYTHONPATH := $(PYTHON_PATH):$(PYTHONPATH)

# Please keep this list sorted for easier maintenance
# See https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html for
# docs on what phony targets are and how to use them.
.PHONY: \
clean \
run-locust-test

# Generic command to run locust test for any model isvc.
run-locust-test: $(VENV)/bin/activate
	MODEL=$(MODEL_LOCUST_DIR) $(LOCUST) --headless --csv results/$(MODEL_LOCUST_DIR)

# Clean the environment
clean:
	rm -rf __pycache__
	rm -rf $(VENV)

# Create virtual environment and install dependencies
$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
