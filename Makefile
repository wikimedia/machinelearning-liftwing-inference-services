VENV = my_venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTHON_PATH = .

export PYTHONPATH := $(PYTHON_PATH):$(PYTHONPATH)

.PHONY: run \
revertrisk-language-agnostic \
revertrisk-multilingual \
article-descriptions \
language-identification \
clean

# Default run command for revertrisk-language-agnostic
run: revertrisk-language-agnostic

# Command for revertrisk-language-agnostic model-server
revertrisk-language-agnostic:
	@$(MAKE) run-server MODEL_NAME="revertrisk-language-agnostic" \
	MODEL_URL="revertrisk/language-agnostic/20231117132654/model.pkl" \
	MODEL_SERVER_PARENT_DIR="revert_risk_model" \
	MODEL_PATH="models/revertrisk/language-agnostic/20231117132654/model.pkl" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR="revertrisk" \
	CUT_DIRS=2 \
	ACCEPT_REGEX="."

# Command for revertrisk-multilingual model-server
revertrisk-multilingual:
	@$(MAKE) run-server MODEL_NAME="revertrisk-multilingual" \
	MODEL_URL="revertrisk/multilingual/20230810110019/model.pkl" \
	MODEL_SERVER_PARENT_DIR="revert_risk_model" \
	MODEL_PATH="models/revertrisk/multilingual/20230810110019/model.pkl" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR="multilingual" \
	CUT_DIRS=2 \
	ACCEPT_REGEX="."

# Command for article-descriptions model-server
article-descriptions: clone-descartes
	@$(MAKE) run-server MODEL_NAME="article-descriptions" \
	MODEL_URL="article-descriptions/" \
	MODEL_SERVER_PARENT_DIR="article_descriptions" \
	MODEL_PATH="models/article-descriptions/" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR="." \
	CUT_DIRS=2 \
	ACCEPT_REGEX="'(bert-base-multilingual-uncased|mbart-large-cc25)'"

# Command for language-identification model-server
language-identification:
	@$(MAKE) run-server MODEL_NAME="langid" \
	MODEL_URL="langid/lid201-model.bin" \
	MODEL_SERVER_PARENT_DIR="langid" \
	MODEL_PATH="models/langid/lid201-model.bin" \
	MODEL_SERVER_DIR="." \
	DEP_DIR="." \
	CUT_DIRS=2 \
	ACCEPT_REGEX="."

# Clone descartes repository if not already present
clone-descartes:
	@if [ ! -d "article_descriptions/model_server/descartes" ]; then \
		git clone https://github.com/wikimedia/descartes.git --branch 1.0.1 article_descriptions/model_server/descartes; \
	fi

# Generic command to run any model server
run-server: $(VENV)/bin/activate $(MODEL_PATH)
	MODEL_PATH=$(MODEL_PATH) MODEL_NAME=$(MODEL_NAME) $(PYTHON) $(MODEL_SERVER_PARENT_DIR)/$(MODEL_SERVER_DIR)/model.py

# Create virtual environment and install dependencies
$(VENV)/bin/activate: $(MODEL_SERVER_PARENT_DIR)/$(MODEL_SERVER_DIR)/$(DEP_DIR)/requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r $(MODEL_SERVER_PARENT_DIR)/$(MODEL_SERVER_DIR)/$(DEP_DIR)/requirements.txt

# Download the model file(s)
$(MODEL_PATH):
	mkdir -p $(MODEL_SERVER_PARENT_DIR)/models
	wget --no-host-directories --recursive --reject "index.html*" \
	--accept-regex $(ACCEPT_REGEX) --cut-dirs=$(CUT_DIRS) \
	--directory-prefix=models \
	--continue https://analytics.wikimedia.org/published/wmf-ml-models/$(MODEL_URL)

# Clean the environment
clean:
	rm -rf __pycache__
	rm -rf $(VENV)
	@if [ -z "$(MODEL_TYPE)" ]; then \
		echo "No MODEL_TYPE specified. Skipping model-specific cleanup."; \
	else \
		echo "Cleaning models in directory models/$(MODEL_TYPE) ..."; \
		rm -rf models/$(MODEL_TYPE); \
	fi
