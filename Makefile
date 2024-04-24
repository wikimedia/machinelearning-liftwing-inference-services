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
readability \
articletopic-outlink-predictor \
articletopic-outlink-transformer \
logo-detection \
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

# Command for readability model-server
readability:
	@$(MAKE) download-nltk-punkt run-server MODEL_NAME="readability" \
	MODEL_URL="readability/multilingual/20230824102026/model.pkl" \
	MODEL_SERVER_PARENT_DIR="readability_model" \
	MODEL_PATH="models/readability/multilingual/20230824102026/model.pkl" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR="." \
	CUT_DIRS=2 \
	ACCEPT_REGEX="."

# Command for articletopic-outlink predictor
articletopic-outlink-predictor:
	@$(MAKE) run-server MODEL_NAME="outlink-topic-model" \
	MODEL_URL="articletopic/outlink/20221111111111/model.bin" \
	MODEL_SERVER_PARENT_DIR="outlink_topic_model" \
	MODEL_PATH="models/articletopic/outlink/20221111111111/model.bin" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR="." \
	CUT_DIRS=2 \
	ACCEPT_REGEX="." \
	PREDICTOR_PORT=8181

# Command for articletopic-outlink transformer
articletopic-outlink-transformer:
	. $(VENV)/bin/activate && \
	$(PYTHON) outlink_topic_model/transformer/transformer.py \
	--predictor_host="localhost:8181" --model_name="outlink-topic-model"

# Command for logo-detection model-server
logo-detection:
	@$(MAKE) run-server MODEL_NAME="logo-detection" \
	MODEL_URL="logo-detection/20240417132942/logo_max_all.keras" \
	MODEL_SERVER_PARENT_DIR="logo_detection" \
	MODEL_PATH="models/logo-detection/20240417132942/logo_max_all.keras" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR="." \
	CUT_DIRS=2 \
	ACCEPT_REGEX="."

# Download NLTK Punkt sentence tokenizer used by readability
download-nltk-punkt: $(VENV)/bin/activate
	@$(PYTHON) -m nltk.downloader punkt

# Clone descartes repository if not already present
clone-descartes:
	@if [ ! -d "article_descriptions/model_server/descartes" ]; then \
		git clone https://github.com/wikimedia/descartes.git --branch 1.0.1 article_descriptions/model_server/descartes; \
	fi

# Generic command to run any model server.
# Adds port argument if defined (e.g articletopic-outlink defines it because it uses both a predictor and transformer)
run-server: $(VENV)/bin/activate $(MODEL_PATH)
	MODEL_PATH=$(MODEL_PATH) MODEL_NAME=$(MODEL_NAME) \
	$(PYTHON) $(MODEL_SERVER_PARENT_DIR)/$(MODEL_SERVER_DIR)/model.py \
	$(if $(PREDICTOR_PORT), --http_port=$(PREDICTOR_PORT))

# Create virtual environment and install dependencies
$(VENV)/bin/activate: $(MODEL_SERVER_PARENT_DIR)/$(MODEL_SERVER_DIR)/$(DEP_DIR)/requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r python/requirements.txt
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
