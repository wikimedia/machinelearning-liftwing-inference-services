VENV = my_venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
PYTHON_PATH = .

export PYTHONPATH := $(PYTHON_PATH):$(PYTHONPATH)

# Please keep this list sorted for easier maintenance
# See https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html for
# docs on what phony targets are and how to use them.
.PHONY: \
article-country \
article-descriptions \
articlequality \
articletopic-outlink-predictor \
articletopic-outlink-transformer \
clean \
clone-descartes \
clone-wmf-kserve-numpy-200 \
download-nltk-punkt \
edit-check \
install-torch \
language-identification \
logo-detection \
readability \
reference-quality \
revertrisk-language-agnostic \
revertrisk-multilingual \
run \
run-server

# Default run command for revertrisk-language-agnostic
# NOTE: the first target in a Makefile is the default to run on just `make`
run: revertrisk-language-agnostic

# Generic command to run any model server.
# Adds isvc-specific arguments if defined:
# - articlequality uses the MAX_FEATURE_VALS to load the features data file.
# - articletopic-outlink uses PREDICTOR_PORT because it runs both a predictor
#   and transformer.
run-server: $(VENV)/bin/activate $(MODEL_PATH)
	MODEL_PATH=$(MODEL_PATH) MODEL_NAME=$(MODEL_NAME) \
	$(if $(MAX_FEATURE_VALS), MAX_FEATURE_VALS=$(MAX_FEATURE_VALS)) \
	$(PYTHON) $(MODEL_SERVER_PARENT_DIR)/$(MODEL_SERVER_DIR)/model.py \
	$(if $(PREDICTOR_PORT), --http_port=$(PREDICTOR_PORT)) \
	$(if $(MODEL_NAME_V2), MODEL_NAME_V2=$(MODEL_NAME_V2))

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

### Targets for running servers locally (plus direct deps)

# Command for article-country model-server
article-country:
	@$(MAKE) run-server MODEL_NAME="article-country" \
	MODEL_URL="article-country/20240901015102/" \
	MODEL_SERVER_PARENT_DIR="src/models/article_country" \
	MODEL_PATH="models/article-country/20240901015102/" \
	DATA_PATH="models/article-country/20240901015102/" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR=".." \
	CUT_DIRS=2 \
	ACCEPT_REGEX="'(category-countries.tsv.gz|ne_10m_admin_0_map_units.geojson|region-groundtruth-2025-01-01-qids.sqlite)'"

# Command for article-descriptions model-server
article-descriptions: clone-descartes
	@$(MAKE) run-server MODEL_NAME="article-descriptions" \
	MODEL_URL="article-descriptions/" \
	MODEL_SERVER_PARENT_DIR="src/models/article_descriptions" \
	MODEL_PATH="models/article-descriptions/" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR="." \
	CUT_DIRS=2 \
	ACCEPT_REGEX="'(bert-base-multilingual-uncased|mbart-large-cc25)'"

# Clone descartes repository if not already present (used by article-descriptions)
clone-descartes:
	@if [ ! -d "src/models/article_descriptions/model_server/descartes" ]; then \
		git clone https://github.com/wikimedia/descartes.git --branch 1.0.1 src/models/article_descriptions/model_server/descartes; \
	fi

# Command for articlequality model-server
articlequality: clone-wmf-kserve-numpy-200
	@$(MAKE) run-server MODEL_NAME="articlequality" \
    MODEL_URL="articlequality/language-agnostic/20250425125943/model.pkl" \
	MODEL_URL_V2="articlequality/language-agnostic/20250425125943/catboost_model.cbm" \
	MODEL_SERVER_PARENT_DIR="src/models/articlequality" \
	MODEL_PATH="models/articlequality/language-agnostic/20250425125943/model.pkl" \
	MODEL_PATH_V2="models/articlequality/language-agnostic/20250425125943/catboost_model.cbm" \
	MODEL_NAME_V2="articlequality_v2" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR=".." \
	CUT_DIRS=2 \
	ACCEPT_REGEX="." \
	MAX_FEATURE_VALS="src/models/articlequality/data/feature_values.tsv"

# Clone the wmf kserve fork that uses numpy v2.0.0 (used by articlequality)
clone-wmf-kserve-numpy-200:
	@if [ ! -d "src/models/articlequality/kserve_repository" ]; then \
		git clone --branch numpy-200 https://github.com/wikimedia/kserve.git src/models/articlequality/kserve_repository; \
	fi

# Command for articletopic-outlink predictor
articletopic-outlink:
	@$(MAKE) run-server MODEL_NAME="outlink-topic-model" \
	MODEL_URL="articletopic/outlink/20221111111111/model.bin" \
	MODEL_SERVER_PARENT_DIR="src/models/outlink_topic_model" \
	MODEL_PATH="models/articletopic/outlink/20221111111111/model.bin" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR="." \
	CUT_DIRS=2 \
	ACCEPT_REGEX="."

# Command for edit-check model-server
edit-check:
	@$(MAKE) install-torch run-server MODEL_NAME="edit-check" \
	MODEL_URL="edit-check/" \
	MODEL_SERVER_PARENT_DIR="src/models/edit_check" \
	MODEL_PATH="models/edit-check/peacock/" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR="." \
	CUT_DIRS=2 \
	ACCEPT_REGEX="'(peacock)'"

# install torch (used by edit-check and likely other model-servers that host LLMs or rely on docker-registry.wikimedia.org/amd-pytorch25:2.5.1rocm6.*)
install-torch: $(VENV)/bin/activate
	$(PIP) install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Command for language-identification model-server
language-identification:
	@$(MAKE) run-server MODEL_NAME="langid" \
	MODEL_URL="langid/lid201-model.bin" \
	MODEL_SERVER_PARENT_DIR="src/models/langid" \
	MODEL_PATH="models/langid/lid201-model.bin" \
	MODEL_SERVER_DIR="." \
	DEP_DIR="." \
	CUT_DIRS=2 \
	ACCEPT_REGEX="."

# Command for logo-detection model-server
logo-detection:
	@$(MAKE) run-server MODEL_NAME="logo-detection" \
	MODEL_URL="logo-detection/20240417132942/logo_max_all.keras" \
	MODEL_SERVER_PARENT_DIR="src/models/logo_detection" \
	MODEL_PATH="models/logo-detection/20240417132942/logo_max_all.keras" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR="." \
	CUT_DIRS=2 \
	ACCEPT_REGEX="."

# Command for readability model-server
readability:
	@$(MAKE) download-nltk-punkt run-server MODEL_NAME="readability" \
	MODEL_URL="readability/multilingual/20240805140437/model.bin" \
	MODEL_SERVER_PARENT_DIR="src/models/readability_model" \
	MODEL_PATH="models/readability/multilingual/20240805140437/model.bin" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR="." \
	CUT_DIRS=2 \
	ACCEPT_REGEX="."

# Download NLTK Punkt sentence tokenizer used by readability
download-nltk-punkt: $(VENV)/bin/activate
	@$(PYTHON) -m nltk.downloader punkt

# Command for revertrisk-language-agnostic model-server
revertrisk-language-agnostic:
	@$(MAKE) run-server MODEL_NAME="revertrisk-language-agnostic" \
	MODEL_URL="revertrisk/language-agnostic/20231117132654/model.pkl" \
	MODEL_SERVER_PARENT_DIR="src/models/revert_risk_model" \
	MODEL_PATH="models/revertrisk/language-agnostic/20231117132654/model.pkl" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR="revertrisk" \
	CUT_DIRS=2 \
	ACCEPT_REGEX="."

# Command for reference-quality model-server
reference-quality:
	@$(MAKE) run-server MODEL_NAME="reference-quality" \
	MODEL_URL="reference-quality/20250127142109/" \
	MODEL_SERVER_PARENT_DIR="src/models/reference_quality" \
	MODEL_PATH="models/reference-quality/20250127142109/reference-need/model.pkl" \
	FEATURES_DB_PATH="models/reference-quality/20250127142109/reference-risk/features.db" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR="." \
	CUT_DIRS=2 \
	ACCEPT_REGEX="'(reference-need|reference-risk)'"

# Command for revertrisk-multilingual model-server
revertrisk-multilingual:
	@$(MAKE) run-server MODEL_NAME="revertrisk-multilingual" \
	MODEL_URL="revertrisk/multilingual/20230810110019/model.pkl" \
	MODEL_SERVER_PARENT_DIR="src/models/revert_risk_model" \
	MODEL_PATH="models/revertrisk/multilingual/20230810110019/model.pkl" \
	MODEL_SERVER_DIR="model_server" \
	DEP_DIR="multilingual" \
	CUT_DIRS=2 \
	ACCEPT_REGEX="."

### Subtargets used by multiple other targets

# Create virtual environment and install dependencies
$(VENV)/bin/activate: $(MODEL_SERVER_PARENT_DIR)/$(MODEL_SERVER_DIR)/$(DEP_DIR)/requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r python/requirements.txt
	# Conditional installation based on MODEL_NAME to support local run requirements
	@if [ "$(MODEL_NAME)" = "articlequality" ]; then \
		$(PIP) install -r $(MODEL_SERVER_PARENT_DIR)/$(MODEL_SERVER_DIR)/$(DEP_DIR)/requirements_local_run.txt; \
	else \
		$(PIP) install -r $(MODEL_SERVER_PARENT_DIR)/$(MODEL_SERVER_DIR)/$(DEP_DIR)/requirements.txt; \
	fi

# Download the model file(s)
$(MODEL_PATH):
	mkdir -p $(MODEL_SERVER_PARENT_DIR)/models
	wget --no-host-directories --recursive --reject "index.html*" \
	--accept-regex $(ACCEPT_REGEX) --cut-dirs=$(CUT_DIRS) \
	--directory-prefix=models \
	--continue https://analytics.wikimedia.org/published/wmf-ml-models/$(MODEL_URL) \
	$(if $(MODEL_URL_V2), --continue https://analytics.wikimedia.org/published/wmf-ml-models/$(MODEL_URL_V2))
