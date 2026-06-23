#!/bin/bash
set -ex

# This file contains common environment variables that represents good
# defaults for Python libraries that model servers import.
source common_settings.sh
MODEL_SERVER_PATH="${1:-model_server/model.py}"
# Run the model server
exec /srv/venv/bin/python ${MODEL_SERVER_PATH} "${@:2}"
