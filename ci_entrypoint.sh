#!/bin/bash
git init
git add .
if [[ "$1" == "model-server" || "$1" == "model_server" || "$1" == "transformer" || "$1" == "revscoring_model" || "$1" == "." ]]; then
  # for backward compatibility, temporarily accommodate older test variants that still have tox.ini in a specific dir
  tox -c "$1/tox.ini" -e ci-lint
else
  # cater to new test variants that have tox.ini in root dir
  tox -e "$1,$2"
fi
