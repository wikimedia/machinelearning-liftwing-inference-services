#!/bin/bash
git init
git add .
pip uninstall virtualenv -y
tox -c "$1/tox.ini" -e ci
