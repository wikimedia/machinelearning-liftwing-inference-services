#!/bin/bash
git init
git add .
pip uninstall argparse -y
tox -c "$1/tox.ini" -e ci
