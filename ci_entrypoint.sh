#!/bin/bash
git init
git add .
tox -c "$1/tox.ini" -e ci
