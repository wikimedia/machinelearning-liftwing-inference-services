#!/bin/bash
git init
git add .
tox -c model-server/tox.ini -e ci
