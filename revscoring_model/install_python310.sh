#!/usr/bin/env bash
set -euo pipefail

# build and install Python 3.10.14 into /srv/revscoring/python3.10
cd /tmp
wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
tar -xf Python-3.10.14.tgz
cd Python-3.10.14
./configure --enable-optimizations --prefix=/srv/revscoring/python3.10
make -j"$(nproc)"
make altinstall

# install packages with python3.10
cd /srv/revscoring/
/srv/revscoring/python3.10/bin/python3.10 -m ensurepip
/srv/revscoring/python3.10/bin/python3.10 -m pip install --upgrade pip
# Pre-install build tools so SudachiPy can be compiled from source below.
/srv/revscoring/python3.10/bin/python3.10 -m pip install "Cython==0.29.24" wheel setuptools_scm
# SudachiPy 0.5.2 has two issues when pip installs it via its normal isolated build env:
#   1. Cython 3.x (resolved at build time) crashes on tokenizer.pyx; --no-build-isolation
#      forces use of the Cython==0.29.24 already installed above.
#   2. setuptools_scm falls back to version "0.0.0" without git tags; the env var overrides it.
SETUPTOOLS_SCM_PRETEND_VERSION=0.5.2 \
    /srv/revscoring/python3.10/bin/python3.10 -m pip install --no-build-isolation "SudachiPy==0.5.2"
# Install remaining requirements; SudachiPy is already satisfied and will be skipped.
/srv/revscoring/python3.10/bin/python3.10 -m pip install -r /srv/revscoring/revscoring_model/requirements.txt
/srv/revscoring/python3.10/bin/python3.10 -m nltk.downloader omw sentiwordnet stopwords wordnet omw-1.4
