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
/srv/revscoring/python3.10/bin/python3.10 -m pip install -r /srv/revscoring/revscoring_model/requirements.txt
/srv/revscoring/python3.10/bin/python3.10 -m nltk.downloader omw sentiwordnet stopwords wordnet omw-1.4
