# Dockerfile for building a Docker container.  See https://docs.docker.com/engine/reference/builder/
# Install
#    1. Revscoring with dependencies
#    2. Drafttopic
#    3. KServe
#    4. Python ML code that uses Revscoring and KServe to take an input ( wikipedia article revision id )
#       and return an output ( prediction )

FROM docker-registry.wikimedia.org/python3-buster:0.0.1

ENV APP_HOME /app
WORKDIR $APP_HOME
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get install -y \
    python3-pip \
    python3-dev \
    python-dev \
    build-essential \
    python3-enchant \
    g++ \
    gfortran \
    git \
    wget \
    liblapack-dev \
    libopenblas-dev \
    libenchant1c2a \
    aspell-ar \
    aspell-bn \
    aspell-el \
    hunspell-id \
    hunspell-en-us \
    aspell-is \
    aspell-pl \
    aspell-ro \
    aspell-sv \
    aspell-ta \
    aspell-uk \
    myspell-cs \
    myspell-de-at \
    myspell-de-ch \
    myspell-de-de \
    myspell-es \
    myspell-et \
    myspell-fa \
    myspell-fr \
    myspell-he \
    myspell-hr \
    myspell-hu \
    myspell-lv \
    myspell-nb \
    myspell-nl \
    myspell-pt-pt \
    myspell-pt-br \
    myspell-ru \
    myspell-hr \
    hunspell-bs \
    hunspell-ca \
    hunspell-en-au \
    hunspell-en-us \
    hunspell-en-gb \
    hunspell-eu \
    hunspell-gl \
    hunspell-it \
    hunspell-hi \
    hunspell-sr \
    hunspell-vi \
    voikko-fi

COPY requirements.txt ./

RUN pip3 install --upgrade --user pip \
   && git clone https://github.com/wikimedia/drafttopic.git \
   # download word embedding vectors from app/drafttopic/Makefile by searching for strings that match the required targets. i.e search(grep) strings that start with "word2vec/" and remove(sed) ":" from them. also the first layer round brackets ( ) ensure that we don't leave the shell's current directory "app/"
   && (cd ./drafttopic && make $(grep '^word2vec/.*' Makefile | sed 's/:.$//')) \
   && mv ./drafttopic/word2vec ./ \
   && pip3 install -r ./drafttopic/requirements.txt \
   && pip3 install -r ./requirements.txt \
   && python3 ./drafttopic/setup.py install \
   && python3 -m nltk.downloader omw sentiwordnet stopwords wordnet

COPY model.py ./drafttopic/

CMD ["python3", "drafttopic/model.py"]
