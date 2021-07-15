# Dockerfile for building a Docker container.  See https://docs.docker.com/engine/reference/builder/
# Install 
#    1. Revscoring with dependencies
#    2. Draftquality
#    3. KFServing
#    4. Python ML code that uses Revscoring and KFServing to take an input ( wikipedia article revision id )
#       and return an output ( prediction )

FROM docker-registry.wikimedia.org/buster:20210530

ENV APP_HOME /app
WORKDIR $APP_HOME
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-setuptools \
    g++ \
    gfortran \
    liblapack-dev \
    libopenblas-dev \
    libenchant1c2a \
    aspell-ar \
    aspell-bn \
    aspell-el \
    hunspell-id \
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

COPY requirements.txt model.py ./

RUN pip3 install --upgrade --user pip \
  && pip3 install --no-cache-dir -r ./requirements.txt \
  && python3 -m nltk.downloader omw sentiwordnet stopwords wordnet

CMD ["python3", "model.py"]
