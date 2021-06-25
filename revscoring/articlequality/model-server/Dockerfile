# Dockerfile for building a Docker container.  See https://docs.docker.com/engine/reference/builder/
# Install 
#    1. Revscoring with dependencies
#    2. Articlequality
#    3. KFServing
#    4. Python ML code that uses Revscoring and KFServing to take an input ( wikipedia article revision id )
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
   && git clone https://github.com/wikimedia/articlequality.git \
   && pip3 install -r ./articlequality/requirements.txt \
   && pip3 install -r ./requirements.txt \
   && python3 ./articlequality/setup.py install \
   && python3 -m nltk.downloader omw sentiwordnet stopwords wordnet

COPY model.py ./articlequality/

CMD ["python3", "articlequality/model.py"]
