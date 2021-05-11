# Dockerfile for building a Docker container.  See https://docs.docker.com/engine/reference/builder/
	# Install 
	#    1. Revscoring with dependencies
	#    2. English wikipedia goodfaith gradient boosting model
	#    3. Python Machine Learning code that takes an input ( wikipedia article revision id ) 
	#       and returns an output ( prediction )
	
	FROM ubuntu:20.04
	
	ENV APP_HOME /app
	WORKDIR $APP_HOME
	
	ARG DEBIAN_FRONTEND=noninteractive
	RUN apt-get update \
	    && apt-get upgrade -y \
	    && apt-get install -y \
	        python3-pip \
	        python3-dev \
	        g++ \
	        gfortran \
	        liblapack-dev \
	        libopenblas-dev \
	        libenchant1c2a \
	        aspell-ar \
	        aspell-bn \
	        aspell-el \
	        aspell-id \
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
	
	RUN pip3 install --no-cache-dir -r ./requirements.txt \
	   && python3 -m nltk.downloader omw sentiwordnet stopwords wordnet
	
	COPY model.py enwiki.goodfaith.gradient_boosting.model ./
	
	CMD ["python3", "model.py"]
