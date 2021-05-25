FROM ubuntu:20.04

ENV APP_HOME /app
WORKDIR $APP_HOME

RUN apt-get update && apt-get install -y \
        build-essential \
        wget \
        git \
        python3-pip \
        python3-dev \
        g++ \
        unzip \
        && rm -rf /var/cache/apk/*


# Install production dependencies.
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r ./requirements.txt

# Copy local code to container image
COPY model.py  ./

CMD ["python3", "model.py"]
