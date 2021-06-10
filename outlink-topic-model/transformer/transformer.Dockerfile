FROM docker-registry.wikimedia.org/python3-build-buster:0.1.0

COPY  . .
RUN pip3 install --upgrade pip && pip install kfserving
RUN pip3 install -e .
ENTRYPOINT ["python3", "-m", "outlink_transformer"]
