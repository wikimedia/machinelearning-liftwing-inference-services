import os

MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", 100))
MODEL_NAME = os.environ.get("MODEL_NAME")
MAX_CHAR_LENGTH = int(os.environ.get("MAX_CHAR_LENGTH", 1000))
