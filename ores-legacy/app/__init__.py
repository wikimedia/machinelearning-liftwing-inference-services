import logging.config
import os

logging_level = os.getenv("LOG_LEVEL", "INFO").upper()
if logging_level not in set(logging._nameToLevel.keys()):
    logging.error(f"Logging level '{logging_level}' is not valid. Using INFO instead.")
    logging_level = "INFO"


logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "console": {"format": "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": logging_level,
            "formatter": "console",
        }
    },
    "loggers": {
        "": {
            "level": logging_level,
            "handlers": ["console"],
        }
    },
}

logging.config.dictConfig(logging_config)
