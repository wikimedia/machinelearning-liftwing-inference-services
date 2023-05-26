import logging.config
import os

logging.config.fileConfig(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logging.ini"
    ),
    disable_existing_loggers=False,
)
