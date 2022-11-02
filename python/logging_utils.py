import logging

import kserve.constants


def set_log_level(level=None):
    """Helper function to reset the logging level and all the default logger's
    handlers. Useful when the logging level and handlers are already set
    elsewhere in previous code and you don't have control on it.
    """
    if not level:
        level = kserve.constants.KSERVE_LOGLEVEL
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
