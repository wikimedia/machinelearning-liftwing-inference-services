import logging
import logging.config
import os
from copy import deepcopy

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


def configure_kserve_framework_logging(
    framework_log_level=None, access_log_level=None, trace_log_level=None
):
    """Configure KServe/Uvicorn logger levels with separate controls.

    Defaults:
    - framework loggers (`kserve`, `uvicorn.error`) -> INFO
    - high-volume request loggers (`uvicorn.access`, `kserve.trace`) -> WARNING
    """
    framework_log_level = framework_log_level or os.environ.get(
        "FRAMEWORK_LOG_LEVEL", "INFO"
    )
    access_log_level = access_log_level or os.environ.get("ACCESS_LOG_LEVEL", "WARNING")
    trace_log_level = trace_log_level or os.environ.get("TRACE_LOG_LEVEL", "WARNING")

    try:
        import kserve.logging as kserve_logging
    except Exception:
        # Fallback if KServe internals change: still force logger levels.
        logging.getLogger("kserve").setLevel(framework_log_level)
        logging.getLogger("uvicorn.error").setLevel(framework_log_level)
        logging.getLogger("uvicorn.access").setLevel(access_log_level)
        logging.getLogger("kserve.trace").setLevel(trace_log_level)
        return

    config = deepcopy(kserve_logging.KSERVE_LOG_CONFIG)
    logger_level_map = {
        "kserve": framework_log_level,
        "uvicorn.error": framework_log_level,
        "uvicorn.access": access_log_level,
        "kserve.trace": trace_log_level,
    }
    for logger_name, logger_level in logger_level_map.items():
        if logger_name in config.get("loggers", {}):
            config["loggers"][logger_name]["level"] = logger_level

    # Keep KServe module-level config aligned if startup re-applies it.
    kserve_logging.KSERVE_LOG_CONFIG = config
    logging.config.dictConfig(config)
