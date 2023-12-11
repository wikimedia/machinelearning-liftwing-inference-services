import logging
import time
from functools import wraps


def elapsed_time_async(func):
    """Simple decorator for async functions to log their wall clock execution
    time.
    """

    @wraps(func)
    async def elapsed_time_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        total = end - start
        logging.info(f"Function {func.__name__} took {total:.4f} seconds to execute.")
        return result

    return elapsed_time_wrapper


def elapsed_time(func):
    """Simple decorator for functions to log their wall clock execution
    time.
    """

    @wraps(func)
    def elapsed_time_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        total = end - start
        logging.info(f"Function {func.__name__} took {total:.4f} seconds to execute.")
        return result

    return elapsed_time_wrapper
