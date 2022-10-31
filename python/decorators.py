import time
import logging

from functools import wraps


def elapsed_time_async(func):
    @wraps(func)
    async def elapsed_time_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        total = end - start
        logging.info(f"Function {func.__name__} took {total:.4f} seconds to execute.")
        return result

    return elapsed_time_wrapper
