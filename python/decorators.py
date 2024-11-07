import asyncio
import json
import logging
import time
import inspect
from functools import wraps
from python.metric_utils import total_size


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
        logging.info(f"Function {func.__name__} took {total:.2f} seconds to execute.")
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


def log_slow_function(threshold: float = 10.0):
    """Decorator to log function that takes longer than threshold seconds. This decorator works both
    for synchronous and asynchronous functions."""

    def decorator(func):
        @wraps(func)
        def log_slow_requests_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            total = end - start

            args_json = json.dumps(args)
            kwargs_json = json.dumps(kwargs)

            if total > threshold:
                logging.warning(
                    f"Function {func.__name__} called with args: {args_json}, kwargs: {kwargs_json}. "
                    f"Took {total:.2f} seconds to execute."
                )

            return result

        async def log_slow_requests_wrapper_async(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            end = time.perf_counter()
            total = end - start

            args_json = json.dumps(args)
            kwargs_json = json.dumps(kwargs)

            if total > threshold:
                logging.warning(
                    f"Function {func.__name__} called with args: {args_json}, kwargs: {kwargs_json}. "
                    f"Took {total:.2f} seconds to execute."
                )

            return result

        if asyncio.iscoroutinefunction(func):
            return log_slow_requests_wrapper_async
        else:
            return log_slow_requests_wrapper

    return decorator


def fetch_size_kb():
    """Simple decorator for functions to log the size of fetched data in KB."""

    def fetch_size_decorator(func):
        @wraps(func)
        def fetch_size_wrapper(*args, **kwargs):
            resp = func(*args, **kwargs)
            size = len(resp) / 1000
            logging.info(f"Function {func.__name__} fetched data {size:.3f} KB.")
            return resp

        async def fetch_size_wrapper_async(*args, **kwargs):
            resp = await func(*args, **kwargs)
            size = len(resp) / 1000
            logging.info(f"Function {func.__name__} fetched data {size:.3f} KB.")
            return resp

        if inspect.iscoroutinefunction(func):
            return fetch_size_wrapper_async
        else:
            return fetch_size_wrapper

    return fetch_size_decorator


def preprocess_size_kb(key_name: str):
    """Simple decorator for functions to log the size of the preprocessed data in KB."""

    def preprocess_size_decorator(func):
        @wraps(func)
        def preprocess_size_wrapper(*args, **kwargs):
            size = total_size(args[1][key_name]) / 1000
            logging.info(f"Preprocessed data ({key_name}) {size:.3f} KB.")
            return func(*args, **kwargs)

        return preprocess_size_wrapper

    return preprocess_size_decorator
