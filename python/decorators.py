import asyncio
import json
import logging
import time
import inspect
from functools import wraps
from python.metric_utils import (
    FETCH_SIZE_BYTE,
    PRE_SIZE_BYTE,
    get_labels,
    total_size,
)
from kserve.logging import trace_logger


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


def fetch_size_bytes(model_name: str):
    """Simple decorator for functions to log and export the size of fetched data in bytes."""

    def fetch_size_decorator(func):
        @wraps(func)
        def fetch_size_wrapper(*args, **kwargs):
            resp = func(*args, **kwargs)
            size = len(resp)
            FETCH_SIZE_BYTE.labels(**get_labels(model_name)).observe(size)
            trace_logger.info(
                f"Function {func.__name__} fetched data {size:.3f} bytes."
            )
            return resp

        async def fetch_size_wrapper_async(*args, **kwargs):
            resp = await func(*args, **kwargs)
            size = len(resp)
            FETCH_SIZE_BYTE.labels(**get_labels(model_name)).observe(size)
            trace_logger.info(
                f"Function {func.__name__} fetched data {size:.3f} bytes."
            )
            return resp

        if inspect.iscoroutinefunction(func):
            return fetch_size_wrapper_async
        else:
            return fetch_size_wrapper

    return fetch_size_decorator


def preprocess_size_bytes(model_name: str, key_name: str):
    """Simple decorator for functions to log and export the size of the preprocessed data in bytes."""

    def preprocess_size_decorator(func):
        @wraps(func)
        def preprocess_size_wrapper(*args, **kwargs):
            size = total_size(args[1][key_name])
            PRE_SIZE_BYTE.labels(**get_labels(model_name)).observe(size)
            trace_logger.info(f"Preprocessed data ({key_name}) {size:.3f} bytes.")
            return func(*args, **kwargs)

        return preprocess_size_wrapper

    return preprocess_size_decorator
