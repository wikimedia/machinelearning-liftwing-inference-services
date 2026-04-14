import asyncio
import logging
import time

import pytest

from python.decorators import elapsed_time, elapsed_time_async, log_slow_function


@log_slow_function(threshold=0.1)
def dummy_function(arg1: int, arg2: int, sleep_seconds: float = 0.2):
    """This is a dummy function to test the decorator"""
    time.sleep(sleep_seconds)


def test_log_slow_function(caplog):
    """
    Test that the log_slow_function decorator logs a message when the decorated function
    takes longer than the threshold
    """
    seconds_to_wait = 0.2
    _ = dummy_function(1, 2, sleep_seconds=seconds_to_wait)

    # Check that the log message is captured
    expected_log_message = f"""Function dummy_function called with args: [1, 2], kwargs: {{"sleep_seconds": {seconds_to_wait}}}."""
    assert expected_log_message in caplog.text


@elapsed_time
def fast_sync_function():
    return True


@elapsed_time(threshold=0.2)
def slow_sync_function():
    time.sleep(0.25)
    return True


@elapsed_time(threshold=0.5)
def under_threshold_sync_function():
    time.sleep(0.05)
    return True


@elapsed_time_async
async def fast_async_function():
    return True


@elapsed_time_async(threshold=0.2)
async def slow_async_function():
    await asyncio.sleep(0.25)
    return True


@elapsed_time_async(threshold=0.5)
async def under_threshold_async_function():
    await asyncio.sleep(0.05)
    return True


def test_elapsed_time_logs_by_default(caplog):
    caplog.set_level(logging.INFO)
    fast_sync_function()
    assert "Function fast_sync_function took" in caplog.text


def test_elapsed_time_threshold_logs_only_when_slow(caplog):
    caplog.set_level(logging.INFO)
    slow_sync_function()
    assert "Function slow_sync_function took" in caplog.text


def test_elapsed_time_threshold_skips_fast_calls(caplog):
    caplog.set_level(logging.INFO)
    under_threshold_sync_function()
    assert "Function under_threshold_sync_function took" not in caplog.text


@pytest.mark.asyncio
async def test_elapsed_time_async_logs_by_default(caplog):
    caplog.set_level(logging.INFO)
    await fast_async_function()
    assert "Function fast_async_function took" in caplog.text


@pytest.mark.asyncio
async def test_elapsed_time_async_threshold_logs_only_when_slow(caplog):
    caplog.set_level(logging.INFO)
    await slow_async_function()
    assert "Function slow_async_function took" in caplog.text


@pytest.mark.asyncio
async def test_elapsed_time_async_threshold_skips_fast_calls(caplog):
    caplog.set_level(logging.INFO)
    await under_threshold_async_function()
    assert "Function under_threshold_async_function took" not in caplog.text


if __name__ == "__main__":
    pytest.main()
