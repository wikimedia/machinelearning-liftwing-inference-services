import time

import pytest

from python.decorators import log_slow_function


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


if __name__ == "__main__":
    pytest.main()
