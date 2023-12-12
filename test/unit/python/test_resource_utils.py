from unittest import mock

import pytest

from python.resource_utils import get_cpu_count


def test_get_cpu_count():
    """
    Test the various formats of the sysfs's cpu.max file, created
    when a Cgroup v2 is used.
    """
    mock_open = mock.mock_open(read_data="max 100000")
    mock_cpu_count = mock.Mock(return_value=8)
    mock_path_exists = mock.Mock(return_value=True)

    # If "max" is contained in the cpu.max file, we default to
    # the host's cpu count.
    # This can be tested with "docker run ..." without --cpus args.
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.cpu_count", mock_cpu_count):
            with mock.patch("os.path.exists", mock_path_exists):
                assert get_cpu_count() == 8

    # If we have two integers in the cpu.max file, we divide them
    # to get the number of cpus.
    # First case: the first integer is greater than the latter.
    # This can be tested with "docker run --cpus=2 ..."
    mock_open = mock.mock_open(read_data="200000 100000")
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.cpu_count", mock_cpu_count):
            with mock.patch("os.path.exists", mock_path_exists):
                assert get_cpu_count() == 2

    # If we have two integers in the cpu.max file, we divide them
    # to get the number of cpus.
    # Second case: the first integer is smaller than the latter.
    # This can be tested with "docker run --cpus=0.5 ..."
    mock_open = mock.mock_open(read_data="50000 100000")
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.cpu_count", mock_cpu_count):
            with mock.patch("os.path.exists", mock_path_exists):
                assert get_cpu_count() == 1

    # If we have two integers in the cpu.max file, we divide them
    # to get the number of cpus.
    # Third case: the first or the second integer is zero.
    # It shouldn't happen but better safe than sorry.
    mock_open = mock.mock_open(read_data="50000 0")
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.cpu_count", mock_cpu_count):
            with mock.patch("os.path.exists", mock_path_exists):
                assert get_cpu_count() == 1

    # If the cpu.max file doesn't exist, we default to the host's cpu count.
    mock_path_exists.return_value = False
    with mock.patch("builtins.open", mock_open):
        with mock.patch("os.cpu_count", mock_cpu_count):
            with mock.patch("os.path.exists", mock_path_exists):
                assert get_cpu_count() == 8


if __name__ == "__main__":
    pytest.main()
