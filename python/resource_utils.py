import logging
import os
import pyopencl


def get_cpu_count():
    """
    Helper that returns the current cpu count, counting the restrictions
    imposed by Cgroups v2 (Cgroups v1 are not supported).
    The calculation is an approximation of what a container
    considers a virtual CPU.
    """
    # The os's cpu_count function is not cgroup aware, and from
    # https://github.com/python/cpython/issues/80235 it seems that there
    # is no plan to change its current behavior.
    host_cpu_count = os.cpu_count()
    if not host_cpu_count:
        logging.error("Failed to get the host's cpu count.")
    if not os.path.exists("/sys/fs/cgroup/cpu.max"):
        logging.info("Not inside a Cgroup v2, defaulting to the host's cpu count")
        return host_cpu_count
    with open("/sys/fs/cgroup/cpu.max") as f:
        try:
            cfs_quota_us, cfs_period_us = (v for v in f.read().strip().split())
            if cfs_quota_us == "max":
                logging.info(
                    "Found 'max' in the cpu.max file, defaulting to "
                    "the host's cpu count."
                )
                return host_cpu_count
            if int(cfs_quota_us) <= 0 or int(cfs_period_us) <= 0:
                logging.error(
                    "Found one or more zero values in cpu.max, defaulting to 1."
                )
                return 1
            cgroup_cpu_count = int(cfs_quota_us) // int(cfs_period_us)
            if cgroup_cpu_count < 1:
                logging.info(
                    "The cpu count calculated is less than one, defaulting to 1."
                )
                return 1
            return cgroup_cpu_count
        except ValueError:
            logging.exception(
                "The format of the cpu.max file doesn't contain two integers, "
                "defaulting to the host's cpu count."
            )
    return host_cpu_count


def gpu_is_available():
    try:
        platforms = pyopencl.get_platforms()
        for platform in platforms:
            devices = platform.get_devices(device_type=pyopencl.device_type.GPU)
            if devices:
                logging.info(f"Found GPU: {devices[0].name}")
                return True
        logging.info("Not found GPU.")
    except Exception as e:
        logging.exception(
            f"Exception occurred when detecting whether a GPU is available. Reason: {e}"
        )
    return False


def set_omp_num_threads():
    """
    Set the OMP_NUM_THREADS environment variable (if not already present)
    using the result of get_cpu_count().
    """
    num_threads = os.environ.get("OMP_NUM_THREADS")
    if num_threads:
        logging.info(
            f"The OMP_NUM_THREADS is already set to {num_threads}, "
            "not going to override it with the container's cpu count."
        )
        return
    cpu_count = str(get_cpu_count())
    os.environ["OMP_NUM_THREADS"] = cpu_count
    logging.info(f"Set OMP_NUM_THREADS to {cpu_count}.")
