import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from kserve import utils as kserve_utils

from .decorators import elapsed_time_async


def create_process_pool(asyncio_aux_workers: int = None) -> ProcessPoolExecutor:
    """Create a Python Process pool to offload blocking/long cpu-bound code
    that can potentially block/stall the main asyncio loop thread.
    The default thread pool executor set by Kserve in [1] is meant
    for blocking I/O calls. In our case we run async HTTP calls only,
    and we need separate processes to run blocking CPU-bound.
    More info:
      https://docs.python.org/3/library/asyncio-eventloop.html#executing-code-in-thread-or-process-pools
      https://github.com/kserve/kserve/blob/release-0.8/python/kserve/kserve/model_server.py#L129-L130)

    Parameters:
        asyncio_aux_workers: the process pool's maximum number of workers.

    Returns:
        The instance of the Process Pool.
    """
    if asyncio_aux_workers is None:
        asyncio_aux_workers = min(32, kserve_utils.cpu_count() + 4)

    logging.info(
        "Create a process pool of {} workers to support "
        "model scoring blocking code.".format(asyncio_aux_workers)
    )
    return ProcessPoolExecutor(max_workers=asyncio_aux_workers)


def refresh_process_pool(process_pool: ProcessPoolExecutor, asyncio_aux_workers: int):
    """Shutdown and re-create a process pool. Useful when exeptions like
    BrokenProcessPool are raised (the pool is unusable after that).
    """
    process_pool.shutdown()
    return create_process_pool(asyncio_aux_workers)


@elapsed_time_async
async def run_in_process_pool(
    process_pool: ProcessPoolExecutor, function, *function_args
) -> Any:
    """Run a function in a ProcessPoolExecutor instance.
    Parameters:
        function: the function to run in the process pool. Please note:
                  There is some overhead in passing a function to another
                  process, since it involves pickle to serialize/deserialize
                  code and data.
        process_pool: the process pool executor instance.
        function_args: the function's arguments to use.

    Returns:
        Any, since the code is executed inside the process pool, and
        the return data is passed as-is.
    """
    return await asyncio.get_event_loop().run_in_executor(
        process_pool, function, *function_args
    )
