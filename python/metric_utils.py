from sys import getsizeof
from itertools import chain
from collections import deque

from prometheus_client import Histogram

PROM_LABELS = ["model_name"]
FETCH_SIZE_BYTE = Histogram(
    "fetch_size_bytes",
    "fetched data size in bytes",
    buckets=[1000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000],
    labelnames=PROM_LABELS,
)

PRE_SIZE_BYTE = Histogram(
    "preprocess_size_bytes",
    "preprocessed data size in bytes",
    buckets=[1000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000],
    labelnames=PROM_LABELS,
)


def get_labels(model_name):
    return {PROM_LABELS[0]: model_name}


def total_size(o, handlers={}):
    """Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """

    def dict_handler(d):
        return chain.from_iterable(d.items())

    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
