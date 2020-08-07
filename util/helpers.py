from collections import abc
from functools import wraps
from time import time
import os, errno

from numpy import float64

def nested_dict_iter(nested):
    for key, value in nested.items():
        if isinstance(value, abc.Mapping):
            yield from nested_dict_iter(value)
        else:
            yield key, value

def time_fn(func):
    @wraps(func)
    def _time_it(*args, **kwargs) -> float64:
        start_time = time()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_time = time() - start_time
            return elapsed_time
    return _time_it
