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

