import numpy as np
from .compatibility import numba

@numba.njit()
def isclose(a, b, atol, rtol):
    return np.abs(a - b) <= (atol + rtol * np.abs(b))


@numba.njit
def clip(x, xmin, xmax):
    return min(max(x, xmin), xmax)