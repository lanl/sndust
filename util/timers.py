import numpy as np
from timeit import repeat

def timefunc(correct, s, func, *args, **kwargs):
    """
    Benchmark *func* and print out its runtime.
    """
    print(s.ljust(20), end=" ")
    # Make sure the function is compiled before the benchmark is
    # started
    res = func(*args, **kwargs)
    if correct is not None:
        assert np.allclose(res, correct), (res, correct)
    # time it
    print('{:>8.4f} us'.format(min(repeat(
        lambda: func(*args, **kwargs), number=100000, repeat=10))))
    return res
