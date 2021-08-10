import os

if os.environ.get("NBKODE_NONUMBA", 0):
    from . import numbasub as numba  # noqa: F401

    def is_jitted(func):
        return True

    NO_NUMBA = True

else:
    import numba  # noqa: F401
    from numba.extending import is_jitted  # noqa: F401

    numba.jitclass = numba.experimental.jitclass

    NO_NUMBA = False