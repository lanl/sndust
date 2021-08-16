import numpy as np
from numba import from_dtype, njit, prange, double, void, int32, jit, vectorize, cfunc



MAX_KNOTS = 1024
interp_dt = np.dtype(
    [
        ("np", np.int32),
        ("knots", np.float64, (MAX_KNOTS)),
        ("coef", np.float64, (4, MAX_KNOTS))
    ], align=True
)

interp_ndt = from_dtype(interp_dt)

def cspline_create(x, y):
    n = x.size

    b = np.zeros(n-1)
    d = np.zeros(n-1)
    c = np.zeros(n)

    delx = np.zeros(n-1)
    delr = np.zeros(n-1)

    delx = np.diff(x)
    der = np.diff(y) / delx
    A = np.zeros((n, n))
    r = np.zeros((n, 1))
    A[0,0]      = 1
    A[-1,-1]    = 1
    for i in range(1,n-1):
        A[i, i] = 2.0 * (delx[i] + delx[i-1])
        A[i, i-1]  = delx[i-1]
        A[i, i+1]  = delx[i]
        r[i, 0] = 3.0 * (der[i] - der[i-1])
    c = np.linalg.solve(A, r)[:,0]
    d[:] = (c[1:] - c[:-1]) / (3.0 * delx[:])
    b[:] = der[:] - (delx[:]/3.)*(2.*c[:-1]+c[1:])
     #return b, c[:-1], d
    ret = np.empty(1, dtype=interp_ndt)[0]

    ret['np'] = n
    ret['knots'][:n:1] = x[::1]
    ret['coef'][0][:n-1:1]=y[:-1:1]
    ret['coef'][1][:n-1:1]=b[::1]
    ret['coef'][2][:n-1:1]=c[:-1:1]
    ret['coef'][3][:n-1:1]=d[::1]
    return ret
     #return ((n, ([b[::1], c[:-1:1], d[::,1]])))


@njit()
def binsearch(val: np.float64, x: np.ndarray) -> np.int64:
    l = 0
    r = x.size
    while l < r:
        m = np.int64((l+r)/2)
        if x[m] > val:
            r = m
        else:
            l = m + 1
    return r

@njit(double(double, interp_ndt))
def eval_cspline(t: float, indt: interp_ndt) -> np.float64:
    ix = binsearch(t, indt.knots)
    h = t - indt.knots[ix]
    h2 = h * h
    h3 = h2 * h
    yy = indt.coef[0][ix] + indt.coef[1][ix] * h + indt.coef[2][ix] * h2 + indt.coef[3][ix] * h3
    return yy

@njit(double(double, interp_ndt))
def eval_cspline_derv(t: float, indt: interp_ndt) -> np.float64:
    ix = binsearch(t, indt.knots)
    h = t - indt.knots[ix]
    h2 = h * h
    dy = indt.coef[1][ix] + 2.0 * indt.coef[2][ix] * h + 3.0 * indt.coef[3][ix] * h2
    return dy

# ##-------python versions for timing checks
def py_binsearch(val: np.float64, x: np.ndarray) -> np.int64:
    l = 0
    r = x.size
    while l < r:
        m = np.int64((l+r)/2)
        if x[m] > val:
            r = m
        else:
            l = m + 1
    #print(val, x[r-1], x[r])
    return r

def py_eval_cspline(t: float, indt: interp_ndt) -> np.float64:
    ix = binsearch(t, indt['knots'])
    h = t - indt['knots'][ix]
    h2 = h * h
    h3 = h2 * h
    yy = indt['coef'][0][ix] + indt['coef'][1][ix] * h + indt['coef'][2][ix] * h2 + indt['coef'][3][ix] * h3
    return yy

def py_eval_cspline_derv(t: float, indt: interp_ndt) -> np.float64:
    ix = binsearch(t, indt['knots'])
    h = t - indt['knots'][ix]
    h2 = h * h
    dy = indt['coef'][1][ix] + 2.0 * indt['coef'][2][ix] * h + 3.0 * indt['coef'][3][ix] * h2
    return dy
##----------

if __name__ == "__main__":
    # hack for now
    import sys
    sys.path.append('..')

    from timers import timefunc
    x = np.linspace(0, 2.*np.pi, 1000)
    y = np.cos(x)

    cos_spline = cspline_create(x, y)

    xx = 0.23 * np.pi

    correct = timefunc(None, "python cspline", py_eval_cspline, xx, cos_spline)
    timefunc(correct, "njit cspline", eval_cspline, xx, cos_spline)
    #yy = eval_cspline(xx, x, y, cos_spline)
    print(correct, np.cos(xx))



