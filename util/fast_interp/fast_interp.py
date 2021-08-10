import numpy as np
import numba

# # calculation data for ode
# dust_calc = np.dtype([
#                 ("ks_jdx", np.int32),
#                 ("cbar", np.float64),
#                 ("r_nu", np.float64, MAX_REACTANTS),
#                 ("S", np.float64),
#                 ("catS", np.float64),
#                 ("Js", np.float64),
#                 ("dadt", np.float64),
#                 ("ncrit", np.float64),
#             ], align=True)



@numba.njit()
def cspline(x: np.ndarray, y: np.ndarray):
    n = x.size

    b = np.zeros(n-1)
    d = np.zeros(n-1)
    c = np.zeros(n)

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

    return b, c[:-1], d
    #S_{n-1}(x) = y_{n-1} + b_{n-1}(x - x_{n-1}) + c_{n-1}(x - x_{n-1})^2 + d_{n-1}(x - x_{n-1})^3
    # b_i = f_i - m_i (h_{i+1}^2) / 6, i = 1...n-1
    # a_i = (f_{i+1} - f_i) / (h_{i+1}) - (h_{i+1})(m_{i+1} - m_i) / 6
    # c_i = h_i / 6

@numba.njit()
def binsearch(val: np.float64, x: np.ndarray) -> np.int64:
    l = 0
    r = x.size
    while l < r:
        m = np.int64((l+r)/2)
        if x[m] > val:
            r = m
        else:
            l = m + 1
    print(val, x[r-1], x[r])
    return r

@numba.njit()
def eval_cspline(t: float, x: np.ndarray, y: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    ix = binsearch(t, x)
    h = t - x[ix]
    yy = y[ix] + b[ix] * h + c[ix] * h * h + d[ix] * h * h * h
    return yy

@numba.njit()
def eval_cspline_derv(t: float, x: np.ndarray, y: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    ix = binsearch(t, x)
    h = t - x[ix]
    yy = b[ix] + 2. * c[ix] * h + 3. * d[ix] * h * h
    return yy

x = np.linspace(0, 2.*np.pi, 1000)
y = np.cos(x)

b, c, d = cspline(x, y)

test_ex = np.sin(np.pi)
test_sp = eval_cspline_derv(np.pi, x, y, b, c, d)


print(test_ex, test_sp)
