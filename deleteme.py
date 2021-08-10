import numpy as np

import numba_ode

def rhs(t,y,p):
    return p * y

y0 = [1., 2., 2., 4., 8., 2.4, 1.4, 5.3, 3.4, 9.2, 8.5,2.0,10.4]
p = [-0.1, -0.5, -0.01, -0.3, -0.4, -0.02, -1.3, -0.3, -0.4, -0.1, -3.4, -1.1, -0.3]
t0 = 0
solver = numba_ode.LIRK(rhs, t0, y0, params=p)

ts = np.linspace(0, 10, 100)

import time

tic = time.perf_counter()

ts, ys = solver.run(ts)

toc = time.perf_counter()

print(f"took {toc-tic:0.8} s")
