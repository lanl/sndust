import logging, weakref
from typing import NamedTuple
from timeit import default_timer

import numpy as np
import scipy.integrate as integrate

from simulation_parameters import SolverParameters
import stepper as step

# from stepper import Stepper
from gas import SNGas
from network import Network, nucleation_numpy_type
from numba import double


class SSolver(object):
    def __init__(self, params: SolverParameters, gas: SNGas, net: Network):
        self._gas = gas
        self._net = net
        self._dust_calc = np.empty(self._net.ND, dtype=step.dust_calc)

        # TODO: hacking now to finish, come back and fix this garbage
        _tmp = self._net.generate_nparrays()
        self._dust_par = _tmp["nucleation"]

        # self._arr_par = _tmp["arrhenius"]
        self._cbar = np.zeros(self._net.solution_size)
        self._dust_par[:]["active"] = 1
        self._abstol = params.abstol
        self._reltol = params.reltol

    def initial_value(self) -> np.array:
        return self._gas.concentration_0  # try to add bins

    def env_therms(self, t: np.float64):
        T = double(
            self._gas.Temperature(t)
        )  # this cast is necessary, not sure why just yet
        dT = double(self._gas.Temperature(t, derivative=1))

        rho = self._gas.Density(t)
        drho = self._gas.Density(t, derivative=1)
        return T, dT, rho, drho

    def f(self, t: np.float64, y: np.array) -> np.array:
        T, dT, rho, drho = self.env_therms(t)
        # call calculators
        xpnd = drho / rho
        vol = self._gas.mass_0 / rho
        self._cbar[:] = self._gas.cbar(vol)

        dydt = step._f(self._dust_calc, self._dust_par, y, self._cbar, T, dT)

        step.expand(xpnd, y[0 : self._net.NG], dydt[0 : self._net.NG])
        # erode(dadt, y[self._net.NG+self._net.ND*N_MOMENTS:], dydt[self._net.NG+self._net.ND*N_MOMENTS:], self._net.NG, self._dust_calc)
        # erode_grow(dadt, y, dydt, self._net.NG, self._dust_calc)

        return dydt

    def emit_done(self, t, y):
        if np.all(self._dust_par["active"] == 0):
            return -1
        return 1

    emit_done.terminal = True

    def saturating(self, t, y):
        T, dT, rho, drho = self.env_therms(t)
        # call calculators
        xpnd = drho / rho
        vol = self._gas.mass_0 / rho
        self._cbar[:] = self._gas.cbar(vol)

        step._e_sat(self._dust_calc, self._dust_par, y, self._cbar, T, dT)
        for i in range(self._dust_calc.size):
            _lns = self._dust_calc[i]["lnS"]
            if _lns > -1.0e-1:
                return 1.0
        return -100

    saturating.direction = 1.0

    def solve(self, t0, tf):
        sol = integrate.solve_ivp(
            self.f,
            [t0, tf],
            self.initial_value(),
            method="LSODA",
            dense_output=False,
            events=(self.saturating),
            rtol=self._reltol,
            atol=self._abstol,
            min_step=1.0e-5,
        )

        return sol


#        with open(ivp_sols_path, 'ab') as npf:
#            np.save(npf, sol)

# import matplotlib.pyplot as plt
# tt = np.linspace(spec.time_start, spec.time_bound, 2000)
# z = sol.sol(tt)
# plt.semilogy(tt, z[7].T)
#
# plt.semilogy(sol.t_events[1], sol.y_events[1], 'bo ')
#
# plt.show()
#

# class Solver(object):
#     def __init__(self, spec: SolverSpec, stepper: Stepper):
#         self._ode = spec.integrator(stepper, spec.time_start, stepper.initial_value(), spec.time_bound, max_step=spec.max_timestep, \
#                                         atol=spec.absolute_tol, rtol=spec.relative_tol, vectorized=False)
#         self._steps = 0
#         self._avg_steptime = 0.0
#         self._tot_steptime = 0.0
#         self._stepper = Stepper
#
#
#
#     def __call__(self, obs: Observer):
#         msg = None
#
#         # loop until the ode object halts
#         while self._ode.status == "running":
#             # if the stepper has no remaining work, leave loop
#             if self._stepper.emit_done():
#                 msg = "complete (stepper reports no longer changing)"
#                 break
#
#             # reset timer and take timestep
#             _xtime0 = default_timer()
#             msg = self._ode.step()
#
#             self._tot_steptime += (default_timer() - _xtime0)
#             self._avg_steptime = (self._tot_steptime / float(self._steps+1))
#
#             if np.any(np.isnan(self._ode.y)):
#                 obs(self._steps, force_screen=True)
#                 raise ValueError("NAN detected, leaving")
#
#             obs(self._steps)
#             self._steps = self._steps + 1
#
#
#         print(f"ODE loop exist\n\tstep msg=\'{msg}\'\n\tstatus=\'{self._ode.status}\'\n")
#
#         obs.dump(self._steps)
#
