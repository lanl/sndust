import logging, weakref
from typing import NamedTuple
from timeit import default_timer
from atomic_mass import AMU
import numpy as np
import scipy.integrate as integrate

import simulation_constants as sim_const
from simulation_constants import *
from stepper import Stepper
from observer import Observer

kB_erg = 1.380649E-16

class SolverSpec(NamedTuple):
    time_start: np.float64
    time_bound: np.float64
    max_timestep: np.float64
    absolute_tol: np.float64
    relative_tol: np.float64
    #integrator: integrate.OdeSolver = integrate.DOP853
    integrator: integrate.OdeSolver = integrate.LSODA
#    integrator: integrate.OdeSolver = integrate.Radau

class Solver(object):
    def __init__(self, spec: SolverSpec, stepper: Stepper):
        self._ode = spec.integrator(stepper, spec.time_start, stepper.initial_value(), spec.time_bound, max_step=spec.max_timestep, \
                                        atol=spec.absolute_tol, rtol=spec.relative_tol, vectorized=False)
        self._steps = 0
        self._avg_steptime = 0.0
        self._tot_steptime = 0.0
        self._stepper = stepper
        self._shock = 0.0

    def __call__(self, obs: Observer):
        msg = None
        # loop until the ode object halts
        while self._ode.status == "running":
            # if the stepper has no remaining work, leave loop
            if self._stepper.emit_done():
                msg = "complete (stepper reports no longer changing)"
                break

            # reset timer and take timestep
            _xtime0 = default_timer()
            msg = self._ode.step()

            # check for shock
            self.shock = self._stepper._gas._fS(self._ode.t)

            if self.shock >= 0.5:
                t = self._ode.t
                rho = self._stepper._gas.Density(t)
                T = self._stepper._gas.Temperature(t)
                gas_name = list(self._stepper._net._species_gas)
                gas_conc = self._stepper._gas._c0[:self._stepper._net.NG]
                n_tot = sum([gas_conc[Sidx] * AMU[s.strip()] for Sidx,s in enumerate(gas_name)])
                press = n_tot * kB_erg * T
                # assume diatomic molecules for gas gamma = 7/5
                v_shock = np.sqrt(7.0/5.0 *press/rho)
                self._ode.y[self._stepper._net._NG + self._stepper._net._ND * N_MOMENTS + self._stepper._net._ND * numBins: -1] = 3.0/4.0 * v_shock

            self._tot_steptime += (default_timer() - _xtime0)
            self._avg_steptime = (self._tot_steptime / float(self._steps+1))

            if np.any(np.isnan(self._ode.y)):
                obs(self._steps, force_screen=True)
                raise ValueError("NAN detected, leaving")

            obs(self._steps)
            self._steps = self._steps + 1


        print(f"ODE loop exist\n\tstep msg=\'{msg}\'\n\tstatus=\'{self._ode.status}\'\n")
        obs.dump(self._steps)

