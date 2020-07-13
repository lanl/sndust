###############################################################################
# solver.py
# -----------------------------------------------------------------------------
# author: Christopher Mauney (mauneyc@lanl.gov)
# description: objects for calling the integration method
# date: 06/19/2020
###############################################################################

import logging, weakref
from typing import NamedTuple
from timeit import default_timer

import numpy as np
import scipy.integrate as integrate

import simulation_constants as sim_const
from stepper import Stepper
from observer import Observer

class SolverSpec(NamedTuple):
    """
        SolverSpec holds the parameters of the integrator
    """
    time_start: np.float64
    time_bound: np.float64
    max_timestep: np.float64
    absolute_tol: np.float64
    relative_tol: np.float64
    integrator: integrate.OdeSolver = integrate.LSODA

class Solver(object):
    """
        Solver invokes the stepper and observer.
    """
    def __init__(self, spec: SolverSpec, stepper: Stepper):
        self._ode = spec.integrator(stepper, spec.time_start, stepper.initial_value(), spec.time_bound, max_step=spec.max_timestep, \
                                        atol=spec.absolute_tol, rtol=spec.relative_tol, vectorized=False)
        self._steps = 0
        self._avg_steptime = 0.0 # running average of the timestep
        self._tot_steptime = 0.0 # accumilated time of all previous steps
        self._stepper = stepper

    def __call__(self, obs: Observer):
        msg = None
        while self._ode.status == "running":
            # see if the stepper has reached steady-state
            # i.e., no further changes will happen and the
            # current solution can be accepted
            if self._stepper.emit_done():
                msg = "complete (stepper reports no longer changing)"
                break

            _xtime0 = default_timer()
            msg = self._ode.step()

            self._tot_steptime += (default_timer() - _xtime0)
            self._avg_steptime = (self._tot_steptime / float(self._steps+1))

            # if any nan's have entered, print the last
            # state and try to leave gracefully
            if np.any(np.isnan(self._ode.y)):
                obs(self._steps, force_screen=True)
                raise ValueError("NAN detected, leaving")

            obs(self._steps)
            self._steps = self._steps + 1


        print(f"ODE loop exist\n\tstep msg=\'{msg}\'\n\tstatus=\'{self._ode.status}\'\n")
        obs.dump(self._steps)

