import numpy as np
from typing import List, Tuple, Dict
import periodictable as pt

from physical_constants import amu2g
from simulation_parameters import PhysicsParameters
from particle import P_freex
import sys
from network import Network


class SNGas(object):
    def __init__(self, param: PhysicsParameters, part: P_freex, net: Network):
        self._pid = part.pid
        self._m0 = part.mass
        self._v0 = part.volume
        self._t0 = part.time_init
        self._temp0 = part.temp_init
        self._rho0 = part.rho_init
        self._c0 = np.zeros(net.solution_size)
        self._gamma = param.gas_gamma

        for sp, idx in net.species_map.items():
            try:
                spm = 1.0
                if not "(" in sp:
                    spm = pt.formula(sp).mass * amu2g
                self._c0[idx] = part.composition[sp] * self._rho0 / spm
            except KeyError:
                # print(f"WARNING: can't find {sp} in particle data")
                self._c0[idx] = 1.0e-20

        # print(
        #     f"[1m] C = {part.composition['C']:15.8E}, O = {part.composition['O']:15.8E}"
        # )
        # print(
        #     f"[1m] C = {self._c0[net.sidx('C')]*pt.formula('C').mass*amu2g/self._rho0:15.8E}, O = {self._c0[net.sidx('O')]*pt.formula('O').mass*amu2g/self._rho0:15.8E}"
        # )
        # print(
        #     f"[1] C = {self._c0[net.sidx('C')]:15.8E}, O = {self._c0[net.sidx('O')]:15.8E}"
        # )
        self.premake("C", "O", "CO", net)
        self.premake("Si", "O", "SiO", net)

        for i in range(net.solution_size):
            if np.abs(self._c0[i]) < 1.0e-20:
                self._c0[i] = 1.0e-20
        # print(
        #     f"[2] C = {self._c0[net.sidx('C')]:15.8E}, O = {self._c0[net.sidx('O')]:15.8E}"
        # )
        #
        # sys.exit(0)
        # self._fT = InterpolatedUnivariateSpline(part.times, part.temperatures, k=1)
        # self._fD = InterpolatedUnivariateSpline(part.times, part.densities, k=1)
        # self._fx = InterpolatedUnivariateSpline(part.times, part.position[:,0], k=1)
        # self._fy = InterpolatedUnivariateSpline(part.times, part.position[:,1], k=1)

        # self._fT = Akima1DInterpolator(part.times, part.temperatures)
        # self._fD = Akima1DInterpolator(part.times, part.densities)
        # self._fx = Akima1DInterpolator(part.times, part.position)
        # self._fV = Akima1DInterpolator(part.times, part.velocity)
        #

    def premake(self, s1, s2, sp, net):
        try:
            idx1 = net.sidx(s1)
            idx2 = net.sidx(s2)
            #            idxp = net.sidx(sp)
        except:
            return

        x1 = self._c0[idx1]
        x2 = self._c0[idx2]

        # if x2 > x1:
        #     self._c0[idxp] = x1
        #     self._c0[idx2] = x2 - x1
        #     self._c0[idx1] = 0.0
        # else:
        #     self._c0[idxp] = x2
        #     self._c0[idx1] = x1 - x2
        #     self._c0[idx2] = 0.0
        #
        # uncomment if mass fraction is used in input
        if x2 > x1:
            a = pt.formula(s1).mass / pt.formula(s2).mass
            # self._c0[idxp] = x1 * (a + 1.)
            self._c0[idx2] = x2 - x1 * a
            self._c0[idx1] = 0.0
        else:
            a = pt.formula(s2).mass / pt.formula(s1).mass
            # self._c0[idxp] = x2 * (a + 1.)
            self._c0[idx1] = x1 - x2 * a
            self._c0[idx2] = 0.0

    @property
    def mass_0(self) -> np.float64:
        return self._m0

    @property
    def volume_0(self) -> np.float64:
        return self._v0

    @property
    def concentration_0(self) -> np.array:
        return self._c0

    def cbar(self, volume: np.float64) -> np.float64:
        return self._c0 * (self._v0) / volume

    def Temperature(self, time: np.float64, derivative=0) -> np.float64:
        pgam = 3.0 * (1.0 - self._gamma)
        T = self._temp0 * np.power((time / self._t0), pgam)
        if derivative > 0:
            T *= pgam / time
        return T

    def Density(self, time: np.float64, derivative=0) -> np.float64:
        den = self._rho0 * np.power((time / self._t0), -3.0)
        if derivative > 0:
            den *= -3.0 / time
        return den

    def time_terminal(self, temp):
        return self._t0 * np.power(
            (temp / self._temp0), (1.0 / (3.0 * (1.0 - self._gamma)))
        )
