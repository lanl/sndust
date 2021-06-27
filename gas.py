import numpy as np
from typing import List, Tuple, Dict
import periodictable as pt

from physical_constants import amu2g
from particle import Particle
from scipy.interpolate import InterpolatedUnivariateSpline, Akima1DInterpolator

from network import Network

class SNGas(object):
    def __init__(self, part: Particle, net: Network):

        self._pid = part.pid
        self._sid = part.sid
        self._m0 = part.mass[part.first_idx]
        self._v0 = part.volume[part.first_idx]

        self._c0 = np.zeros(net.solution_size)

        for sp, idx in net.species_map.items():
            try:
                self._c0[idx] = part.composition[sp]
            except KeyError:
                print(f"WARNING: can't find {sp} in particle data")
                self._c0[idx] = 0.0

        self.premake("C","O","CO", net)
        self.premake("Si","O","SiO", net)

        # self._fT = InterpolatedUnivariateSpline(part.times, part.temperatures, k=1)
        # self._fD = InterpolatedUnivariateSpline(part.times, part.densities, k=1)
        # self._fx = InterpolatedUnivariateSpline(part.times, part.position[:,0], k=1)
        # self._fy = InterpolatedUnivariateSpline(part.times, part.position[:,1], k=1)

        self._fT = Akima1DInterpolator(part.times, part.temperatures)
        self._fD = Akima1DInterpolator(part.times, part.densities)
        self._fx = Akima1DInterpolator(part.times, part.position)
        self._fV = Akima1DInterpolator(part.times, part.velocity)


    def premake(self, s1, s2, sp, net):
        try:
            idx1 = net.sidx(s1)
            idx2 = net.sidx(s2)
            idxp = net.sidx(sp)
        except:
            return

        x1 = self._c0[idx1]
        x2 = self._c0[idx2]

        if x2 > x1:
            a = pt.formula(s1).mass / pt.formula(s2).mass
            self._c0[idxp] = x1
            self._c0[idx2] = x2 - x1
            self._c0[idx1] = 0.0
        else:
            a = pt.formula(s2).mass / pt.formula(s1).mass
            self._c0[idxp] = x2
            self._c0[idx1] = x1 - x2
            self._c0[idx2] = 0.0

        # uncomment if mass fraction is used in input
        # if x2 > x1:
        #     a = pt.formula(s1).mass / pt.formula(s2).mass
        #     self._c0[idxp] = x1 * (a + 1.)
        #     self._c0[idx2] = x2 - x1 * a
        #     self._c0[idx1] = 0.0
        # else:
        #     a = pt.formula(s2).mass / pt.formula(s1).mass
        #     self._c0[idxp] = x2 * (a + 1.)
        #     self._c0[idx1] = x1 - x2 * a
        #     self._c0[idx2] = 0.0

    @property
    def mass_0(self) -> np.float64:
        return self._m0

    @property
    def volume_0(self) -> np.float:
        return self._v0

    @property
    def concentration_0(self) -> np.array:
        return self._c0

    def cbar(self, volume: np.float64) -> np.float64:
        return self._c0 * (self._v0) / volume

    def Temperature(self, time: np.float64, derivative=0) -> np.float64:
        return self._fT(time,nu=derivative)

    def Density(self, time: np.float64, derivative=0) -> np.float64:
        return self._fD(time, nu=derivative)
    
    def Velocity(self, time: np.float64, derivative=0) -> np.float64:
        return self._fV(time, nu=derivative)
