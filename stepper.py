import os, sys, logging, itertools
from particle import Particle
from typing import List, Tuple, NamedTuple

import numpy as np
from numba import from_dtype, njit, prange, double, void, jit, vectorize, types

from gas import SNGas
from network import Network, nucleation_par_dt

from physical_constants import k_B, stdP, istdP
from simulation_constants import N_MOMENTS, MIN_CONCENTRATION, MAX_REACTANTS, twothird, fourpi, fourover27

# some definitions
# ----------------
# jit: Just In-Time. Code will be compiled at run-time.

# controls if @jit compile in debug mode
S_DEBUG    = True
# if True, throw an error at startup if code wrapped in @jit cannot be converted from python
# if False, code that cannot be converted is allowed to run in python
S_NOPYTHON = True
# compile @jit code with threading enabled
S_PARALLEL = False
# compile @jit code with relaxed precision requirements, allowing for more aggresive optimization
S_FASTMATH = False

# calculation data for ode
nucleation_calc_dt = np.dtype([
                ("ks_jdx", np.int64),
                ("cbar", np.float64),
                ("r_nu", np.float64, MAX_REACTANTS),
                ("S", np.float64),
                ("catS", np.float64),
                ("Js", np.float64),
                ("dadt", np.float64),
                ("ncrit", np.float64),
            ], align=True)

# numba types for offloading
numba_dust_params_dt = types.Array(from_dtype(nucleation_par_dt), 1, 'C', readonly=True)
numba_dust_calcs_dt = types.Array(from_dtype(nucleation_calc_dt), 1, 'C', readonly=False)

@jit(void(numba_dust_calcs_dt, numba_dust_params_dt, double[:], double[:]), debug=S_DEBUG, nopython=S_NOPYTHON, parallel=S_PARALLEL, fastmath=S_FASTMATH)
#@jit(void(numba_dust_calcs_dt, types.Array(numba_dust_type, 1, 'C', readonly=True), double[:], double[:]), debug=S_DEBUG, nopython=S_NOPYTHON, parallel=S_PARALLEL, fastmath=S_FASTMATH)
def dust_pre(calc_t, dust_t, y, cb):
    for i in prange(calc_t.size):
        calc_t[i].ks_jdx = np.argmin(y[dust_t[i].keysp_idx[:dust_t[i].nkeysp]])
        calc_t[i].cbar = cb[dust_t[i].keysp_idx[calc_t[i].ks_jdx]]

        for j in range(dust_t[i].nr):
            calc_t[i].r_nu[j] = dust_t[i].react_nu[j] / dust_t[i].react_nu[calc_t[i].ks_jdx]

        calc_t[i].S = 0.0
        calc_t[i].Js = 0.0
        calc_t[i].dadt = 0.0
        calc_t[i].ncrit = 0.0

@jit(void(numba_dust_calcs_dt, numba_dust_params_dt, double[:], double), debug=S_DEBUG, nopython=S_NOPYTHON, parallel=S_PARALLEL, fastmath=S_FASTMATH)
def dust_state(calc_t, dust_t, y, T):

    kT = k_B * T

    # hopefully this gets executed in parallel
    for i in prange(calc_t.size):
        if dust_t[i].active == 0: continue

        # the kj-ndx refers to the "local" position in the list of ks for this reaction
        # the ki-ndx refers to the "global" position in the solution array
        ks_jdx = calc_t[i].ks_jdx
        ks_idx = dust_t[i].keysp_idx[ks_jdx]
        c1 = y[ks_idx]

        nks_idx = dust_t[i].react_idx[ dust_t[i].react_idx[:dust_t[i].nr] != ks_idx ]
        # holy shit this acutally works
        nks_jdx = np.where( dust_t[i].react_idx[np.arange(0, dust_t[i].nr)] != ks_idx )

        delg_reduced = (dust_t[i].A / T - dust_t[i].B) + np.sum(np.log(y[nks_idx] * kT * istdP) * calc_t[i].r_nu[nks_jdx])
        lnS = np.log( c1 * kT * istdP ) + delg_reduced

        lnS += np.sum(np.log(y[nks_idx] * kT * istdP) * calc_t[i].r_nu[nks_jdx])
        calc_t[i].catS = (stdP / kT) * np.exp(-delg_reduced)

        w = 1.0 + np.sum(dust_t[i].react_nu[nks_jdx])
        Pii = 1.0 * np.prod( np.power( y[nks_idx] / c1, calc_t[i].r_nu[nks_jdx] ) )

        # this could be conditioned on "ncrit", but for now let's go with
        # it and see if it holds
        if lnS > 0.0:
            iw = 1. / w
            Pii = np.power(Pii, iw)
            mu = fourpi * dust_t[i].a02 * dust_t[i].sigma / kT

            # if there are numerical issues, they likely arise here.
            # this variable should not grow very much, and if >~100 then
            # something is wack
            expJ = -fourover27 * np.power( mu, 3.0 ) / (lnS * lnS)

            Jkin = np.sqrt( 2.0 * dust_t[i].sigma / (np.pi * dust_t[i].react_mass[ks_jdx]) )

            calc_t[i].S  = np.exp(lnS)
            calc_t[i].Js = dust_t[i].omega0 * Jkin * c1 * c1 * Pii * np.exp(expJ)

            calc_t[i].dadt = dust_t[i].omega0 * np.sqrt( 0.5 * kT / (np.pi * dust_t[i].react_mass[ks_jdx]) ) * c1 * (1. - 1./calc_t[i].S)
            calc_t[i].ncrit = np.power( twothird * (mu / lnS), 3.0) + iw


@jit((numba_dust_calcs_dt, numba_dust_params_dt, double[:], double[:], double[:]), debug=S_DEBUG, nopython=S_NOPYTHON, parallel=S_PARALLEL, fastmath=S_FASTMATH)
def dust_moments(calc_t, dust_t, y, cbar, dydt):

    for i in prange(calc_t.size):
        if dust_t[i].active == 0: continue
        if calc_t[i].ncrit < 2.0: continue

        gidx = dust_t[i].prod_idx[0]
        dydt[gidx] = calc_t[i].Js / calc_t[i].cbar

        for j in range(1, N_MOMENTS):
            jdbl = np.float64(j)
            dydt[gidx + j] = dydt[gidx] * np.power(calc_t[i].ncrit, jdbl / 3.) \
                             + (jdbl / dust_t[i].a0) * calc_t[i].dadt * y[gidx + j - 1]

        dydt[dust_t[i].react_idx[:dust_t[i].nr]] -= calc_t[i].cbar * dydt[gidx + 3] * calc_t[i].r_nu[:dust_t[i].nr]

#########################
# chemistry, after network change I haven't put these back in yer
#########################
# @jit((numba_arr_type[:], double[:], double), nopython=S_NOPYTHON, parallel=S_PARALLEL, fastmath=S_FASTMATH)
# def arrhenius_rates(arr_t, y, T):
#     rates = np.zeros(arr_t.size)
#     for i in prange(arr_t.size):
#         rates[i] = arr_t[i].alpha * (T / 300.) ** arr_t[i].beta * np.exp(-arr_t[i].gamma / T)
#     return rates

# @jit((numba_arr_type[:], double[:], double, double[:]), nopython=S_NOPYTHON, parallel=S_PARALLEL, fastmath=S_FASTMATH)
# def arrhenius_f(arr_t, y, T, dydt):
#     if T > 1.0E4: return
#     rates = arrhenius_rates(arr_t, y, T)
#     for i in range(arr_t.size):
#         r_idx = arr_t[i].react_idx[:arr_t[i].nr]
#         p_idx = arr_t[i].prod_idx[:arr_t[i].np]

#         # TODO: add nu
#         f = rates[i] * np.prod( y[r_idx ] )
#         dydt[ r_idx ] -= f
#         dydt[ p_idx ] += f

@jit((double, double[:], double[:]), debug=S_DEBUG, nopython=S_NOPYTHON, parallel=S_PARALLEL, fastmath=S_FASTMATH)
def expand(xpand, y, dydt):
    dydt += y * xpand

@jit((numba_dust_params_dt, double[:]), debug=S_DEBUG, nopython=S_NOPYTHON, parallel=S_PARALLEL, fastmath=S_FASTMATH)
def check_active(dust_t, y):
    for i in range(dust_t.size):
        if np.any(y[dust_t[i].keysp_idx[:dust_t[i].nkeysp]] < 1.0E-1):
            dust_t[i].active = 0

@jit(double[:](numba_dust_params_dt, double[:], double[:], double, double), debug=S_DEBUG, nopython=S_NOPYTHON, parallel=S_PARALLEL, fastmath=S_FASTMATH)
def _f(dust_t, y, cb, T, dT):
    dydt = np.zeros(y.size)
    calc_t = np.empty(dust_t.size, dtype=nucleation_calc_dt)
    dust_pre(calc_t, dust_t, y, cb)
    #if dT < 0: # TODO: this isn't physical, fix
    #check_active(dust_t, y)
    dust_state(calc_t, dust_t, y, T)
    dust_moments(calc_t, dust_t, y, cb, dydt)
    return dydt

#############################################################
from util.fast_interp.fast_interp import cspline_create, eval_cspline, eval_cspline_derv, interp_ndt
import periodictable as pt


def build_rhs_fn(part: Particle, net: Network):
    #dat = build_rhs_data(part, net)
    Tspline = cspline_create(part.times, part.temperatures)
    Dspline = cspline_create(part.times, part.densities)
    xspline = cspline_create(part.times, part.position)

    dpar = net.generate_nparrays()["nucleation"]

    @njit
    def givago():
        return dpar

    cbar = np.zeros(net.NG)
    c0 = np.zeros(net.NG)
    m0 = part.mass[part.first_idx]
    v0 = part.volume[part.first_idx]

    for sp, idx in net.species_map.items():
        try:
            c0[idx] = part.composition[sp]
        except KeyError:
            print(f"WARNING: can't find {sp} in particle data")

    @njit
    def rhs_fn(t, y):
        T = eval_cspline(t, Tspline)
        dT =  eval_cspline_derv(t, Tspline)
        rho = eval_cspline(t, Dspline)
        drho = eval_cspline_derv(t, Dspline)

        # # call calculators
        xpnd = drho / rho
        vol = m0 / rho

        cbar = c0 * v0 / vol
        dydt = _f(dpar, y, cbar, T, dT)
        expand(xpnd, y[0:c0.size], dydt[0:c0.size])

        return dydt

    return rhs_fn

class Stepper(object):
    def __init__(self, part: Particle, net: Network):
        self._net = net
        self._dust_calc = np.empty(self._net.ND, dtype=nucleation_calc_dt)

        # TODO: hacking now to finish, come back and fix this garbage
        _tmp = self._net.generate_nparrays()
        self._dust_par = _tmp["nucleation"]
        # self._arr_par = _tmp["arrhenius"]
        self._cbar = np.zeros(self._net.solution_size)
        self._dust_par[:]["active"] = 1

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

        self._Tspline = cspline_create(part.times, part.temperatures)
        self._Dspline = cspline_create(part.times, part.densities)
        self._xspline = cspline_create(part.times, part.position)

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

    def initial_value(self) -> np.array:
        return self._c0

    def cbar(self, volume: np.float64) -> np.ndarray:
        return self._c0 * (self._v0) / volume

    def Temperature(self, t:np.float64) ->np.float64:
        return eval_cspline(t, self._Tspline)

    def Density(self, t:np.float64) -> np.float64:
        return eval_cspline(t, self._Dspline)

    def InitialMass(self) -> np.float64:
        return self._m0

    def __call__(self, t: np.float64, y: np.array) -> np.ndarray:

        # get interpolant data
        T = eval_cspline(t, self._Tspline)
        dT =  eval_cspline_derv(t, self._Tspline)
        rho = eval_cspline(t, self._Dspline)
        drho = eval_cspline_derv(t, self._Dspline)

        # call calculators
        xpnd = drho / rho
        vol = self._m0 / rho
        self._cbar[:] = self.cbar(vol)
        dydt = _f(self._dust_calc, self._dust_par, y, self._cbar, T, dT)
        expand(xpnd, y[0:self._net.NG], dydt[0:self._net.NG])

        return dydt

    def emit_done(self):
        return np.all(self._dust_par["active"] == 0)

