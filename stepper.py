import os, sys, logging, itertools
from typing import List, Tuple, NamedTuple
from time import time
import numpy as np
from numba import (
    from_dtype,
    njit,
    prange,
    double,
    void,
    int32,
    jit,
    vectorize,
    guvectorize,
    stencil,
)

from gas import SNGas
from network import Network, nucleation_numpy_type

from physical_constants import k_B, stdP, istdP
from simulation_constants import *
from util.timer import Timer


# calculation data for ode
dust_calc = np.dtype(
    [
        ("ks_jdx", np.int32),
        ("cbar", np.float64),
        ("r_nu", np.float64, MAX_REACTANTS),
        ("S", np.float64),
        ("catS", np.float64),
        ("lnS", np.float64),
        ("Js", np.float64),
        ("dadt", np.float64),
        ("ncrit", np.float64),
    ],
    align=True,
)

dust_env = np.dtype(
    [
        ("temperature", np.float64),
        ("d_temperature", np.float64),
        ("kT", np.float64),
        ("rho", np.float64),
        ("d_rho", np.float64),
        ("volume", np.float64),
    ],
    align=True,
)

ode_wksp = np.dtype(
    [
        ("parameters", nucleation_numpy_type),
        ("environment", dust_env),
        ("calculated", dust_calc),
    ],
    align=True,
)

# numba types for offloading
numba_dust_type = from_dtype(nucleation_numpy_type)
numba_dust_calc = from_dtype(dust_calc)
numba_ode_wksp = from_dtype(ode_wksp)


# set up data types for calcuation, used for processing during time-step
@jit(void(numba_dust_calc[:], numba_dust_type[:], double[:], double[:]), **jit_kwargs)
def dust_pre(calc_t, dust_t, y, cb):
    for i in prange(calc_t.size):
        calc_t[i].ks_jdx = np.argmin(y[dust_t[i].keysp_idx[: dust_t[i].nkeysp]])
        calc_t[i].cbar = np.maximum(1.0e-20, cb[dust_t[i].keysp_idx[calc_t[i].ks_jdx]])
        for j in range(dust_t[i].nr):
            calc_t[i].r_nu[j] = (
                dust_t[i].react_nu[j] / dust_t[i].react_nu[calc_t[i].ks_jdx]
            )

        calc_t[i].S = 0.0
        calc_t[i].Js = 0.0
        calc_t[i].dadt = 0.0
        calc_t[i].ncrit = 0.0


# determine quantities necessary for nucleation of a reaction
@jit(void(numba_dust_calc[:], numba_dust_type[:], double[:], double), **jit_kwargs)
def dust_state(calc_t, dust_t, y, T):
    kT = k_B * T

    # hopefully this gets executed in parallel
    for i in range(calc_t.size):
        if dust_t[i].active == 0:
            continue

        # the kj-ndx refers to the "local" position in the list of ks for this reaction
        # the ki-ndx refers to the "global" position in the solution array
        ks_jdx = calc_t[i].ks_jdx
        ks_idx = dust_t[i].keysp_idx[ks_jdx]
        c1 = y[ks_idx]

        nks_idx = dust_t[i].react_idx[dust_t[i].react_idx[: dust_t[i].nr] != ks_idx]
        # holy shit this acutally works
        nks_jdx = np.where(dust_t[i].react_idx[np.arange(0, dust_t[i].nr)] != ks_idx)

        psum = np.sum(np.log(y[nks_idx] * kT * istdP) * calc_t[i].r_nu[nks_jdx])
        delg_reduced = (dust_t[i].A / T - dust_t[i].B) + psum
        calc_t[i].lnS = np.log(c1 * kT * istdP) + delg_reduced
        calc_t[i].catS = (stdP / kT) * np.exp(-delg_reduced)

        w = 1.0 + np.sum(dust_t[i].react_nu[nks_jdx])
        Pii = 1.0 * np.prod(np.power(y[nks_idx] / c1, calc_t[i].r_nu[nks_jdx]))
        # this could be conditioned on "ncrit", but for now let's go with
        # it and see if it holds
        if calc_t[i].lnS > 0.0:
            iw = 1.0 / w
            Pii = np.power(Pii, iw)
            mu = fourpi * dust_t[i].a02 * dust_t[i].sigma / kT

            # if there are numerical issues, they likely arise here.
            # this variable should not grow very much, and if >~100 then
            # something is wack
            expJ = -fourover27 * np.power(mu, 3.0) / (calc_t[i].lnS * calc_t[i].lnS)

            Jkin = np.sqrt(
                2.0 * dust_t[i].sigma / (np.pi * dust_t[i].react_mass[ks_jdx])
            )

            calc_t[i].S = np.exp(calc_t[i].lnS)
            calc_t[i].Js = dust_t[i].omega0 * Jkin * c1 * c1 * Pii * np.exp(expJ)

            calc_t[i].dadt = (
                dust_t[i].omega0
                * np.sqrt(0.5 * kT / (np.pi * dust_t[i].react_mass[ks_jdx]))
                * c1
                * (1.0 - 1.0 / calc_t[i].S)
            )
            calc_t[i].ncrit = np.power(twothird * (mu / calc_t[i].lnS), 3.0) + iw


# calculate the dust moments K_i, 0 <= i < 4
@jit(
    (numba_dust_calc[:], numba_dust_type[:], double[:], double[:], double[:]),
    **jit_kwargs,
)
def dust_moments(calc_t, dust_t, y, cbar, dydt):
    for i in prange(calc_t.size):
        if dust_t[i].active == 0:
            continue
        if calc_t[i].ncrit < 2.0:
            continue
        gidx = dust_t[i].prod_idx[0]
        dydt[gidx] = calc_t[i].Js / calc_t[i].cbar
        for j in range(1, N_MOMENTS):
            jdbl = np.float64(j)
            dydt[gidx + j] = (
                dydt[gidx] * np.power(calc_t[i].ncrit, jdbl / 3.0)
                + (jdbl / dust_t[i].a0) * calc_t[i].dadt * y[gidx + j - 1]
            )

        dydt[dust_t[i].react_idx[: dust_t[i].nr]] -= (
            calc_t[i].cbar * dydt[gidx + 3] * calc_t[i].r_nu[: dust_t[i].nr]
        )


# non steady-state kernels
# may move elsewhere

# use ln (p_{n-1}/p_n) = g_n - g_{n-1}


@vectorize([double(double, double)])
def part_pres(y, T):
    return y * k_B * T


@stencil
def del_gn(p):
    return np.log(p[-1] / p[0])


# γ n = 1/kT ( g_n − g_{n−1} − g_c ) − ln S
# - g_n : chemical potential (at p = p_std) of n-mer
# ---- cap approx: g_n = 4 pi a0^2 sig (n - 1)^{2/3} + (n - 1) g_c + g_1
# ----             g_1 = del g_0 of gas
# - g_c : chemical potential (at p = p_std) of bulk (this is delG_0 ~ -A/T + B used by Nozowa)
#
# simplification with approx:
# γ n = mu ( (n-1)^{2/3} - (n-2)^{2/3} ) − ln S


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


@jit((double, double[:], double[:]), **jit_kwargs)
def expand(xpand, y, dydt):
    for i in prange(y.size):
        dydt[i] += xpand * y[i]


# @guvectorize([(double, double[:], double[:])], '(),(n)->(n)', nopython=S_NOPYTHON)
# def expand(xpand, y, dydt):
#     for i in range(y.size):
#         dydt[i] = xpand * y[i]

# @jit((double[:], double[:], double[:], int32, numba_dust_calc[:]), debug=S_DEBUG, nopython=S_NOPYTHON, parallel=S_PARALLEL, fastmath=S_FASTMATH)
#         dydt[i] = xpand * y[i]


@jit((double[:], double[:], double[:]), **jit_kwargs)
def conc_update(d_conc, dydt, y):
    for i in prange(y.size):
        dydt[i] += d_conc[i]  # + y[i]


# checks if nucleation reaction should be continued
# disables the reaction if any of the key species are exhausted
@jit((numba_dust_type[:], double[:]), **jit_kwargs)
def check_active(dust_t, y):
    for i in prange(dust_t.size):
        if np.any(y[dust_t[i].keysp_idx[: dust_t[i].nkeysp]] < 1.0e-30):
            dust_t[i].active = 0


# dy / dt = f(y, t), function to give to the integrator
# TODO: also provide the jacobian J = df / dy
@jit(
    double[:](
        numba_dust_calc[:], numba_dust_type[:], double[:], double[:], double, double
    ),
    **jit_kwargs,
)
def _f(calc_t, dust_t, y, cb, T, dT):
    dydt = np.zeros(y.size)
    dust_pre(calc_t, dust_t, y, cb)

    check_active(dust_t, y)
    dust_state(calc_t, dust_t, y, T)
    dust_moments(calc_t, dust_t, y, cb, dydt)
    return dydt


@jit(
    (numba_dust_calc[:], numba_dust_type[:], double[:], double[:], double, double),
    **jit_kwargs,
)
def _e_sat(calc_t, dust_t, y, cb, T, dT):
    dust_pre(calc_t, dust_t, y, cb)

    check_active(dust_t, y)
    dust_state(calc_t, dust_t, y, T)


#############################################################
# class Stepper(object):
#     def __init__(self, gas: SNGas, net: Network):
#         self._gas = gas
#         self._net = net
#         self._dust_calc = np.empty(self._net.ND, dtype=dust_calc)
#
#         # TODO: hacking now to finish, come back and fix this garbage
#         _tmp = self._net.generate_nparrays()
#         self._dust_par = _tmp["nucleation"]
#
#         # self._arr_par = _tmp["arrhenius"]
#         self._cbar = np.zeros(self._net.solution_size)
#         self._dust_par[:]["active"] = 1
#
#     def initial_value(self) -> np.array:
#         return self._gas.concentration_0 # try to add bins
#
#
#     def __call__(self, t: np.float64, y: np.array) -> np.array:
#         # get interpolant data
#
#         v_gas = double(self._gas.Velocity(t))
#         T = double(self._gas.Temperature(t)) # this cast is necessary, not sure why just yet
#         dT = double(self._gas.Temperature(t, derivative=1))
#
#
#
#         rho = self._gas.Density(t)
#         drho = self._gas.Density(t, derivative=1)
#
#         # call calculators
#         xpnd = drho / rho
#         vol = self._gas.mass_0 / rho
#         self._cbar[:] = self._gas.cbar(vol)
#
#         dydt = _f(self._dust_calc, self._dust_par, y, self._cbar, T, dT)
#
#         expand(xpnd, y[0:self._net.NG], dydt[0:self._net.NG])
#         #erode(dadt, y[self._net.NG+self._net.ND*N_MOMENTS:], dydt[self._net.NG+self._net.ND*N_MOMENTS:], self._net.NG, self._dust_calc)
#         #erode_grow(dadt, y, dydt, self._net.NG, self._dust_calc)
#
#
#         return dydt
#
#     def emit_done(self):
#         return np.all(self._dust_par["active"] == 0)
#
#
#
