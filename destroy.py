from dataclasses import dataclass, field
from typing import Union

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from sputterINPUT import *
from sputterDict import *
from atomic_mass import *
import numba
from numba import jit
import time
from erosionDict import *
from particle import *
from network import *
from gas import *

onethird = 1./3.
twothird = 2. * onethird
fourpi = 4. * np.pi
#echarge = e.emu.value
echarge = np.sqrt(14.4) # for annoying reasons, use e in units sqrt(eV AA)
bohrr = 0.5291772106699999
kB_eV = 8.617333262145E-5
kB_erg = 1.380649E-16
solZ = 0.012
g2amu = 6.022e+23
amu2g = 1. / g2amu
JtoEV = 6.242e+18

def destroy(g: SNGas, p: Particle, net: Network):
    volume = p.volume
    T = p.temperatures
    vc = p.velocity
    species = list(p.composition.keys())
    abun_list = list(p.composition.values())
    n_tot = sum([abun_list[Sidx] * AMU[s.strip()] for Sidx,s in enumerate(species)])
    grain_names = net._species_dust
    dest = np.zeros((len(T),16))
    print('one call')
    for i in list(range(len(T))):
        dec = calc_TOTAL_dadt(grain_names,T[i],n_tot,abun_list,species,vc[i],p,g,net,volume[i]) / 1E4
        dest[i] = dec
    return dest

#will need to pass in an array or dictionary or all the abundances
def calc_TOTAL_dadt(grain_list,T,n,abun,abun_name,vc,p: Particle,g: SNGas,net: Network,volume):
    destruct_list = np.zeros(len(grain_list))
    vd = vc / 100000
    si = np.sqrt( (vd ** 2) / (2 * kB_erg * T))
    if si > 10:
        return non_THERMAL_dadt(grain_list,T,n,abun,abun_name,vd,p,g,net,volume)
    else:
        return THERMAL_dadt(grain_list,T,n,abun,abun_name,p,g,net,volume)

#will need to pass in an array or dictionary or all the abundances
def THERMAL_dadt(grain_list,T,n,abun,abun_name,p: Particle,g: SNGas,net: Network,volume):
    destruct_list = np.zeros(len(grain_list))
    for GRidx,grain in enumerate(grain_list):
        grain = str(grain.replace('(s)',''))
        if grain not in data:
            destruct_list[GRidx] = 0
            continue
        v = data[grain]
        dadt = 0
        for idx,val in enumerate(abun):
            i_abun_name = list(abun_name)[idx]
            pref = val * np.sqrt( 8.0 * kB_erg * T / (np.pi * ions[i_abun_name]["mi"] * amu2g))
            ## these two lines take forever
            start = time.time()
            yp = Yield(u0 = v["u0"],md = v["md"],mi = ions[i_abun_name]["mi"],zd = v["zd"],zi = ions[i_abun_name]["zi"],K = v["K"])
            sidx = net.sidx(grain)
            g._c0[sidx] = g._c0[sidx] + yp.Y(x * kB_eV * T)/volume
            dadt += pref * quad(lambda x: x * np.exp(-x) * yp.Y(x * kB_eV * T), a=yp.eth/(kB_eV * T) , b=np.infty)[0]
        dadt *= (v["md"] * amu2g) / (2. * v["rhod"]) * n
        destruct_list[GRidx] = dadt
    return destruct_list

#will need to pass in an array or dictionary or all the abundances
def non_THERMAL_dadt(grain_list,T,n,abun,abun_name,vd,p: Particle,g: SNGas,net: Network,volume):
    destruct_list = np.zeros(len(grain_list))
    for GRidx,grain in enumerate(grain_list):
        grain = str(grain.replace('(s)',''))
        if grain not in data:
            destruct_list[GRidx] = 0
            continue
        v = data[grain]
        dadt = 0
        for idx,val in enumerate(abun):
            i_abun_name = list(abun_name)[idx]
            pref = val
            x = 1./2. * ions[i_abun_name]["mi"] * amu2g / 1000 * np.power(vd / 1000,2) * JtoEV
            yp = Yield(u0 = v["u0"],md = v["md"],mi = ions[i_abun_name]["mi"],zd = v["zd"],zi = ions[i_abun_name]["zi"],K = v["K"])
            sidx = net.sidx(grain)
            g._c0[sidx] = g._c0[sidx] + yp.Y(x)/volume
            dadt += pref * yp.Y(x)
        dadt *= (v["md"] * amu2g * vd) / (2. * v["rhod"]) * n
        destruct_list[int(GRidx)] = dadt
    return destruct_list

