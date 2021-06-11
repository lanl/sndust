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
from erosionDict import grainsCOMP
from network import *
from gas import *
from simulation_constants import dTime
from atomic_mass import AMU

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

def destroy(g: SNGas, net: Network, volume, rho, y, T, vc):
    species = list(net._species_gas) #name of species
    abun_list = np.zeros(len(species)) # concentration
    for idx,val in enumerate(species):
        abun_list[idx] = g._c0[idx]
    n_tot = sum([abun_list[Sidx] * AMU[s.strip()] for Sidx,s in enumerate(species)])
    grain_names = net._species_dust
    
    dest, g_change = calc_TOTAL_dadt(grain_names,T,n_tot,abun_list,species,vc,g,net,volume,rho,y)

    return dest, g_change # dest is in cm

#will need to pass in an array or dictionary or all the abundances
def calc_TOTAL_dadt(grain_list,T,n,abun,abun_name,vc,g: SNGas,net: Network,volume,rho, y):
    destruct_list = np.zeros(len(grain_list))
    vd = vc / 100000
    si = np.sqrt( (vd ** 2) / (2 * kB_erg * T))
    if si > 10:
        return non_THERMAL_dadt(grain_list,T,n,abun,abun_name,vd,g,net,volume,rho, y)
    else:
        return THERMAL_dadt(grain_list,T,n,abun,abun_name,g,net,volume,y)

#will need to pass in an array or dictionary or all the abundances
def THERMAL_dadt(grain_list,T,n,abun,abun_name,g: SNGas,net: Network,volume, y):
    g_c0_change = np.zeros(len(abun_name))
    destruct_list = np.zeros(len(grain_list))
    n_gas = net._species_gas
    for GRidx,grain in enumerate(grain_list):
        if y[n_gas +(GRidx*4+0)] == 0:
            continue
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
            grnComps = grainsCOMP[grain]["react"]
            prod_coef = grainsCOMP[grain]["reacAMT"]
            for cidx,coef in enumerate(prod_coef):
                sidx = net.sidx(grnComps[cidx])
                g_c0_change = yp.Y(x)*coef/(volume*np.sum(prod_coef))
                #g._c0[sidx] = g._c0[sidx] + yp.Y(x * kB_eV * T)/(volume*np.sum(prod_coef))*coef
            dadt += pref * quad(lambda x: x * np.exp(-x) * yp.Y(x * kB_eV * T), a=yp.eth/(kB_eV * T) , b=np.infty)[0]
        dadt *= (v["md"] * amu2g) / (2. * v["rhod"]) * n # in cm/s
        destruct_list[GRidx] = dadt
    return destruct_list, g_c0_change

#will need to pass in an array or dictionary or all the abundances
def non_THERMAL_dadt(grain_list,T,n,abun,abun_name,vd,g: SNGas,net: Network,volume,rho, y):
    g_c0_change = np.zeros(len(abun_name))
    destruct_list = np.zeros(len(grain_list))
    n_gas = len(net._species_gas)
    for GRidx,grain in enumerate(grain_list):
        if y[n_gas +(GRidx*4+0)] == 0:
            continue
        if grain not in data:
            destruct_list[GRidx] = 0
            continue
        v = data[grain]
        cross_sec = np.cbrt(y[n_gas +(GRidx*4+0)]/y[n_gas+(GRidx*4+3)]) * v["a0"]# in cm 
        velo = calc_dvdt(abun[0], T, rho, abun, abun_name, vd, cross_sec, g, net) * dTime
        grain = str(grain.replace('(s)',''))
        dadt = 0
        for idx,val in enumerate(abun):
            i_abun_name = list(abun_name)[idx]
            pref = val
            x = 1./2. * ions[i_abun_name]["mi"] * amu2g / 1000 * np.power(vd / 1000,2) * JtoEV
            yp = Yield(u0 = v["u0"],md = v["md"],mi = ions[i_abun_name]["mi"],zd = v["zd"],zi = ions[i_abun_name]["zi"],K = v["K"])
            grnComps = grainsCOMP[grain]["react"]
            prod_coef = grainsCOMP[grain]["reacAMT"]
            for cidx,coef in enumerate(prod_coef):
                sidx = net.sidx(grnComps[cidx])
                g_c0_change = yp.Y(x)*coef/(volume*np.sum(prod_coef))
                #g._c0[sidx] = g._c0[sidx] + yp.Y(x)*coef/(volume*np.sum(prod_coef))
            dadt += pref * yp.Y(x)
        dadt *= (v["md"] * amu2g * velo) / (2. * v["rhod"]) * n # cm/s
        destruct_list[int(GRidx)] = dadt
    return destruct_list, g_c0_change

def calc_dvdt(n_h, T, rho, abun, abun_name,velo, a_cross, g: SNGas, net: Network):
    G_tot = np.zeros(len(abun))
    for idx,val in enumerate(abun):
        sidx = net.sidx(abun_name[idx])
        m = g._c0[sidx] * AMU[abun_name[idx]] 
        s = m * velo**2 /(2*kB_erg*T)
        G_tot[idx] = 8*s/(3*np.sqrt(np.pi))*(1+9*np.pi*s**2/64)**2
    dvdt = -3*kB_erg*T/(2*a_cross*rho)*np.sum(abun*G_tot)
    return dvdt

