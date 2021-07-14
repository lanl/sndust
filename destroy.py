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
from simulation_constants import dTime, edges, numBins, onehalf, N_MOMENTS
from atomic_mass import AMU
from printer import *

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

def destroy(g: SNGas, net: Network, volume, y, T, vc):
    species = list(net._species_gas) #name of species
    abun_list = np.zeros(len(species)) # concentration
    for idx,val in enumerate(species):
        abun_list[idx] = g._c0[idx]
    n_tot = sum([abun_list[Sidx] * AMU[s.strip()] for Sidx,s in enumerate(species)])
    #prnt(n_tot)
    grain_names = net._species_dust
    dest, g_change = calc_TOTAL_dadt(grain_names,T,n_tot,abun_list,species,vc,g,net,volume,y)
    return dest, g_change # dest is in cm

#will need to pass in an array or dictionary or all the abundances
def calc_TOTAL_dadt(grain_list,T,n,abun,abun_name,vc,g: SNGas,net: Network,volume, y):
    destruct_list = np.zeros(len(grain_list))
    vd = vc / 100000
    si = np.sqrt( (vd ** 2) / (2 * kB_erg * T))
    if si > 10:
        return non_THERMAL_dadt(grain_list,T,n,abun,abun_name,vd,g,net,volume, y)
    else:
        return THERMAL_dadt(grain_list,T,n,abun,abun_name,g,net,volume,y)

#will need to pass in an array or dictionary or all the abundances
def THERMAL_dadt(grain_list,T,n,abun,abun_name,g: SNGas,net: Network,volume, y):
    g_c0_change = np.zeros(len(abun_name))
    destruct_list = np.zeros(len(grain_list)*numBins)
    n_gas = net.NG
    for GRidx,grain in enumerate(grain_list):
        grain = str(grain.replace('(s)',''))
        if grain not in data:
            prnt('skip')
            destruct_list[GRidx] = 0
            continue
        v = data[grain]
        dadt = 0
        for idx,val in enumerate(abun):
            i_abun_name = list(abun_name)[idx]
            pref = val * np.sqrt( 8.0 * kB_erg * T / (np.pi * ions[i_abun_name]["mi"] * amu2g))
            yp = Yield(u0 = v["u0"],md = v["md"],mi = ions[i_abun_name]["mi"],zd = v["zd"],zi = ions[i_abun_name]["zi"],K = v["K"])
            grnComps = grainsCOMP[grain]["react"]
            prod_coef = grainsCOMP[grain]["reacAMT"]
            for cidx,coef in enumerate(prod_coef):
                sidx = net.sidx(grnComps[cidx])
                g_c0_change[sidx] = yp.Y(x)*coef/(volume*np.sum(prod_coef))
                #g._c0[sidx] = g._c0[sidx] + yp.Y(x * kB_eV * T)/(volume*np.sum(prod_coef))*coef
            dadt = dadt + pref * quad(lambda x: x * np.exp(-x) * yp.Y(x * kB_eV * T), a=yp.eth/(kB_eV * T) , b=np.infty)[0]
        dadt = dadt * (v["md"] * amu2g) / (2. * v["rhod"]) * n # in cm/s
        destruct_list[GRidx*numBins:GRidx*numBins+numBins] = dadt
    #prnt(destruct_list)
    return destruct_list, g_c0_change

#will need to pass in an array or dictionary or all the abundances
def non_THERMAL_dadt(grain_list,T,n,abun,abun_name,vd,g: SNGas,net: Network,volume, y):
    prnt('got to nonthermal')
    g_c0_change = np.zeros(len(abun_name))
    destruct_list = np.zeros(len(grain_list)*numBins)
    n_gas = len(net._species_gas)
    for sizeIDX in list(range(numBins)):
        for GRidx,grain in enumerate(grain_list):
            grain = str(grain.replace('(s)',''))
            if grain not in data:
                prnt('skip')
                destruct_list[GRidx] = 0
                continue
            v = data[grain]
            cross_sec = (edges[sizeIDX] + edges[sizeIDX+1]) * onehalf #np.cbrt(y[n_gas +(GRidx*4+0)]/y[n_gas+(GRidx*4+3)]) * v["a0"]# in cm 
            velo = calc_dvdt(abun[0], T, v["rhod"], abun, abun_name, vd, cross_sec, g, net) * dTime
            prnt(velo)
            grain = str(grain.replace('(s)',''))
            dadt = 0
            for idx,val in enumerate(abun):
                i_abun_name = list(abun_name)[idx]
                pref = val
                x = 1./2. * ions[i_abun_name]["mi"] * amu2g / 1000 * np.power(velo /1000,2) * JtoEV
                yp = Yield(u0 = v["u0"],md = v["md"],mi = ions[i_abun_name]["mi"],zd = v["zd"],zi = ions[i_abun_name]["zi"],K = v["K"])
                grnComps = grainsCOMP[grain]["react"]
                prod_coef = grainsCOMP[grain]["reacAMT"]
                for cidx,coef in enumerate(prod_coef):
                    sidx = net.sidx(grnComps[cidx])
                    g_c0_change[sidx] = yp.Y(x)*coef/(volume*np.sum(prod_coef))
                    #g._c0[sidx] = g._c0[sidx] + yp.Y(x)*coef/(volume*np.sum(prod_coef))
                dadt = dadt + pref * yp.Y(x)
                prnt(dadt)
            dadt = dadt * (v["md"] * amu2g * velo) / (2. * v["rhod"]) * n # cm/s
            destruct_list[int(GRidx)*numBins + sizeIDX] = dadt
    prnt(destruct_list)
    return destruct_list, g_c0_change

#def decel(n_tot, n_gas, rho_d, abun, temp, cross_sec, g: SNGas):
#    m_i = g._c0[i]*AMU[abun_name[i]]*amu2g
#    dv/dt = -3*n_tot*kB_erg*T/(2*cross_sec)*np.sum(abum*8.0/(3.8*np.sqrt(np.pi)))*np.sqrt(m_i*v_d**2/(2*kB_erg*T)*np.sqrt(1+9*np.pi/64*m_i*v_d**2/(2*kB_erg*T))

def calc_dvdt(n_h, T, rho, abun, abun_name,velo, a_cross, g: SNGas, net: Network):
    G_tot = np.zeros(len(abun))
    for idx,val in enumerate(abun):
        sidx = net.sidx(abun_name[idx])
        m = g._c0[sidx] * AMU[abun_name[idx]]*amu2g 
        s = m * velo**2 /(2*kB_erg*T)
        G_tot[idx] = 8*s/(3*np.sqrt(np.pi))*(1+9*np.pi*s**2/64)**2
    dvdt = -3*n_h*kB_erg*T/(2*a_cross*rho)*np.sum(abun/n_h*G_tot)
    return dvdt

