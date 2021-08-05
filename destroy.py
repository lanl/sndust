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
from simulation_constants import edges, numBins, onehalf, N_MOMENTS
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

# here we're establishing the abundance (gas concentrations) and their names in vector form and thenstarting the sputtering/destruct. calculations
def destroy(g: SNGas, net: Network, volume, y, T, v_gas,dTime, dydt):
    gas_name = list(net._species_gas) # name of species
    gas_conc = np.zeros(len(gas_name)) # concentration, units # of particles/cm^3
    for idx,val in enumerate(gas_name): # going and updating the empty list with the gas concentrations 
        gas_conc[idx] = g._c0[idx]
    n_tot = sum([gas_conc[Sidx] * AMU[s.strip()] for Sidx,s in enumerate(gas_name)]) # total number density of the gas
    grain_names = net._species_dust
    dest, g_change, new_dydt = calc_TOTAL_dadt(grain_names,T,n_tot,gas_conc,gas_name,v_gas,g,net,volume,y,dTime, dydt)
    return dest, g_change, new_dydt # dest is in cm

# here, we take the gas velocity and calculate s_i to determine if we have thermal or non thermal sputtering. the cutoff is assigned to s_i = 10
def calc_TOTAL_dadt(grain_list,T,n_tot,gas_conc,gas_name,v_gas,g: SNGas,net: Network,volume, y, dTime, dydt):
    sz_idx0 = net._NG + net._ND * N_MOMENTS
    v_d = y[sz_idx0 + net._ND * numBins: sz_idx0 + net._ND * numBins * 2]
    destruct_list = np.zeros(len(grain_list)*numBins)
    g_c0_change = np.zeros(len(gas_name))

    for sizeIDX in list(range(numBins)):
        for GRidx,grain in enumerate(grain_list):
            #if we have no grains, skip
            if y[sz_idx0 + GRidx * numBins + sizeIDX] <= 0:
                continue
            #if we don't have data for the grain, skip
            grain = str(grain.replace('(s)',''))
            if grain not in data:
                continue
            v = data[grain]
            m_i = ions['C']["mi"] * amu2g
            si = np.sqrt(m_i * (v_d[GRidx*numBins + sizeIDX] ** 2) / (2 * kB_erg * T))
            if si > 1:
                dest, del_g = non_THERMAL_dadt(GRidx, grain, sizeIDX, T, n_tot, gas_conc, gas_name, v_d[GRidx*numBins + sizeIDX], net, volume, y,dTime)
            else:
                dest, del_g = THERMAL_dadt(grain,T,n_tot,gas_conc,gas_name,net,volume,y)
            destruct_list[GRidx * numBins + sizeIDX] = dest
            g_c0_change += del_g
            cross_sec = (edges[sizeIDX] + edges[sizeIDX+1]) * onehalf  # units of cm
            dydt[sz_idx0 + net.ND * numBins + GRidx * numBins + sizeIDX] = calc_dvdt(n_tot,T, v["rhod"], gas_conc, gas_name, v_d[GRidx*numBins+sizeIDX], cross_sec)
    return destruct_list, g_c0_change, dydt

def THERMAL_dadt(grain,T,n_tot,gas_conc,gas_name,net: Network,volume, y):
    g_c0_change = np.zeros(len(gas_name))
    v = data[grain]
    dadt = 0
    m_sp = 0
    idx = net.sidx('C')
    val = gas_conc[idx]
    # here this is the argument in the summation
    i_gas_name = list(gas_name)[idx]
    A_i = val / n_tot # #/cm^3 * cm^3 = # of particles -- unitless
    m_i = ions[i_gas_name]["mi"] * amu2g # mass of gas species-not sure if it should be mulitplied by the abundance, in grams
    # pref is in cgs --  cm/s
    pref = A_i * np.sqrt( 8.0 * kB_erg * T / (np.pi * m_i)) # units in cm/s
    yp = Yield(u0 = v["u0"],md = v["md"],mi = ions[i_gas_name]["mi"],zd = v["zd"],zi = ions[i_gas_name]["zi"],K = v["K"])
    integral = quad(lambda x: x * np.exp(-x) * yp.Y(x * kB_eV * T), a=yp.eth/(kB_eV * T) , b=np.infty)[0]
    #sput_yield = quad(lambda x: yp.Y(x * kB_eV * T), a=yp.eth/(kB_eV * T) , b=np.infty)[0]
    grnComps = grainsCOMP[grain]["reactants"]
    prod_coef = grainsCOMP[grain]["reactants_amount"]
    for cidx,coef in enumerate(prod_coef):
        sidx = net.sidx(grnComps[cidx])
        # calculate change in conecntrations = sputtered amount * coefficant / (volume * sum of coef)
        # number of sputtered atoms, multiplied by num_grn_dest to account for the number of this size bins effected
        g_c0_change[sidx] = integral*coef/(volume*np.sum(prod_coef)) # num_grains_dest
        # adding the mass of the sputtered species to m_sp
        m_sp = m_sp + coef*integral/(np.sum(prod_coef)) * AMU[grnComps[cidx]] * amu2g # now it is in grams
    # dadt is in cm/s
    dadt = dadt + pref * integral
    # now it is in
    dadt = dadt * (-1) * m_sp / (2. * v["rhod"]) * n_tot # in cm/s, the last part is the coeff outside the sum and is unitless
    return dadt, g_c0_change

def non_THERMAL_dadt(GRidx, grain, sizeIDX,T,n_gas,gas_conc,gas_name,v_d,net: Network,volume, y,dTime):
    g_c0_change = np.zeros(len(gas_name))
    v = data[grain]
    cross_sec = (edges[sizeIDX] + edges[sizeIDX+1]) * onehalf  # units of cm
    grain = str(grain.replace('(s)',''))
    dadt = 0
    m_sp = 0
    idx = net.sidx('C')
    val = gas_conc[idx]
    i_gas_name = list(gas_name)[idx]
    A_i = val / n_gas # units of # of particles
    x = 1./2. * ions[i_gas_name]["mi"] * amu2g / 1000 * np.power(v_d /1000,2) * JtoEV # divide by 1000 to get mass,velocity to mks
    yp = Yield(u0 = v["u0"],md = v["md"],mi = ions[i_gas_name]["mi"],zd = v["zd"],zi = ions[i_gas_name]["zi"],K = v["K"])
    grnComps = grainsCOMP[grain]["reactants"]
    prod_coef = grainsCOMP[grain]["reactants_amount"]
    for cidx,coef in enumerate(prod_coef):
        sidx = net.sidx(grnComps[cidx])
        g_c0_change[sidx] = g_c0_change[sidx] + yp.Y(x)*coef/(volume*np.sum(prod_coef)) # units of # of atoms
        m_sp = m_sp + coef*yp.Y(x)/(np.sum(prod_coef)) * AMU[grnComps[cidx]] * amu2g # grams
    dadt = dadt + A_i * yp.Y(x) # units of # of atoms -- unitless
    dadt = dadt * (m_sp * v_d) / (2. * v["rhod"]) * n_gas # cm/s
    return -1*np.abs(dadt), g_c0_change


def calc_dvdt(n_gas, T, rho_d, gas_conc, gas_name,v_d, a_cross):
    G_tot = np.zeros(len(gas_conc))
    A = gas_conc / n_gas #unitless
    for idx,val in enumerate(A):
        m_i = AMU[gas_name[idx]]*amu2g # m_i in grams
        s = np.sqrt(m_i * v_d**2 /(2*kB_erg*T)) # s_i is unitless
        G_tot[idx] = 8*s/(3*np.sqrt(np.pi))*np.sqrt((1+9*np.pi*s**2/64)) # unitless
    dvdt = -3*n_gas*kB_erg*T/(2*a_cross*rho_d)*np.sum(A*G_tot) # units of cm/s^2
    return dvdt

