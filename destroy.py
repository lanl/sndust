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
def destroy(g: SNGas, net: Network, volume, y, T, v_gas,dTime):
    gas_name = list(net._species_gas) # name of species
    gas_conc = np.zeros(len(gas_name)) # concentration, units # of particles/cm^3
    for idx,val in enumerate(gas_name): # going and updating the empty list with the gas concentrations 
        gas_conc[idx] = g._c0[idx]
    n_tot = sum([gas_conc[Sidx] * AMU[s.strip()] for Sidx,s in enumerate(gas_name)]) # total number density of the gas
    grain_names = net._species_dust
    dest, g_change = calc_TOTAL_dadt(grain_names,T,n_tot,gas_conc,gas_name,v_gas,g,net,volume,y,dTime)
    return dest, g_change # dest is in cm

# here, we take the gas velocity and calculate s_i to determine if we have thermal or non thermal sputtering. the cutoff is assigned to s_i = 10
def calc_TOTAL_dadt(grain_list,T,n_tot,gas_conc,gas_name,v_gas,g: SNGas,net: Network,volume, y, dTime):
    # here v_gas is already in cgs, I converted it to make the input data which is in cgs units
    si = np.sqrt( (v_gas ** 2) / (2 * kB_erg * T))
    if si > 10:
        return non_THERMAL_dadt(grain_list,T,n_tot,gas_conc,gas_name,v_gas,g,net,volume, y,dTime)
    else:
        return THERMAL_dadt(grain_list,T,n_tot,gas_conc,gas_name,g,net,volume,y)

# biscaro et. al 2016 equation 14
def THERMAL_dadt(grain_list,T,n_tot,gas_conc,gas_name,g: SNGas,net: Network,volume, y):
    g_c0_change = np.zeros(len(gas_name))
    destruct_list = np.zeros(len(grain_list)*numBins)
    for GRidx,grain in enumerate(grain_list):
        sizeBins = y[len(gas_conc)+N_MOMENTS*net._ND+GRidx*numBins:len(gas_conc)+N_MOMENTS*net._ND+GRidx*numBins+numBins]
        num_grains_dest = np.count_nonzero(sizeBins)
        if sizeBins == np.zeros(numBins):
            prnt('no grain to destroy, skip')
            continue
        grain = str(grain.replace('(s)',''))
        if grain not in data:
            continue
        v = data[grain]
        dadt = 0
        m_sp = 0
        # here this is the argument in the summation
        for idx,val in enumerate(gas_conc):
            i_gas_name = list(gas_name)[idx]
            A_i = val * volume # #/cm^3 * cm^3 = # of particles -- unitless
            m_i = A_i * ions[i_gas_name]["mi"] * amu2g # mass of gas species-not sure if it should be mulitplied by the abundance, in grams
            # pref is in cgs --  cm/s
            pref = A_i * np.sqrt( 8.0 * kB_erg * T / (np.pi * m_i)) # units in cm/s
            yp = Yield(u0 = v["u0"],md = v["md"],mi = ions[i_gas_name]["mi"],zd = v["zd"],zi = ions[i_gas_name]["zi"],K = v["K"])
            integral = quad(lambda x: x * np.exp(-x) * yp.Y(x * kB_eV * T), a=yp.eth/(kB_eV * T) , b=np.infty)[0]
            sput_yield = quad(lambda x: yp.Y(x * kB_eV * T), a=yp.eth/(kB_eV * T) , b=np.infty)[0]
            grnComps = grainsCOMP[grain]["reactants"] 
            prod_coef = grainsCOMP[grain]["reactants_amount"]
            for cidx,coef in enumerate(prod_coef):
                sidx = net.sidx(grnComps[cidx])
                # calculate change in conecntrations = sputtered amount * coefficant / (volume * sum of coef)
                g_c0_change[sidx] =  sput_yield*coef/(volume*np.sum(prod_coef)) * num_grains_dest # number of sputtered atoms, multiplied by num_grn_dest to account for the number of this size bins effected
                # adding the mass of the sputtered species to m_sp
                m_sp = m_sp + coef*sput_yield/(np.sum(prod_coef)) * AMU[grnComps[cidx]] * amu2g # now it is in grams
            # *confused screeching* I think the integral should be unitless -- in biscaro, the integrand is unitless
            # dadt is in cm/s
            dadt = dadt + pref * integral
        # now it is in 
        dadt = dadt * (-1) * m_sp / (2. * v["rhod"]) * n_tot # in cm/s, the last part is the coeff outside the sum and is unitless
        destruct_list[GRidx*numBins:GRidx*numBins+numBins] = dadt
        empty = np.where(sizeBins == 0)
        for i in empty:
            destruct_list[GRidx*numBins + i] = 0
    return destruct_list, g_c0_change

#will need to pass in an array or dictionary or all the abundances
def non_THERMAL_dadt(grain_list,T,n_gas,gas_conc,gas_name,v_gas,g: SNGas,net: Network,volume, y,dTime):
    g_c0_change = np.zeros(len(gas_name))
    destruct_list = np.zeros(len(grain_list)*numBins)
    # we need to look at each size bin b.c. dv/dt depends on the grain cross section
    for sizeIDX in list(range(numBins)):
        for GRidx,grain in enumerate(grain_list):
            if y[len(gas_conc)+N_MOMENTS*net._ND+GRidx*numBins+sizeIDX] == 0:
                #prnt('no grain to destroy, skip')
                continue
            grain = str(grain.replace('(s)',''))
            if grain not in data:
                continue
            v = data[grain]
            cross_sec = (edges[sizeIDX] + edges[sizeIDX+1]) * onehalf  # units of cm
            v_d = calc_dvdt(n_gas,T, v["rhod"], gas_conc, gas_name, v_gas, cross_sec, g, net, volume) * dTime # units of cm/s
            grain = str(grain.replace('(s)',''))
            dadt = 0
            m_sp = 0
            for idx,val in enumerate(gas_conc):
                i_gas_name = list(gas_name)[idx]
                A_i = val * volume # units of # of particles
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
            destruct_list[int(GRidx)*numBins + sizeIDX] = dadt
    return destruct_list, g_c0_change

def calc_dvdt(n_gas, T, rho_d, gas_conc, gas_name,v_gas, a_cross, g: SNGas, net: Network, volume):
    G_tot = np.zeros(len(gas_conc))
    A = gas_conc * volume #unitless
    for idx,val in enumerate(A):
        m_i = val * AMU[gas_name[idx]]*amu2g # m_i in grams, not sure if mass of the gas species (1 molecule) or the total mass of the gas species
        s = np.sqrt(m_i * v_gas**2 /(2*kB_erg*T)) # s_i is unitless
        G_tot[idx] = 8*s/(3*np.sqrt(np.pi))*np.sqrt((1+9*np.pi*s**2/64)) # unitless
    dvdt = -3*n_gas*kB_erg*T/(2*a_cross*rho_d)*np.sum(A*G_tot) # units of cm/s^2
    return dvdt

