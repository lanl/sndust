import sys, os, h5py, argparse
from collections import namedtuple
from typing import NamedTuple
import periodictable as pt
import numpy as np
import matplotlib.pyplot as plt

# some other useful constants
amu         = 1.6605402E-24 # g
avo_N	    = 6.02E23 # avogadro
Tev2K	    = 1.160E4 # ev->K
sinyear	    = 3.154E7 # secs in a year
rad_const   = 7.5646E-15 # rad, in cgs
sb_const    = 5.6704E-5 # S-B const
kB_const    = 1.380658E-16 # boltzmann, cgs
solar_mass  = 1.988E33 # solar masses in g

onethird = 1.0 / 3.0
fourpi	= 4.0 * np.pi
fourpiover3 = fourpi / 3.0

class CCSNInfo(NamedTuple):
    label: str
    mass_progenitor: np.float64
    energy_explostion: np.float64
    mass_bounce: np.float64
    mass_injected: np.float64
    temperature_injected: np.float64
    energy_injected: np.float64
    mass_remnant: np.float64
    model_id: np.int32

class DataInfo(NamedTuple):
    label: str
    units: str
    dtype: np.dtype = np.float64
    vector_data: bool = True
    grid_data: bool = True
    edge_value: bool = False
    scale: np.float64 = 1.0

metadata = [ DataInfo(label="PNS mass",     units="g",                  vector_data=False,      grid_data=False,                        scale=solar_mass),
             DataInfo(label="time",         units="s",                  vector_data=False,      grid_data=False),
             DataInfo(label="radius",       units="cm",                                                             edge_value=True),
             DataInfo(label="vel",          units="cm/s",                                                           edge_value=True),
             DataInfo(label="Pgas",         units="g/(cm.s2)"),
             DataInfo(label="Prad",         units="g/(cm.s2)"),
             DataInfo(label="aiso",         units=None,     dtype=np.int32,                     grid_data=False),
             DataInfo(label="ziso",         units=None,     dtype=np.int32,                     grid_data=False),
             DataInfo(label="xiso",         units=None),
             DataInfo(label="ye",           units=None),
             DataInfo(label="xel",          units=None),
             DataInfo(label="zel",          units=None,     dtype=np.int32,                     grid_data=False),
#             DataInfo(label="iso_names",    units=None,     dtype=str,  vector_data=True,       grid_data=False),
             DataInfo(label="egas",         units="erg/cm3"),
             DataInfo(label="erad",         units="erg/g"),
             DataInfo(label="density",      units="g/cm3"),
             DataInfo(label="mass",         units="g"),
             DataInfo(label="temp",         units="K")
            ]


def put_initial_composition(ogrp, ngrp):

    prec = np.float128
    N    = ogrp['xiso'].shape[0]
    Niso = ogrp['xiso'].shape[1]

    rho0 = np.asarray(ogrp['density'], dtype=prec)[0]
    xiso = np.asarray(ogrp['xiso'], dtype=prec)
    aiso = np.asarray(ogrp['aiso'], dtype=prec)
    ziso = np.asarray(ogrp['ziso'], dtype=prec)

    z0 = int(np.amin(ziso))
    zN = int(np.amax(ziso))

    nden = np.zeros((N, zN+1))
    mfrc = np.zeros((N, zN+1))

    for z in range(z0, zN+1):
        zidx = np.where( ziso == z )[0]
        xaf =  np.sum(xiso[:, zidx] / aiso[zidx], axis=1)

        aavg = np.sum(aiso[zidx] * xiso[:, zidx], axis=1) / (np.sum(xiso[:, zidx], axis=1))
        mfrc[:, z] = xaf * aavg
        nden[:, z] = (mfrc[:, z] / (aavg * amu)) * rho0

    mfrc[np.isnan(mfrc)] = 0.0
    nden[np.isnan(nden)] = 0.0

    ngrp.create_dataset("initial_z", data=np.arange(z0, zN+1), dtype=np.float64)
    ngrp.create_dataset("initial_mf", data=mfrc, dtype=np.float64)
    ngrp.create_dataset("initial_nd", data=nden, dtype=np.float64)


def populate_ccsn_group(h5gi, ccsn_info, rootg):

    ngrp = rootg.create_group(f"{ccsn_info.model_id:03d}")
    for md in metadata:
        if md.grid_data:
            if md.edge_value:
                dat = h5gi[md.label]
            else:
                dat = 0.5 * (h5gi[md.label][1:] + h5gi[md.label][:-1])
        else:
            if md.vector_data:
                dat=h5gi[md.label]
            else:
                dat=[h5gi[(md.label)].value]
        dat *= np.asarray(md.scale)
        ds = ngrp.create_dataset(md.label, data=dat, dtype=md.dtype)
        ds.attrs.create("units", np.string_(md.units))

    put_initial_composition(h5gi, ngrp)

def get_model(mid, mdat):
    for modl in mdat:
        if modl["run"]==mid:
            return modl
    return None

def main(ccsn_file, modfile, outputfile):

    hfile = h5py.File(ccsn_file, 'r')

    root_grp = h5py.File(outputfile, 'w')

    moddat = np.genfromtxt(modfile, names=True, dtype=None, delimiter=",", encoding=None)
    _added = 0
    for hfk in hfile.keys():
        modl = get_model(hfk[:-4], moddat)
        if modl is None: continue

        snm = CCSNInfo(label                = modl["run"],
                       mass_progenitor      = modl["mprog"],
                       mass_bounce          = modl["mbounce"],
                       mass_injected        = modl["minj"],
                       temperature_injected = modl["tinj"],
                       energy_injected      = modl["Einj"],
                       energy_explostion    = modl["Eexp"],
                       mass_remnant         = modl["mrem"],
                       model_id            = modl["num"])


        dset = hfile[hfk]
        ngrp = populate_ccsn_group(hfile[hfk], snm, root_grp)

        print(f"{snm.label} moved")
        _added += 1
        # NOTE: keeping filesize down for development, also missing hydro data
        if _added > 2:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile", type=str, help="hdf5 input file")
    parser.add_argument("modelfile", type=str, help="model info")
    parser.add_argument("-o", "--outputfile", type=str, help="hdf5 outputfile")
    args = parser.parse_args()
    outf = "ccsn_data.hdf5"
    if args.outputfile is not None:
        outf = args.outputfile

    main(args.inputfile, args.modelfile, outf)
