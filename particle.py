# read input data.
# this will likely change as other/newer formats are produ
import numpy as np
import os, sys, json, argparse

from itertools import takewhile, repeat
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, NamedTuple

import numpy as np
import h5py as h5
import periodictable as pt

from physical_constants import eV2K, amu2g


@dataclass
class Particle:
    pid: int = -1
    sid: str = "not assigned"
    times: np.array = None
    temperatures: np.array = None
    densities: np.array = None
    position: np.array = None
    mass: np.array = None
    volume: np.array = None
    velocity: np.array = None

    first_idx: int = 0
    last_idx: int = -1

    composition: Dict[str, float] = field(default_factory=dict)


@dataclass
class P_freex:
    pid: int = -1
    radius: float = 0.0
    time_init: float = 0.0
    temp_init: float = 1.0
    rho_init: float = 1.0e-20
    mass: float = 0.0
    volume: float = 0.0

    composition: Dict[str, float] = field(default_factory=dict)


def get_radius_dat(suf: str):
    rdfn = f"./data/wd_crad/rad_{suf}.dat"
    rdat = np.genfromtxt(rdfn, dtype=[("cid", "i8"), ("radius", "f8")])
    return rdat


def get_ele_name(el):
    _t = "".join([char for char in el if char.isalpha()])
    return _t.capitalize()


def transform_comp(els, mfs):
    unique_el = []
    collected_mfs = []

    for el in els:
        xel = get_ele_name(el)
        if xel not in unique_el:
            unique_el.append(xel)
            collected_mfs.append(0.0)

    odic = dict(zip(els, mfs))
    ndic = dict(zip(unique_el, collected_mfs))
    for el, mf in odic.items():
        xcel = get_ele_name(el)
        ndic[xcel] += mf

    return ndic


def load_particles_freex(sdir: str):
    pl = []
    npair = sdir.split("/")[-1]
    rdat = get_radius_dat(npair)
    with os.scandir(sdir) as entries:
        for entry in entries:
            if entry.is_file():
                print(f"File found: {entry.path}")
                with open(entry.path, "r") as tf:
                    rid = entry.path.split("_")[-1][0:4]
                    lines = tf.readlines()
                    datl = []
                    for line in lines[1:]:
                        if line == "\n":
                            continue
                        datl.append(line)

                    elems = lines[0].split()[6:]
                    elfix = []
                    for el in elems:
                        elfix.append(el.split(")")[1])
                for dl in datl:
                    p = P_freex()
                    tok = dl.split()
                    p.pid = int(rid)
                    p.radius = float(rdat[p.pid][1])
                    p.time_init = float(tok[0])
                    p.temp_init = float(tok[1])
                    p.rho_init = float(tok[2])
                    p.mass = float(tok[3])
                    mfs = []
                    for t in tok[6:]:
                        mfs.append(float(t))
                    p.composition = transform_comp(elfix, mfs)
                    pl.append(p)
    pl = sorted(pl, key=lambda p: p.pid)
    return pl

    # with open(starting_file, 'r') as sf:
    #     lines = sf.readlines()
    #     elems = lines[0].split()[6:]
    #     for line in lines[1:]:
    #         p = P_freex()
    #         tok = line.split()
    #         p.pid = int(tok[0])
    #         p.radius = float(tok[1])
    #         p.time_init = float(tok[2])
    #         p.temp_init = float(tok[3])
    #         p.rho_init = float(tok[4])
    #         p.mass = float(tok[5])
    #         p.volume = p.mass / p.rho_init
    #         mfs = []
    #         for t in tok[6:]:
    #             mfs.append(float(t))
    #         p.composition = dict(zip(elems, mfs))
    #
    #         pl.append(p)
    return pl


# TODO clarify and unify the input data
def load_particle(
    compf: str, hydrofn: str, mdl_idx: int, p_idx: int, start_scan: np.float64 = None
) -> Particle:
    p = Particle()

    #    hf = h5.File(h5fn, 'r')

    # mdl = list(hf.keys())[mdl_idx]
    # hfp = hf[mdl]

    cdat = np.genfromtxt(compf, dtype=None, names=True, encoding=None)
    comp = dict(zip(cdat["element"], cdat["mass_fraction"]))

    #    for i, z in enumerate(hfp["initial_z"]):
    #    if hfp["initial_mf"][p_idx, i] > 1.0E-12:
    #        comp.update({str(pt.elements[z]): hfp["initial_nd"][p_idx, i]})

    p.pid = p_idx
    p.sid = f"WDTMP_{p_idx}"

    #    tdat = np.genfromtxt(hydrofn, dtype=None, names=True)

    #    select_idx = np.where( tdat["id"] == p_idx )
    # p.times = tdat["time"][select_idx][:-1]
    # p.temperatures = tdat["temperature"][select_idx][:-1]
    # p.densities = tdat["rho"][select_idx][:-1]
    # p.mass = tdat["mass"][select_idx][:-1]
    # p.position = tdat["xc"][select_idx][:-1]
    # p.velocity = tdat["vc"][select_idx][:-1]
    # p.composition = comp
    # p.volume = p.mass / p.densities
    #
    gamma = 1.6
    NT = 1000
    time0 = 56.250
    temp0 = 8.4e7
    tempN = 650

    rho0 = 1.04e-1
    mas0 = 7.37e28

    timeN = time0 * np.power((tempN / temp0), (1.0 / (3.0 * (1.0 - gamma))))

    ts = np.logspace(np.log10(time0), np.log10(timeN), NT)
    temps = temp0 * np.power((ts / time0), 3.0 * (1.0 - gamma))
    rhos = rho0 * np.power((ts / time0), -3.0)

    p.times = ts
    p.temperatures = temps
    p.densities = rhos
    p.mass = mas0 * np.ones_like(rhos)
    p.position = 1.0 * np.ones_like(rhos)
    p.velocity = 1.0 * np.ones_like(rhos)
    p.composition = comp
    p.volume = p.mass / p.densities

    if start_scan is not None:
        while p.temperatures[p.first_idx] > start_scan:
            if p.first_idx > p.temperatures.size - 1:
                raise ValueError(f"no temperature less than {start_scan} in data")
            p.first_idx += 1

    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help="hdf5 input file")

    args = parser.parse_args()

    ps = load_particles_freex(args.datadir)

    print(f"{ps[0].mass}\n")
    # parser.add_argument("inputfile", type=str, help="hdf5 input file")
    # parser.add_argument("hydrofile", type=str, help="hydro run file")
    # parser.add_argument("mid", type=int, help="mid")
    # parser.add_argument("pid", type=int, help="pid")
    #
    # args = parser.parse_args()
    # p = load_particle( args.inputfile, args.hydrofile, args.mid, args.pid)
    import matplotlib.pyplot as plt

    x = []
    cs = ps[0].composition.keys()
    ys = []

    for c in cs:
        ys.append([])

    for p in ps:
        x.append(p.radius)
        for i, c in enumerate(cs):
            ys[i].append(p.composition[c])

    for i, c in enumerate(cs):
        plt.loglog(x, ys[i], label=c)
    plt.legend()
    plt.show()
    # plt.loglog(p.times, p.densities)
    # plt.show()
    # hf = h5.File(args.inputfile, 'r')

    # mdl = list(hf.keys())[2]
    # hfp = hf[mdl]

    # comp = dict()
    # max_c = -1
    # max_n = 0
    # for n in range(1500):
    #     for i, z in enumerate(hfp["initial_z"]):
    #         if z == 6:
    #             if hfp["initial_nd"][n, i] > max_c:
    #                 max_c = hfp["initial_nd"][n, i]
    #                 max_n = n
    #                 break

    # print(f"max c = {max_c} @ p={max_n}")
