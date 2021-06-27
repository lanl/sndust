# read input data.
# this will likely change as other/newer formats are produ
import numpy as np
import os, sys, json, argparse

from itertools import takewhile,repeat
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

# TODO clarify and unify the input data
def load_particle( h5fn: str, hydrofn: str, mdl_idx:int, p_idx: int, start_scan: np.float64=None) -> Particle:
    p = Particle()

    hf = h5.File(h5fn, 'r')

    mdl = list(hf.keys())[mdl_idx]
    hfp = hf[mdl]

    comp = dict()
    for i, z in enumerate(hfp["initial_z"]):
        if hfp["initial_mf"][p_idx, i] > 1.0E-12:
            comp.update({str(pt.elements[z]): hfp["initial_nd"][p_idx, i]})

    p.pid = p_idx
    p.sid = f"{str(mdl)}_{p_idx}"

    tdat = np.genfromtxt(hydrofn, dtype=None, names=True)

    select_idx = np.where( tdat["id"] == p_idx )
    p.times = tdat["time"][select_idx][:-1]
    p.temperatures = tdat["temperature"][select_idx][:-1]
    p.densities = tdat["rho"][select_idx][:-1]
    p.mass = tdat["mass"][select_idx][:-1]
    p.position = tdat["xc"][select_idx][:-1]
    p.velocity = tdat["vc"][select_idx][:-1]
    p.composition = comp

    if start_scan is not None:
        while p.temperatures[p.first_idx] > start_scan:
            if p.first_idx > p.temperatures.size - 1:
                raise ValueError(f"no temperature less than {start_scan} in data")
            p.first_idx += 1

    p.volume = p.mass / p.densities[p.first_idx]

    return p

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile", type=str, help="hdf5 input file")
    parser.add_argument("hydrofile", type=str, help="hydro run file")
    parser.add_argument("mid", type=int, help="mid")
    parser.add_argument("pid", type=int, help="pid")

    args = parser.parse_args()
    p = load_particle( args.inputfile, args.hydrofile, args.mid, args.pid)
    import matplotlib.pyplot as plt

    plt.loglog(p.times, p.densities)
    plt.show()
