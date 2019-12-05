import os, sys, logging, itertools
from typing import List, Tuple, NamedTuple

import numpy as np

from util.molmass import Formula

from gas import SNGas
from physical_constants import amu2g, ang2cm, k_B, stdP, istdP
from simulation_constants import MAX_REACTANTS, MIN_CONCENTRATION, twopi, fourpi, fourpiover3, fourover27, twothird

# contiguous structure of dust data.
dust_type = np.dtype([
                ("keysp_idx", np.int32),
                ("keysp_mass", np.float64),
                ("keysp_nu", np.float64),
                ("NR", np.int32),
                ("react_idx", np.int32, (MAX_REACTANTS,)),
                ("react_mass", np.float64, (MAX_REACTANTS,)),
                ("react_nu", np.float64, (MAX_REACTANTS,)),
                ("A", np.float64),
                ("B", np.float64),
                ("sigma", np.float64),
                ("a0", np.float64),
                ("a02", np.float64),
                ("a03", np.float64),
                ("fourpia02", np.float64),
                ("omega0", np.float64),
                ("twopim", np.float64),
                ("itwopim", np.float64)
            ], align=True)

# calculation data for ode
dust_calc = np.dtype([
                ("S", np.float64),
                ("Js", np.float64),
                ("dadt", np.float64),
                ("ncrit", np.float64)
            ], align=True)

# contains non-numeric dust data, for io and such
class dust_meta(NamedTuple):
    name: str
    formula: str
    keysp: str
    rctsp: List[str] = []

class DustLoader(object):
    """
    DustLoader reads in the data which is consumed by the Nucleator.
    There's a lot of half-realized pythonisms here; it's mostly held together by
    twigs and desperation. Feel free to modify
    """
    def __init__(self, dust_filename: str):
        self._dustf = dust_filename
        self._stor = list()
        self._uniq_sp = []

    def load(self) -> None:
        no_phs = lambda s: s.replace("(g)","").replace("(s)","")
        decomp = lambda s: [1.0, s] if not s[0].isdigit() else [np.float64(s[0]), s[1:]]

        dataset = np.genfromtxt(self._dustf, dtype=None, comments='#', skip_header=0, names=True, autostrip=True, encoding=None)
        for row in dataset:
            keysp = [no_phs(r) for r in row["key_species"].split(",")]
            rxnsp = [no_phs(r) for r in row["chemical_reactions"].split("->")[0].split('+')]
            drxnp = [decomp(r) for r in rxnsp]
            dkeyp = [decomp(r) for r in keysp]
            self._stor.append( [ row["grains"], f"{' + '.join(rxnsp)} = {row['grains']}", dkeyp, drxnp, row["A"], row["B"], row["sigma"], row["a0"] ] )

            [self._uniq_sp.append(s[1]) for s in drxnp if s[1] not in self._uniq_sp]

    @property
    def n_grains(self):
        return len(self._stor)

    @property
    def uniq_species(self):
        return self._uniq_sp

    def __call__(self, gas: SNGas) -> Tuple[dust_meta, np.array]:

        mmass  = lambda sp: amu2g * Formula(sp).mass

        for row in self._stor:

            ksidx, key_species = gas.minimum_species(row[2])

            dm = dust_meta(row[0], row[1], key_species[1])

            dt = np.array(0, dtype=dust_type)

            dt["keysp_idx"] = ksidx
            dt["keysp_mass"] = mmass(key_species[1])
            dt["keysp_nu"] = key_species[0]

            reactants = row[3]
            dt["NR"] = len(reactants)
            for i in range(dt["NR"]):
                dt["react_idx"][i] = gas.species_idx(reactants[i][1])
                dt["react_mass"][i] = mmass(reactants[i][1])
                dt["react_nu"][i] = reactants[i][0] / dt["keysp_nu"]
                dm.rctsp.append(reactants[i][1])

            dt["A"] = 1.0E4  * row[4]
            dt["B"] = row[5]
            dt["sigma"] = row[6]
            dt["a0"] = ang2cm * row[7]

            dt["a02"] = dt["a0"]**2
            dt["a03"] = dt["a0"]**3
            dt["fourpia02"] = dt["a02"] * fourpi
            dt["omega0"] = dt["a03"] * fourpiover3
            dt["twopim"] = dt["keysp_mass"] * twopi
            dt["itwopim"] = 1.0 / dt["twopim"]

            yield dm, dt

        yield None
        return

class Nucleator(object):
    """
    Nucleator holds the dust arrays and provides an interface to access
    them. This data is ideally read-only, and there is some effort to expose
    things in that context, but some laziness is also here.
    """
    def __init__(self, loader: DustLoader, gas: SNGas):
        self._dust = np.empty(loader.n_grains, dtype=dust_type)
        self._metadust = list()
        self._consume_loader(loader, gas)

    def _consume_loader(self, loader: DustLoader, gas: SNGas) -> None:
        for i, d in enumerate(loader(gas)):
            if d is None: break
            self._dust[i] = d[1]
            self._metadust.append(d[0])

    @property
    def N(self):
        return self._dust.size

    @property
    def species(self):
        return [ md.name for md in self._metadust ]

    @property
    def data(self) -> np.array:
        return self._dust

    def __getitem__(self, key: int) -> np.array:
        return self._dust[key]

