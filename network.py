import os, sys, json
import itertools as it
from typing import List, Dict, Tuple

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import periodictable as pt

from simulation_constants import N_MOMENTS, MAX_REACTANTS, MAX_PRODUCTS, fourpiover3, numBins, NDust
from physical_constants import amu2g, ang2cm

Composition = Dict[str, int]
ParsedReaction = List[Composition]
ReactionSet = Dict[str, list]

#def rxn_parse(formula: str, eqlop: str = "->", addop: str = "+") -> ParsedReaction:
#        comp_lr = [list(map(lambda m: (m, 1.0) if not m[0].isdigit() else (m[1:], np.float64(m[0])), [s.strip() for s in r.split(addop)])) for r in formula.split(eqlop)]
#        return [dict(comp) for comp in comp_lr]

def remove_phase(formula: str) -> str:
    try:
        sidx = formula.index('(')
    except:
        return formula
    return formula[:sidx]

def extend_nptype(olddtype, newtype):
    lst = [ (k, v[0]) for k, v in it.chain(olddtype.fields.items(), newtype.fields.items()) ]
    return np.dtype(lst)

def stack_arrays(arr1, arr2):
    ret = np.array(0, dtype=extend_nptype(arr1.dtype, arr2.dtype))

    for arr in (arr1, arr2):
        ret[list(arr.dtype.names)] = arr[list(arr.dtype.names)]

    return ret

base_reaction_type = np.dtype([
    ("nr", np.int32),
    ("react_idx", np.int32, (MAX_REACTANTS,)),
    ("react_mass", np.float64, (MAX_REACTANTS,)),
    ("react_nu", np.float64, (MAX_REACTANTS,)),
    ("np", np.int32),
    ("prod_idx", np.int32, (MAX_PRODUCTS,)),
    ("prod_mass", np.float64, (MAX_PRODUCTS,)),
    ("prod_nu", np.float64, (MAX_PRODUCTS,)),
    ("active", np.int32)
], align=True)

nucleation_type = np.dtype([
    ("nkeysp", np.int32),
    ("keysp_idx", np.int32, (MAX_REACTANTS)),
    ("bin_idx", np.int32),
    ("A", np.float64),
    ("B", np.float64),
    ("sigma", np.float64),
    ("a0", np.float64),
    ("a02", np.float64),
    ("omega0", np.float64),
    ("rho_p", np.float64)
], align=True)

nucleation_numpy_type = extend_nptype(base_reaction_type, nucleation_type)

class Network:
    def __init__(self, reaction_filename:str):
        self._reaction_filename = reaction_filename
        with open(self._reaction_filename) as jf:
            self._reactions = json.load(jf)

        self._species = dict()
        self._species_gas = list()
        self._species_dust = list()

        self._reactions_gas = list()
        self._reactions_dust = list()

        self._solution_len = -1
        #self._gas_slice = None
        #self._dust_slice = None

        #self._load()
        self._ND = 0
        self._NG = 0
        self._map_species()
        self._cat_reactions()

    def _map_species(self):
        _idx = 0
        _uniqs = list()
        # go through network, adding species
        for reaction in self._reactions:
            for _, species in it.chain(reaction["reactants"], reaction["products"]):
                if species not in _uniqs:
                    _uniqs.append(species)
                    # self._species[species] = _idx
                    if "(s)" in species:
                    #    _idx += 4
                        self._ND += 1
                        self._species_dust.append(species)
                    else:
                    #    _idx += 1
                        self._NG += 1
                        self._species_gas.append(species)
        NDust = self._ND
        for i in range(self._NG):
            self._species[self._species_gas[i]] = i

        for i in range(self._ND):
            self._species[self._species_dust[i]] = self._NG + i * N_MOMENTS

    def _cat_reactions(self):
        for reaction in self._reactions:
            _dust_yes = False
            for p in reaction["products"]:
                if "(s)" in p:
                    _dust_yes = True
                    break

            if _dust_yes:
                self._reactions_dust.append(reaction)
            else:
                self._reactions_gas.append(reaction)

    @property
    def solution_size(self) -> np.int32:
        return self._NG + N_MOMENTS * self._ND + self._ND * numBins # trying to get bins

    @property
    def NG(self):
        return self._NG

    @property
    def ND(self):
        return self._ND

    @property
    def species(self) -> List[str]:
        return list(self._species)

    @property
    def species_map(self) -> Dict[str, np.int32]:
        return self._species

    def sidx(self, species: str) -> np.int32:
        return self._species[species]

    def generate_nparrays(self):
        buf = { "nucleation" :  [] }
        for reaction in self._reactions:
            base_npt = np.array(0, dtype=base_reaction_type)
            base_npt["nr"] = len(reaction["reactants"])
            for i in range(base_npt["nr"]):
                _rstr = reaction["reactants"][i][1]
                base_npt["react_idx"][i] = self._species[_rstr]
                base_npt["react_nu"][i] = reaction["reactants"][i][0]
                base_npt["react_mass"][i] = pt.formula(_rstr).mass * amu2g

            base_npt["np"] = len(reaction["products"])
            for i in range(base_npt["np"]):
                _pstr = reaction["products"][i][1]
                base_npt["prod_idx"][i] = self._species[_pstr]
                base_npt["prod_nu"][i] = reaction["products"][i][0]
                base_npt["prod_mass"][i] = pt.formula(remove_phase(_pstr)).mass * amu2g

            base_npt["active"] = 1

            # this is a hacked polymorphism, stacking base and derived
            # datatypes together to make a new datatype
            sub_npt = np.empty(0)
            _par = reaction["parameters"]
            if reaction["type"] == "nucleation":
                sub_npt = np.array(0, dtype=nucleation_type)
                _ks = _par["key_species"]
                sub_npt["nkeysp"] = len(_ks)
                for i in range(sub_npt["nkeysp"]):
                    sub_npt["keysp_idx"][i] = self._species[_ks[i]]

                sub_npt["A"] = _par["A"]
                sub_npt["B"] = _par["B"]
                sub_npt["sigma"] = _par["sigma"]
                sub_npt["a0"] = _par["a0"] * ang2cm
                sub_npt["a02"] = sub_npt["a0"] ** 2
                sub_npt["omega0"] = fourpiover3 * sub_npt["a0"] ** 3
                sub_npt["rho_p"] = base_npt["prod_mass"][0] / sub_npt["omega0"]

            buf[reaction["type"]].append(stack_arrays(base_npt, sub_npt))

        ret = dict.fromkeys(buf)
        for k, v in buf.items():
            if k == "nucleation":
                _npt = extend_nptype(base_reaction_type, nucleation_type)
                ret[k] = np.empty(len(v), dtype=_npt)
                for i in range(len(v)):
                    ret[k][i] = buf[k][i]

        return ret

    def generate_header(self):
        ret = [f"n_{s}" for s in self._species]
        return ret

if __name__ == "__main__":
    n = Network("data/reactions.json")

    xn = n.generate_nparrays()

    for x in xn["nucleation"]:
        print(x)
    print(n.generate_header())
