import os, sys, itertools
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np

Composition = Dict[str, int]
ParsedReaction = List[Composition]

def rxn_parse(formula: str, eqlop: str = "->", addop: str = "+") -> ParsedReaction:
        comp_lr = [list(map(lambda m: (m, 1.0) if not m[0].isdigit() else (m[1:], np.float64(m[0])), [s.strip() for s in r.split(addop)])) for r in formula.split(eqlop)]
        return [dict(comp) for comp in comp_lr]

@dataclass
class BaseReaction:
    rid: int
    formula: str
    reactants: Composition
    ridx: np.array
    products: Composition
    pidx: np.array

@dataclass
class ArrheniusReaction(BaseReaction):
    alpha: np.float64
    beta: np.float64
    gamma: np.float64

@dataclass
class NucleationReaction(BaseReaction):
    key_speicies: List[str]
    A: np.float64
    B: np.float64
    sigma: np.float64
    a0: np.float64

def load_reactions(rxnfile: str)->Tuple[List[ArrheniusReaction], List[NucleationReaction]]:
    arrs = []
    nucl = []
    itr_id = itertools.count()
    with open(rxnfile, "r") as f:

        for line in f:

            if line.startswith("#") or line.strip() == "": continue
            rxn, params = line.split(":")

            reacts, prods = rxn_parse(rxn)

            ptk = params.split()

            baser = ( next(itr_id), " ".join(rxn.split()), reacts, np.empty(len(reacts), dtype=np.int32), prods, np.empty(len(prods), dtype=np.int32) )

            if np.int(ptk[-1]) == 101:
                ksl = []
                for s in ptk[0].split(","):
                    ksl.append(s)
                dpar = [np.float64(p) for p in ptk[1:-1]]
                nucl.append(NucleationReaction(*baser, ksl, *dpar ))
            elif np.int(ptk[-1]) == 7:
                arrs.append(ArrheniusReaction(*baser, *[np.float64(p) for p in ptk[:-1]]))

    return arrs, nucl

if __name__ == "__main__":

    #print(rxn_parse("C + O -> CO + PHOTON"))
    #print(rxn_parse("3Fe+4O->Fe3O4(s)"))
    #print(rxn_parse("O  H: OH", eqlop=":", addop=None))
    rxns = load_reactions("data/reactions.dat")

    for R in rxns:
        for ar in R:
            print(f"id: {ar.rid:5d}, reactions: {ar.formula:<28}, reactants: {ar.reactants}, products: {ar.products}")