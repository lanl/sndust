import os, sys, itertools, argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import json

Composition = Dict[str, int]
ParsedReaction = List[Composition]

def rxn_parse(formula: str, eqlop: str = "->", addop: str = "+") -> ParsedReaction:
        comp_lr = [list(map(lambda m: (m, 1.0) if not m[0].isdigit() else (m[1:], np.float64(m[0])), [s.strip() for s in r.split(addop)])) for r in formula.split(eqlop)]
        return [dict(comp) for comp in comp_lr]

@dataclass
class BaseReaction:
    rid: int
    typeid: int 
    formula: str
    reactants: Composition
    products: Composition

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

def load_reactions(rxnfile: str)->Tuple[ArrheniusReaction, NucleationReaction]:
    rxns = []
    itr_id = itertools.count()
    with open(rxnfile, "r") as f:

        for line in f:

            if line.startswith("#") or line.strip() == "": continue
            rxn, params = line.split(":")

            reacts, prods = rxn_parse(rxn)

            ptk = params.split()
            rtype = int(ptk[-1])

            baser = ( next(itr_id), rtype, " ".join(rxn.split()), reacts, prods )

            if rtype == 101:
                ksl = []
                for s in ptk[0].split(","):
                    ksl.append(s)
                dpar = [np.float64(p) for p in ptk[1:-1]]
                rxns.append(NucleationReaction(*baser, ksl, *dpar ))
            elif rtype == 7:
                rxns.append(ArrheniusReaction(*baser, *[np.float64(p) for p in ptk[:-1]]))

    return rxns

def convert_to_json(rxns: Tuple[ArrheniusReaction,NucleationReaction], json_file_path: str)-> None:
    rxn_dict_tup = [asdict(rxn) for rxn in rxns]
    with open(json_file_path, "w") as jsf:
        json.dump(rxn_dict_tup, jsf, indent=2)
    print(f"Data written to {json_file_path}")
    

def main():
    ap = argparse.ArgumentParser(description="regenerate a network file in a more portable json format")
    ap.add_argument("-f", "--network_filename", help="name of the file", type=str, required=True)
    ap.add_argument("-o", "--output_filename", help="name of output file", type=str, default="nrxn.json")
    args = ap.parse_args()

    network_filename = args.network_filename
    output_filename = args.output_filename 
    
    rxns_tuple = load_reactions(network_filename)

    convert_to_json(rxns_tuple, output_filename)

if __name__ == "__main__":
    main()
