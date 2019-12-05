import os, sys
import numpy as np

from molmass import *
from elements import ELEMENTS

if __name__ == "__main__":
    cp = {"N2": 0.78084,
         "O2": 0.20946,
         "Ar": 0.0934,
         "CO2": 0.00041332,
         "CH4": 0.00000114}

    ep = {}
    for mol, bw in cp.items():
        f = Formula(mol)
        molc = f.composition()
        for el in molc:
            if el[0] not in ep.keys():
                ep[el[0]] = 0.0

            #ep[el[0]] += el[2] * bw / ELEMENTS[el[0]].mass
            ep[el[0]] += bw * el[1]

    ntot = 0
    for e, n in ep.items():
        ntot += n

    for e, n in ep.items():
        print(f"{e}: {n/ntot:8.7f}")