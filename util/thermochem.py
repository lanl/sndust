from abc import get_cache_token
import os, sys, json
import re

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# test_net = [ [[[1, "C"]], ["C(gr)"]],
#             # [[[1, "Si"]], "Si(L)"],
#             [[[1, "Mg"], [1, "O"]], ["MgO(L)", "MgO(cr)"]] ,
#             [[[1, "SiO"],[1, "O"]], ["SiO2(L)", "SiO2(a-qz)", "SiO2(b-qz)", "SiO2(b-crt)"]],
#             [[[1, "Mg"], [1, "SiO"],[2, "O"]], ["MgSiO3(I)","MgSiO3(II)","MgSiO3(III)", "MgSiO3(L)"]],
#             [[[2, "Mg"], [1, "SiO"],[3, "O"]], ["Mg2SiO4(s)","Mg2SiO4(L)"]],
#             [[[2, "AL"], [3, "O"]], ["AL2O3(L)","AL2O3(a)"]],
#             [[[1, "Na"], [1, "CL"]], ["NaCL(s)","NaCL(L)"]],
#             [[[1, "Na"], [1, "Br"]], ["NaBr(s)", "NaBr(L)"]],
#             [[[1, "Si"], [1, "C"]], ["SiC(b)", "SiC(b-1)", "SiC(L)"]]
#             ]
test_net2 = [ [[[1, "C"]], "C"],
            # [[[1, "Si"]], "Si(L)"],
            [[[1, "Mg"], [1, "O"]], "MgO"] ,
            [[[1, "SiO"],[1, "O"]], "SiO2"],
            [[[1, "Mg"], [1, "SiO"],[2, "O"]], "MgSiO3"],
            [[[2, "Mg"], [1, "SiO"],[3, "O"]], "Mg2SiO4"],
            [[[2, "AL"], [3, "O"]], "AL2O3"],
            [[[1, "Na"], [1, "CL"]], "NaCL"],
            [[[1, "Na"], [1, "Br"]], "NaBr"],
            [[[1, "Si"], [1, "C"]], "SiC"],
            [[[1, "Ba"], [1, "O"]], "BaO"],
            [[[1, "S"],  [2, "Na"], [3, "O"]], "Na2SO3"],
            [[[1, "K"]], "K"]
            ]

coeff_npt = np.dtype([
            ("_T", np.float64, (2,)),
            ("_c", np.float64, (9,))
            ])
# g/rt = h/rt - s/r
# g/rt = a1 (1 - ln T) - a2 T / 2 - a3 T ** 2 / 6 - a4 T ** 3 / 12 - a5 T ** 4 / 20 + a6 / T - a7
#    // G(T)/RT = H(T)/RT - S(T)/R
#     if (func == GIBBS) {
#         params[0] = -0.5 / T2;
#         params[1] = (log(T) + 1.0) / T;
#         params[2] = 1.0 - log(T);
#         params[3] = -0.5 * T;
#         params[4] = -T2 / 6.0;
#         params[5] = -T3 / 12.0;
#         params[6] = -T4 / 20.0;
#         params[7] = 1.0 / T;
#     }
# void Nasa9Polynomial::gibbs(const double *const p_params, double &g) const
# {
#     int tr = tRange(-2.0 * p_params[3]);

#     g = -mp_coefficients[tr][8];
#     for (int i = 0; i < 8; ++i)
#         g += mp_coefficients[tr][i] * p_params[i];
# }

def Tterms(T):
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    logT = np.log(T)
    terms = np.zeros(8)
    terms[0] = -0.5 / T2
    terms[1] = (logT + 1.0) / T
    terms[2] = 1.0 - logT
    terms[3] = -0.5 * T
    terms[4] = -T2 / 6.0
    terms[5] = -T3 / 12.0
    terms[6] = -T4 / 20.0
    terms[7] = 1.0 / T

    return terms

def gibbs(T, coefficents):
    trms = Tterms(T)
    g = - coefficents[8]
    g += np.sum(coefficents[:-1] * trms)
    return g

def load_record(database, species):
    for rec in database:
        if rec["name"] == species:
            return rec
    raise Exception(f"cannot find {species}")

def get_condensed_phases(database, species):
    cspec = []
    #rgxs = "[\(\[].*?[\)\]]"
    ignore = "NO NN NO2".split()
    for rec in database:
        name = rec["name"]
        start = name.find("(")
        end = name.find(")")
        if name[start+1:end] in ignore:
            continue
        if start != -1 and end != -1:
            if species == name[:start]:
                cspec.append(name)
    return cspec

def coeff(entry):
    return np.array([ float(val.replace("D","E")) for val in (entry["a"] + entry["b"]) ])

def get_coeffs(rec):
    _ntinv = rec["n_tintervals"]
    r_coef = np.zeros((_ntinv, 11))

    for i, coef in enumerate(rec["coeff"]):
        r_coef[i][0] = coef["temperature_lo"]
        r_coef[i][1] = coef["temperature_hi"]

        for j, val in enumerate(coef["a"] + coef["b"]):
            r_coef[i][j + 2] = np.float64(val.replace("D","E"))
            #r_coef[i][j + 2] = val

    return r_coef

def select_coeff(acoef, T):
    for i in range(acoef.shape[0]):
        if acoef[i,0] <= T and acoef[i,1] >= T:
            return acoef[i, 2:]
    return None

def delta_gibbs_reaction(database, rxn):

    prods = get_condensed_phases(database, rxn[1])
    nreact = len(rxn[0])

    r_stoc = [ r[0] for r in rxn[0] ]
    r_recs = [ load_record(database, r[1]) for r in rxn[0] ]
    #p_recs = [ load_record(database, p) for p in rxn[1] ]
    p_recs = [ load_record(database, p) for p in prods ]

    rcoefs = [ get_coeffs(r) for r in r_recs ]
    pcoefs = np.concatenate([ get_coeffs(p) for p in p_recs ], axis=0)

    _npnt = 10000

    _tmin = np.min(pcoefs[:, 0])
    _tmax = np.max(pcoefs[:, 1])
    _Trng = np.linspace(_tmin, _tmax, _npnt)

    del_G = np.zeros(_npnt)
    idxs = np.zeros(_npnt, dtype=np.bool)
    for i in range(_npnt):
        pc = select_coeff(pcoefs, _Trng[i])
        if pc is None:
            print("bd")
            print(pcoefs.shape[0])
            print(_Trng[i])
            sys.exit()


        idx_ok = True
        for j in range(nreact):
            rc = select_coeff(rcoefs[j], _Trng[i])
            if rc is None:
                del_G[i] = np.nan
                idx_ok = False
            else:
                del_G[i] -= r_stoc[j] * gibbs(_Trng[i], rc)
        if idx_ok:
            del_G[i] += gibbs(_Trng[i], pc)
            idxs[i] = True
    # plt.plot(_Trng, del_G, label=f"{rxn[-1]}")
#    plt.show()
    return _Trng[idxs], del_G[idxs]

def fitfunc(x, a, b):
    return -a/x + b

if __name__ == "__main__":
    _data = "/home/mauneyc/scratch/misc_dev/dtra/integ/data/nasa9_data.json"
    jsondata = None
    with open(_data, "r") as jsf:
        jsondata = json.load(jsf)

    if jsondata is None:
        print("Bad")
        sys.exit()

    #test_sp = "SiO2 AL2O3 MgSiO3 Mg2SiO4 FeO C SiC BaO K Na2SO3"
    #for sp in test_sp.split():
    #    print(sp, get_condensed_phases(jsondata, sp))
    fitdat = []
    for _r in test_net2:
        fitdat.append(delta_gibbs_reaction(jsondata, _r))

    for (xd, yd), s in zip(fitdat, test_net2):
        popt, pcov = curve_fit(fitfunc, xd, yd)
        _r = "+".join([f"{x}{y}" for x, y in s[0] ])
        _p = s[1]
        _v = " ".join([f"{x / s[0][0][0]:5.4E}" for x in popt])
        print(f"{_r} = {_p} : {_v}")
#        print(s[1], popt / s[0][0][0])
    # plt.plot(np.linspace(1.0E3, 5.0E3, 1000), np.zeros(1000), "--")
    # plt.legend()
    # plt.show()