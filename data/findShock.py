import numpy as np
from dataclasses import dataclass, field

class test:
    def __init__(self,press,velo):
        pgas = np.array(press)
        self.pgas = pgas.astype(np.float)
        vel = np.array(velo)
        self.vel = vel.astype(np.float)

"""
 find_shocks attempts to locate a shocked cell looking at the for a 
 midpoint pressure change with a velocity drop

 fields: object with pressure (pgas) and velocity (vel)
 eps: parameter for cutoff; use 1.0 for more aggresive search

 returns: an array of indices of the cell index where we've detected a shock 
"""
def find_shocks(fields, eps)->np.array:
    si = []
    for i in range(2, fields.pgas.size-2):
        dp1 = fields.pgas[i+1] - fields.pgas[i-1]
        if np.abs(dp1)/np.minimum(fields.pgas[i+1],fields.pgas[i-1]) > eps \
                and fields.vel[i-1] > fields.vel[i+1]:
            si.append(i)

    return np.asarray(si, dtype=np.int32)

