from dataclasses import dataclass, field
from typing import Union

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

import numba
from numba.experimental import jitclass
from numba import float64

onethird = 1./3.
twothird = 2. * onethird
fourpi = 4. * np.pi
#echarge = e.emu.value
echarge = np.sqrt(14.4) # for annoying reasons, use e in units sqrt(eV AA)
bohrr = 0.5291772106699999
kB_eV = 8.617333262145E-5
kB_erg = 1.380649E-16
solZ = 0.012
g2amu = 6.022e+23
amu2g = 1. / g2amu


spec = [
    ('u0', float64),
    ('md', float64),
    ('mi', float64),
    ('zd', float64),
    ('zi', float64),
    ('K', float64),
    ('mu', float64),
    ('asc', float64),
    ('alph', float64),
    ('eth', float64),
]

@jitclass(spec)
#@dataclass

class Yield:
    def __init__(self,u0, md, mi, zd, zi, K):
        self.u0 = u0
        self.md = md
        self.mi = mi
        self.zd = zd
        self.zi = zi
        self.K = K
        self.mu = 0
        self.asc = 0
        self.alph = 0
        self.eth = 0
        self.mu = self.md / self.mi

        if self.mu > 1.0:
            self.alph = 0.3 * (self.mu - 0.6) ** twothird
        elif self.mu > 0.5:
            self.alph = 0.1 / self.mu + 0.25 * (self.mu - 0.5) ** 2
        else:
            self.alph = 0.2

        gi = 4.0 * self.mi * self.md / (self.mi + self.md) ** 2
        imu = 1.0 / self.mu
        if imu <= 0.3:
            self.eth = self.u0 / ( gi * (1.0 - gi) )
        else:
            self.eth = 8.0 * self.u0 * imu ** onethird
        self.asc = 0.885 * bohrr / np.sqrt( self.zi ** twothird + self.zd ** twothird )

    # use Y as scalar (for integration) or vector (for evaluation over a field)
    def Y(self, E: Union[np.float64, np.array]) -> Union[np.float64, np.array]:
        eps = self.asc * E * self.md / ( self.zi * self.zd * echarge ** 2 * (self.mi + self.md) )
        x = np.sqrt(eps)
        s = 3.441 * x * np.log(eps + 2.718) / ( 1.0 + 6.35 * x + eps * (6.882 * x - 1.708) )
        xsc = fourpi * self.asc * self.zi * self.zd * echarge ** 2 * self.mi * s / ( self.mi + self.md )
        es = self.eth / E
        return 4.2E-2 * xsc * self.alph * (1.0 - es**twothird) * (1.0 - es)**2. / ( self.u0 * (self.K * self.mu + 1.0) )
