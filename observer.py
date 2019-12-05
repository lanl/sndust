import os, sys, logging, itertools
from timeit import default_timer
from simulation_constants import N_MOMENTS
from typing import List

import numpy as np
import gzip as gz
import json
import h5py as h5
import scipy.integrate as integrate

from simulation_constants import N_MOMENTS
from physical_constants import sec2day
from network import Network
from gas import SNGas
from stepper import Stepper, dust_calc
#from solver import Solver

class Observer(object):
    def __init__(self, obs_file: str, net: Network, gas: SNGas, step: Stepper, solv, screen_every: int = -1,
                    store_every: int = -1, write_every: int = -1):
        self._screen_every = screen_every
        self._store_every = store_every
        self._write_every = write_every

        self._gas = gas
        self._net = net
        self._step = step
        self._solv = solv
        self._ode = self._solv._ode

        self._obsfname = f"{obs_file}.hdf5"
        self._logfname = f"{obs_file}.log"

        # self._h5f = h5.File(self._obsfname, "w")
        self._h5type = np.dtype([
            ("step", np.int32),
            ("time", np.float64),
            ("time_step", np.float64),
            ("temperature", np.float64),
            ("density", np.float64),
            ("volume", np.float64),
            ("x", np.float64),
            *[ (f"N_{s}", np.float64) for s in self._net._species_gas ],
            *[ (f"M_{s}", np.float64, (N_MOMENTS,)) for s in self._net._species_dust ],
            *[ (f"calc_{s}", dust_calc) for s in self._net._species_dust ]
        ])

        self._store_chunk = self._write_every // self._store_every
        self._store = np.zeros(self._store_chunk, dtype=self._h5type)
        self._store_idx = 0

        with h5.File(self._obsfname, "w") as hf:
            grp = hf.create_group("root")
            grp.attrs["gas_names"] = self._net._species_gas
            grp.attrs["initial_composition"] = self._gas.concentration_0
            grp.attrs["dust_names"] = self._net._species_dust
            grp.attrs["dust_sigma"] = [ _d["parameters"]["sigma"] for _d in self._net._reactions_dust ]
            grp.attrs["dust_radius"] = [ _d["parameters"]["a0"].a0 for _d in self._net._reactions_dust]

        with open(self._logfname, "w") as lf:
            print(f"log file for {self._gas._sid}", file=lf)
            print(f"=============================")

        self._tot_storetime = 0
        self._tot_dumptime = 0
        self._tot_screentime = 0

    def _store_h5dat(self, step, time):
        if self._ode.t_old is None:
            return
        rho = self._gas.Density(self._ode.t)
        self._store[self._store_idx]["step"] = step
        self._store[self._store_idx]["time"] = time
        self._store[self._store_idx]["time_step"] = self._ode.t - self._ode.t_old
        self._store[self._store_idx]["temperature"] = self._gas.Temperature(time)
        self._store[self._store_idx]["density"] = rho
        self._store[self._store_idx]["volume"] = self._gas.mass_0 / rho
        self._store[self._store_idx]["x"] = self._gas._fx(time)

        for i, s in enumerate(self._net._species_gas):
            self._store[self._store_idx][f"N_{s}"] = self._ode.y[i]

        for i, s in enumerate(self._net._species_dust):
            _didx = self._net._NG + i * N_MOMENTS
            self._store[self._store_idx][f"M_{s}"] = self._ode.y[_didx : _didx + N_MOMENTS]
            self._store[self._store_idx][f"calc_{s}"] = self._step._dust_calc[i]
        self._store_idx += 1

    def dump(self, step:int):

        with h5.File(self._obsfname, "a") as h5f:
            h5f["root"].create_dataset(f"steps_{step}", (self._store_chunk,), data=self._store, dtype=self._h5type)
        self._store_idx = 0


    def runout(self, step, tf, res = 1):
        _scrn = f"running out data to final time...(last step time = {self._ode.t})\n"
        added_times = []
        dt = (tf - self._ode.t) / float(res * self._store_chunk)
        stp = step
        tim = self._ode.t

        for i in range(res):
            self._store_idx = 0
            for j in range(self._store_chunk):
                tim += dt
                stp += 1
                self._store_h5dat(stp, tim)
                added_times.append(tim)
                #h5f.create_dataset(f"steps_{stp}", (self._store_chunk,), data=self._store, dtype=self._h5type)

            self.dump(stp)

        _scrn += f"dumped {len(added_times)} new data; times are: \n"
        _scrn += ",".join([ f'{_adt:6.5E}' for _adt in added_times ])
        _scrn += "\n"
        with open(self._logfname, "a") as _lf:
            print(_scrn, file=_lf)

    def __call__(self, step: np.int32, force_screen: bool = False):

        if step % self._store_every == 0:
            _xtime0 = default_timer()
            self._store_h5dat(step, self._ode.t)
            self._tot_storetime += (default_timer() - _xtime0)

        if step % self._write_every == 0:
            _xtime0 = default_timer()
            self.dump(step)
            self._tot_dumptime += (default_timer() - _xtime0)

        if (step % self._screen_every == 0 and self._screen_every > 0) or force_screen:
            _xtime0 = default_timer()
            try:
                # s = self._store[-1]
                s = self._store[self._store_idx - 1]
            except:
                return
            _scrn = f"== {self._gas._sid}[{self._gas._pid}] solution at step {step:8d} ==\n"
            _syshead = ""
            _sysstat = ""
            # for k, v in s.items():
            for n in self._h5type.names[:8]:
                #if k == "sys": continue

                _syshead += f"{n:<11s} "
                if n == "step":
                    # _sysstat += f"{v:<8d} "
                    _sysstat += f"{s[n]:<8d} "
                else:
                    #_sysstat += f"{v:>8.7E} "
                    _sysstat += f"{s[n]:>8.7E} "

            _scrn += f"{_syshead}\n"
            _scrn += "----------------------------------\n"
            _scrn += f"{_sysstat}\n"
            # print(_syshead)
            # print("----------------------------------")
            # print(_sysstat)

            # print("==--gas--==")
            _scrn += "==--gas--==\n"
            #for k, v in s["sys"]["gas"].items():
            #    print(f"{k:>8s} [{v:>6.5E}]")
            for species in self._net._species_gas:
                _key = f"N_{species}"
                _scrn += f"{species:>8s} [{s[_key]:>6.5E}]\n"

            # print("==--dust--==")
            _scrn += "==--dust--==\n"
            # for k, v in s["sys"]["dust"].items():
            #     print(f"{k:<8s}{'|' if v['active'] == 1 else 'o'} : S[{v['saturation']: >6.5E}] J[{v['nucleation']: >6.5E}] n*[{v['critical_size']: >6.5E}]")
            #     _moms = ", ".join([ f"M_{i}[{m: >6.5E}]" for i, m in enumerate(v["moments"]) ])
            #     print(f"++ moments = {_moms}")
            for species in self._net._species_dust:
                    _ckey = f"calc_{species}"
                    _mkey = f"M_{species}"
                    _moms = ", ".join([ f"M_{i}[{m: >6.5E}]" for i, m in enumerate(s[_mkey]) ])
                    _scrn += f"{species}: S[ {s[_ckey]['S']: >6.5E} ] J[ {s[_ckey]['Js']: >6.5E} ]\n"
                    _scrn += f"++++ moments ++++ {_moms}\n"

            tot_call_time = np.sum([self._tot_dumptime, self._tot_screentime, self._tot_storetime])
            timpct = [ _tim / tot_call_time * 100.0 for _tim in [self._tot_dumptime, self._tot_screentime, self._tot_storetime] ]
            _scrn += "==--analytics--==\n"
            _scrn += f" [  total integrate step time ] {self._solv._tot_steptime:5.4E} s\n"
            _scrn += f" [   avg integrate step time  ] {self._solv._avg_steptime:5.4E} s\n"
            _scrn += f" [       total obs time       ] {tot_call_time:5.4E} s\n"
            _scrn += f" [ [store%] [dump%] [screen%] ] {' '.join([f'{_t:5.3f}' for _t in timpct])}\n"
            _scrn += "=======================\n"
            with open(self._logfname, "a") as _logf:
                print(_scrn, file=_logf)
            self._tot_screentime += (default_timer() - _xtime0)