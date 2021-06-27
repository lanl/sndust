import os, sys, logging, itertools
from timeit import default_timer
from simulation_constants import N_MOMENTS
from typing import List

import numpy as np
import gzip as gz
import json
import h5py as h5
import scipy.integrate as integrate

from simulation_constants import N_MOMENTS,dTime,numBins
from physical_constants import sec2day
from network import Network
from gas import SNGas
from stepper import Stepper, dust_calc
#from solver import Solver

OBS_LINESEP1 = "----------------------------------\n"
OBS_LINESEP2 = "==================================\n"

class Observer(object):
    def __init__(self, obs_file: str, net: Network, gas: SNGas, step: Stepper, solv, rt_settings: dict):

        self._triggers = np.array([
                (rt_settings["history_every"], "history"),
                (rt_settings["screen_short_every"], "screen_short"),
                (rt_settings["screen_all_every"], "screen_all"),
                (rt_settings["store_hdf5_every"], "store_hdf5"),
                (rt_settings["write_hdf5_every"], "write_hdf5")
                ], 
            dtype=([("value", "i8"), ("trigger", "U32")])
        )
        self._gas = gas
        self._net = net
        self._step = step
        self._solv = solv
        self._ode = self._solv._ode
        
        self._obsfname = f"{obs_file}.hdf5"
        self._histfname = f"{obs_file}.hist"

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
            *[ (f"calc_{s}", dust_calc) for s in self._net._species_dust ],
            *[ (f"sizeBin_{s}", np.float64, (numBins,)) for s in self._net._species_dust ] # trying to add size bins to int
        ])

        self._store_chunk = rt_settings["write_hdf5_every"] // rt_settings["store_hdf5_every"]
        self._store = np.zeros(self._store_chunk, dtype=self._h5type)
        self._store_idx = 0

        with h5.File(self._obsfname, "w") as hf:
            grp = hf.create_group("root")
            grp.attrs["gas_names"] = self._net._species_gas
            grp.attrs["initial_composition"] = self._gas.concentration_0
            grp.attrs["dust_names"] = self._net._species_dust
            grp.attrs["dust_sigma"] = [ _d["parameters"]["sigma"] for _d in self._net._reactions_dust ]
            grp.attrs["dust_radius"] = [ _d["parameters"]["a0"].a0 for _d in self._net._reactions_dust]

        with open(self._histfname, "w") as lf:
            print(f"hisotry file for {self._gas._sid}", file=lf)
            print(OBS_LINESEP2, file=lf)

        self._tot_storetime = 0
        self._tot_dumptime = 0
        self._tot_screentime = 0

    def _get_frame(self, step):
        frame = np.array(1, dtype=self._h5type)
        rho = self._gas.Density(self._ode.t)
        frame["step"] = step
        frame["time"] = self._ode.t
        frame["time_step"] = self._ode.t - self._ode.t_old
        frame["temperature"] = self._gas.Temperature(self._ode.t)
        frame["density"] = rho
        frame["volume"] = self._gas.mass_0 / rho
        frame["x"] = self._gas._fx(self._ode.t)

        for i, s in enumerate(self._net._species_gas):
            frame[f"N_{s}"] = self._ode.y[i]

        for i, s in enumerate(self._net._species_dust):
            _didx = self._net._NG + i * N_MOMENTS
            _sidx = self._net._NG + len(self._net._species_dust) * N_MOMENTS + i * numBins
            frame[f"M_{s}"] = self._ode.y[_didx : _didx + N_MOMENTS]
            frame[f"calc_{s}"] = self._step._dust_calc[i]
            frame[f"sizeBin_{s}"] = self._ode.y[_sidx : _sidx + numBins] # trying to add size bins to int
        return frame

    def _store_h5dat(self, frame):
        if self._ode.t_old is None:
            return
        self._store[self._store_idx] = frame
        self._store_idx += 1

    def _dump_hdf5(self, step:int):

        with h5.File(self._obsfname, "a") as h5f:
            h5f["root"].create_dataset(f"steps_{step}", (self._store_chunk,), data=self._store, dtype=self._h5type)
        self._store_idx = 0

    def _long_header(self, frame):
        blob = f"== {self._gas._sid}[{self._gas._pid}] solution at step {frame['step']:8d} ==\n"
        syshead = ""
        sysstat = ""
        for n in self._h5type.names[1:7]:
            syshead += f"{n:<11s} "
            sysstat += f"{frame[n]:>8.7E} "

        blob += f"{syshead}\n"
        blob += OBS_LINESEP1
        blob += f"{sysstat}\n"
        return blob
    
    def _solution_all(self, frame):
        blob = "==--gas--==\n"
        for species in self._net._species_gas:
                _key = f"N_{species}"
                blob += f"{species:>8s} [{frame[_key]:>6.5E}]\n"
        blob = "==--gas--==\n"
        for species in self._net._species_dust:
                _ckey = f"calc_{species}"
                _mkey = f"M_{species}"
                _moms = ", ".join([ f"M_{i}[{m: >6.5E}]" for i, m in enumerate(frame[_mkey]) ])
                blob += f"{species}: S[ {frame[_ckey]['S']: >6.5E} ] J[ {frame[_ckey]['Js']: >6.5E} ]\n"
                blob += f"++++ moments ++++ {_moms}\n"
                _skey = f"sizeBin_{species}" # trying to add size bins to int
                _sizes = ", ".join([ f"sizeBin_{i}[{m: >6.5E}]" for i, m in enumerate(frame[_skey]) ]) # trying to add size bins to int
                blob += f"+++ size bins +++ {_sizes}\n" # trying to add size bins to int
        return blob

    def _solution_short(self, frame):
        blob = f"temperature = {frame['temperature']} K, density = {frame['density']} g/cm3\n"
        return blob

    def _short_header(self, frame):
        blob =  f"== [pid={self._gas._pid} step={frame['step']:12d}] ==\n"
        blob += f"- time = {frame['time']:>6.5E} | dt = {frame['time_step']:>6.5E}\n"
        blob += OBS_LINESEP1
        return blob

    def __call__(self, step: np.int32, force_screen: bool = False):
        dTime = self._ode.t - self._ode.t_old

        actions = self._triggers["trigger"][np.where( step % self._triggers["value"] == 0 )[0]]
        if actions.size > 0:
            
            frame = self._get_frame(step)

            for action in actions:
                if action == "store_hdf5":
                    _xtime0 = default_timer()
                    self._store_h5dat(frame)
                    self._tot_storetime += (default_timer() - _xtime0)
                elif action == "write_hdf5":
                    _xtime0 = default_timer()
                    self._dump_hdf5(step)
                    self._tot_dumptime += (default_timer() - _xtime0)
                elif action == "history":
                    with open(self._histfname, "a") as histf:
                        print(self._long_header(frame) + self._solution_all(frame), file=histf)
                elif action == "screen_short":
                    print(self._short_header(frame) + self._solution_short(frame))
                elif action == "screen_all":
                    pass
                else:
                    raise ValueError(f"{action} is not understood trigger")

            # print("==--dust--==")

   #         tot_call_time = np.sum([self._tot_dumptime, self._tot_screentime, self._tot_storetime])
   #         timpct = [ _tim / tot_call_time * 100.0 for _tim in [self._tot_dumptime, self._tot_screentime, self._tot_storetime] ]
   #         _scrn += "==--analytics--==\n"
   #         _scrn += f" [  total integrate step time ] {self._solv._tot_steptime:5.4E} s\n"
   #         _scrn += f" [   avg integrate step time  ] {self._solv._avg_steptime:5.4E} s\n"
   #         _scrn += f" [       total obs time       ] {tot_call_time:5.4E} s\n"
   #         _scrn += f" [ [store%] [dump%] [screen%] ] {' '.join([f'{_t:5.3f}' for _t in timpct])}\n"
   #         _scrn += "=======================\n"
   #         with open(self._logfname, "a") as _logf:
   #             print(_scrn, file=_logf)
   #         self._tot_screentime += (default_timer() - _xtime0)
