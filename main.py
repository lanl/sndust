import os, sys, argparse, logging, json, traceback
import itertools as it

import numpy as np
from mpi4py.futures import MPIPoolExecutor

import simulation_constants as sim_const
from gas import SNGas
from particle import Particle, load_particle
# from network import Network
from network import Network
from stepper import Stepper
from solver import SolverSpec, Solver
from observer import Observer

ONLY_MODEL_2 = True
#np.seterr(invalid='raise')

def duster(settings, model_id, zone_id):
    assert ONLY_MODEL_2 and model_id == 2, "ONLY HAVE HYDRO DATA FOR MODEL_IDX=2"

    print(f"M{model_id} (Z{zone_id}) loading input...(slow for now)")
    p = load_particle(settings["particle_inputfile"], \
                      settings["hydrorun_inputfile"], \
                      model_id, zone_id)

    output_d = f"output_M{model_id:03d}"
    os.makedirs(output_d, exist_ok=True)
    output_f = os.path.join(output_d, f"dust{zone_id:04}")

    net     = Network(settings["network_file"])

    print(f"M{model_id} (Z{zone_id}) loaded, beginnging run: output[{output_f}]")

    gas     = SNGas(p, net)

    restart = settings["restart"]
    t_start = p.times[p.first_idx]
    t_end = p.times[p.last_idx]

    if restart:
        fName = './output_M'+str(model_id).zfill(3)+'/old'+str(zone_id).zfill(4)+'.hdf5'
        resF = h.File(fName,'r')
        keys = list(resF['root'].keys())
        data = resF['root'][keys[-1]][-1]
        t_start = data[1]
        NG = net._NG+7
        conc = list(data)[7:NG]
        mom = list(data)[NG:NG+net._ND]
        bins = list(data)[NG+net._ND*2:NG+net._ND*3]
        gas._c0 = np.append(np.append(conc,mom),bins)
        resF.close()

    step    = Stepper(gas, net)
    spec    = SolverSpec(time_start = t_start, time_bound = t_end, absolute_tol = settings["abs_tol"], \
                         relative_tol = settings["rel_tol"], max_timestep = settings["max_dt"])

    print(f"time_start = {p.times[p.first_idx]}")

    solv    = Solver(spec, step)
    obs     = Observer(output_f, net, gas, step, solv, settings)

    msg = f"M{model_id} (Z{zone_id}) ok"
    try:
        solv(obs)
    except Exception as e:
        msg=f"M{model_id} (Z{zone_id}) did not finish ok\n{repr(e)}"
        print(msg)
        traceback.print_exc(file=sys.stdout)
#        obs.dump(solv._steps)
    finally:
#        obs.runout(solv._steps, settings["align_tend_value"], res=settings["align_tend_resolution"])
        ## TIMING
        from stepper import S_DEBUG, S_FASTMATH, S_NOPYTHON, S_PARALLEL
        print(f"DEBUG={S_DEBUG}, NOPYTHON={S_NOPYTHON}, FASTMATH={S_FASTMATH}, PARALLEL={S_PARALLEL}")
    return msg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-keys", action="store_true")
    ap.add_argument("-c", "--configfile", type=str, required=True)
    ap.add_argument("-N", "--ncpu", type=int, default=3)

    args = ap.parse_args()

    with open(args.configfile, "r") as jfs:
        settings = json.load(jfs)

    model_id = 2 # note, zone ids from 0 ... 1545
    zone_ids = np.arange(0, 100) # TODO: use particle data to get all zone numbersrestart = settings["restart"]
    if restart:
        zone_ids = settings["res_zones"]
    
    if 0:
        # TODO: better schedualing
        with MPIPoolExecutor(max_workers=args.ncpu) as pool:
            for result in pool.map(duster, it.repeat(settings), it.repeat(model_id), zone_ids):
                print(result)
    else:
        res_msg = duster(settings, model_id, 200)
        print(res_msg)
        # for iz in zone_ids:
        #     duster(settings, model_id, iz)

    print("done")





