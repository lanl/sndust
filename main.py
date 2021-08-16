import os, sys, argparse, logging, json, traceback
import itertools as it

import numpy as np
from mpi4py.futures import MPIPoolExecutor

import simulation_constants as sim_const
from gas import SNGas
from particle import Particle, load_particle, generate_test_particle
# from network import Network
from network import Network
from stepper import Stepper
from solver import SolverSpec, Solver
from observer import Observer

ONLY_MODEL_2 = True
#np.seterr(invalid='raise')

def duster(settings, model_id, zone_id):
    assert ONLY_MODEL_2 and model_id == 2, "ONLY HAVE HYDRO DATA FOR MODEL_IDX=2"

    if(settings["particle_inputfile"] == "GENERATE_TEST"):
        p = generate_test_particle()
        output_d = "output_TEST"
        output_f = "TEST"
        print(f"M test particle generated")
    else:
        print(f"M{model_id} (Z{zone_id}) loading input...(slow for now)")
        p = load_particle(settings["particle_inputfile"], \
                          settings["hydrorun_inputfile"], \
                          model_id, zone_id)

        print(f"M{model_id} (Z{zone_id}) loaded")

        output_d = f"output_M{model_id:03d}"
        output_f = f"dust{zone_id:04}"

    output_f = os.path.join(output_d, output_f)
    print(f"Preparing output: {output_f}")
    os.makedirs(output_d, exist_ok=True)

    net     = Network(settings["network_file"])

    gas     = SNGas(p, net)
    #step    = Stepper(gas, net)
    step    = Stepper(p, net)
    spec    = SolverSpec(time_start = p.times[p.first_idx], time_bound = p.times[p.last_idx], absolute_tol = settings["abs_tol"], \
                    relative_tol = settings["rel_tol"], max_timestep = settings["max_dt"])
    solv    = Solver(spec, p, net, step)
    obs     = Observer(output_f, net, gas, step, solv, screen_every=settings["screen_every"], store_every=settings["store_every"], write_every=settings["write_every"])

    msg = f"M{model_id} (Z{zone_id}) ok"
    try:
        solv(obs)
    except Exception as e:
        msg=f"M{model_id} (Z{zone_id}) did not finish ok\n{repr(e)}"
        print(msg)
        traceback.print_exc(file=sys.stdout)
        obs.dump(solv._steps)
    finally:
        obs.runout(solv._steps, settings["align_tend_value"], res=settings["align_tend_resolution"])
    return msg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-keys", action="store_true")
    ap.add_argument("-c", "--configfile", type=str, required=True)
    ap.add_argument("-N", "--ncpu", type=int, default=3)

    args = ap.parse_args()

    with open(args.configfile, "r") as jfs:
        settings = json.load(jfs)

    model_id = 2
    zone_ids = np.arange(0, 100) # TODO: use particle data to get all zone numbers
    if 0:
        # TODO: better schedualing
        with MPIPoolExecutor(max_workers=args.ncpu) as pool:
            for result in pool.map(duster, it.repeat(settings), it.repeat(model_id), zone_ids):
                print(result)
    else:
        duster(settings, model_id, zone_ids[0])
        # for iz in zone_ids:
        #     duster(settings, model_id, iz)

    print("done")





