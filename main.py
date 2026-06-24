import os, sys, argparse, logging, traceback
import itertools as it
from pickle import UnpicklingError

import numpy as np
# from mpi4py.futures import MPIPoolExecutor

from simulation_parameters import SimulationParameters
from simulation_constants import jit_kwargs
from gas import SNGas
from particle import P_freex, load_particles_freex

# from network import Network
from network import Network

# from stepper import Stepper
from solver import SSolver

np.seterr(invalid="warn", divide="warn")

def print_short_list(my_list, max_length=2):
    print(
        my_list
        if len(my_list) <= max_length
        else my_list[:max_length] + ["..."] + my_list[-max_length:]
    )


def initialize(settings):
    output_d = settings.output.directory
    os.makedirs(settings.output.directory, exist_ok=True)

    net = Network(settings.input.network_filepath)

    particles = load_particles_freex(settings.input.particle_filepath)

    gases = {p.pid: SNGas(settings.physics, p, net) for p in particles}

    jit_kwargs["debug"] = settings.jit.debug
    jit_kwargs["fastmath"] = settings.jit.fastmath
    jit_kwargs["parallel"] = settings.jit.parallel
    jit_kwargs["boundscheck"] = settings.jit.bc
    jit_kwargs["error_model"] = settings.jit.divzero
    jit_kwargs["nopython"] = settings.jit.nopython

    return net, particles, gases


def duster(settings, network, particle, gas):
    # gas = SNGas(settings.physics, particle, network)
    ssolv = SSolver(settings.solver, gas, network)
    temp_term = settings.physics.terminal_temperature
    time_term = gas.time_terminal(temp_term)
    msg = f"P{particle.pid}"
    sol = None
    try:
        #        solv(obs)
        sol = ssolv.solve(particle.time_init, time_term)
        msg += f" finished okay"
    except Exception as e:
        msg += f" did not finish ok\n{repr(e)}"
        print(msg)
        traceback.print_exc(file=sys.stdout)
        #        sys.exit(0)
    #        obs.dump(solv._steps)
    finally:
        with open(settings.output.filepath, "ab") as simf:
            np.save(simf, (particle, sol))
    #        obs.runout(solv._steps, settings["align_tend_value"], res=settings["align_tend_resolution"])
    ## TIMING
    #        print(f"DEBUG={S_DEBUG}, NOPYTHON={S_NOPYTHON}, FASTMATH={S_FASTMATH}, PARALLEL={S_PARALLEL}")
    # print(sol)
    return msg


def simulation(settings, network, particles, gases):
    with open(settings.output.filepath, "wb") as simf:
        np.save(simf, settings)

    plist = []
    if settings.solver.select == "all":
        plist = particles
    else:
        plist = [p for p in particles if p.pid in settings.solver.select]

    for p in plist:
        msg = duster(settings, net, p, gases[p.pid])
        print(msg)


def plotter_3(settings, network, particles, gases):
    lookat = [s for s in net.species if "(" in s]
    lai = [network.sidx(l) for l in lookat]
    print("\n\n")
    cell_ids = []
    cell_rads = []
    cell_results = []
    cell_times = []
    cell_cbs = []
    cell_rhos = []
    cell_vols = []
    cell_nds = []
    cell_rvgs = []
    cell_masses = []
    times = []
    masses = []
    pmas = dict.fromkeys(lookat, [])
    max_mass = -1.0
    max_mass_cell = 0
    # sim data header
    with open(settings.output.filepath, "rb") as simf:
        sol = np.load(simf, allow_pickle=True)
        i = 0
        while True:
            try:
                p, sol = np.load(simf, allow_pickle=True)

                cell_results.append(dict.fromkeys(lookat, 0.0))
                cell_cbs.append(dict.fromkeys(lookat, 0.0))
                cell_nds.append(dict.fromkeys(lookat, 0.0))
                cell_rvgs.append(dict.fromkeys(lookat, 0.0))
                cell_masses.append(dict.fromkeys(lookat, 0.0))

                ylast = sol.y[:, -1]
                tlast = sol.t[-1]

                ssolv = SSolver(settings.solver, gases[i], network)
                ssolv.f(tlast, ylast)

                cell_times.append(tlast)
                cell_rhos.append(gases[i].Density(tlast))
                cell_vols.append(gases[i].mass_0 / cell_rhos[-1])
                cell_ids.append(p.pid)
                cell_rads.append(p.radius)

                totmass = 0.0
                for di, (ls, li) in enumerate(zip(lookat, lai)):
                    #                    if ylast[li + 3] < 1.0e-8:
                    #                        continue
                    cell_results[-1][ls] = ylast[li]
                    cell_cbs[-1][ls] = ssolv._dust_calc[di]["cbar"]
                    cell_nds[-1][ls] = ylast[li] * cell_cbs[-1][ls]
                    cell_rvgs[-1][ls] = ssolv._dust_par[di]["a0"] * np.power(
                        ylast[li + 3] / ylast[li], 1.0 / 3.0
                    )
                    ksidx = ssolv._dust_par[di]["keysp_idx"][
                        ssolv._dust_calc[di]["ks_jdx"]
                    ]
                    cell_masses[-1][ls] = ssolv._dust_par[di]["prod_mass"][0] * (
                        cell_nds[-1][ls] * cell_vols[-1]
                    )
                    print(f"tot: {totmass}, new: {cell_masses[-1][ls]}, s: {ls}")
                    totmass += cell_masses[-1][ls]
                    pmas[ls].append(cell_masses[-1][ls])

                if totmass > max_mass and i != 74:
                    print(f"cell max {i} totmass = {totmass}")
                    max_mass = totmass
                    max_mass_cell = i
                    ys = sol.y
                    ts = sol.t
                    print(ts)
                    masses = dict.fromkeys(lookat, np.zeros(ts.shape[0]))
                    times = []
                    for n in range(ts.shape[0]):
                        t = ts[n]
                        times.append(t)
                        y = ys[:, n]

                        ssolv.f(t, y)
                        for di, (ls, li) in enumerate(zip(lookat, lai)):
                            rho = gases[i].Density(t)
                            vol = gases[i].mass_0 / rho
                            cbs = ssolv._dust_calc[di]["cbar"]
                            nds = y[li] * cbs

                            masses[ls][n] =  ssolv._dust_par[di]["prod_mass"][0] * (
                                    nds * vol
                         )

                    print(masses)
                # cell_times.append(sol.t)
                # solys = sol.y
                #
                # print(solys.shape, times.shape)
                # masses.append(dict.fromkeys(lookat, []))
                #
                # ssolv = SSolver(settings.solver, gases[i], network)
                #
                # for n in range(solys.shape[0]):
                #     for m in range(times.shape[0]):
                #         t = times[m]
                #         ys = solys[n]
                #         ssolv.f(t, ys)
                #         for di, (ls, li) in enumerate(zip(lookat, lai)):
                #             rho = gases[i].Density(t)
                #             vol = gases[i].mass_0 / rho
                #             cbs = ssolv._dust_calc[di]["cbar"]
                #             nds = ys[li] * cbs
                #
                #             masses[-1][ls].append(
                #                 ssolv._dust_par[di]["prod_mass"][0] * (nds * vol)
                #             )
                #
                i = 1 + i

            except Exception as e:
                print(f"end of input, {i}")
                traceback.print_exc(file=sys.stdout)
                break

    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    marker_style = dict(linestyle="solid", markersize=5, markevery=1.0)

    def plot_qn(sp, domain, quan, mark="o"):
        plt.semilogy(domain, quan[sp], label=sp, marker=mark, **marker_style)

    # plt.plot(domain, quan[sp], label=sp, marker=mark, **marker_style)
    # plt.loglog(xs, ys, label=sp, marker=mark, **marker_style)

    #    plot_qn("C(s)", cell_nds)
    #    plot_qn("Al2O3(s)", cell_nds)

    print(f"{len(cell_masses[0])}")
    for n, sp in enumerate(lookat):
        p = []
        x = []
        #        for i in range(len(cell_times)):
        #    p.append(cell_rvgs[i][sp])
        print(len(cell_masses))
        for i in range(len(cell_masses)):
            p.append(cell_masses[i][sp])
            x.append(particles[i].radius)

        #        plot_qn(sp, cell_times, cell_rvgs, list(Line2D.filled_markers)[n])
        # print(f"{pmas[sp]}")
        if n % 3 == 0:
            plt.semilogy(x,p, "s", label=sp)
        elif n % 3 == 1:
            plt.semilogy(x,p, "^", label=sp)
        else:
            plt.semilogy(x,p, "o", label=sp)

    #        plot_qn(sp, cell_times, cell_masses, list(Line2D.filled_markers)[n])
    #    plot_qn("C(s)", cell_rvgs)
    # plot_qn("Al2O3(s)", cell_rvgs)

    # plt.semilogy(cell_ids, cell_rhos, label="density", linestyle="dashed")

    plt.xlabel("radius")
    plt.ylabel("mass")
    plt.title(os.path.basename(settings.input.particle_filepath))
    # plt.ylim(bottom=1e-22)
    plt.legend()
    plt.show()


def plotter_2(settings, network, particles, gases):
    lookat = [s for s in net.species if "(" in s]
    lai = [network.sidx(l) for l in lookat]
    print("\n\n")
    cell_ids = []
    cell_rads = []
    cell_results = []
    cell_times = []
    cell_cbs = []
    cell_rhos = []
    cell_vols = []
    cell_nds = []
    cell_rvgs = []
    cell_masses = []
    # sim data header
    with open(settings.output.filepath, "rb") as simf:
        sol = np.load(simf, allow_pickle=True)
        i = 0
        while True:
            try:
                p, sol = np.load(simf, allow_pickle=True)
                cell_results.append(dict.fromkeys(lookat, 0.0))
                cell_cbs.append(dict.fromkeys(lookat, 0.0))
                cell_nds.append(dict.fromkeys(lookat, 0.0))
                cell_rvgs.append(dict.fromkeys(lookat, 0.0))
                cell_masses.append(dict.fromkeys(lookat, 0.0))

                ylast = sol.y[:, -1]
                tlast = sol.t[-1]

                ssolv = SSolver(settings.solver, gases[i], network)
                ssolv.f(tlast, ylast)

                cell_times.append(tlast)
                cell_rhos.append(gases[i].Density(tlast))
                cell_vols.append(gases[i].mass_0 / cell_rhos[-1])
                cell_ids.append(p.pid)
                cell_rads.append(p.radius)

                for di, (ls, li) in enumerate(zip(lookat, lai)):
                    if ylast[li + 3] < 1.0e-8:
                        continue
                    cell_results[-1][ls] = ylast[li]
                    cell_cbs[-1][ls] = ssolv._dust_calc[di]["cbar"]
                    cell_nds[-1][ls] = ylast[li] * cell_cbs[-1][ls]
                    cell_rvgs[-1][ls] = ssolv._dust_par[di]["a0"] * np.power(
                        ylast[li + 3] / ylast[li], 1.0 / 3.0
                    )
                    ksidx = ssolv._dust_par[di]["keysp_idx"][
                        ssolv._dust_calc[di]["ks_jdx"]
                    ]
                    cell_masses[-1][ls] = ssolv._dust_par[di]["prod_mass"][0] * (
                        cell_nds[-1][ls] * cell_vols[-1]
                    )

                i = 1 + i
            except Exception as e:
                print(f"end of input, {i}")
                traceback.print_exc(file=sys.stdout)
                break

    from pathlib import Path

    p = Path(settings.output.directory)
    # datoutf = os.path.join("/Users/mauneye/devel/scratch/sciscratch/wd_rundir_20240122/",f"asci_{p.name}")
    datoutf = os.path.join(f"{p.name}", "plotdat")
    cols = "mol cid radius num_dens avg_size mass".split()
    header_str = "".join([f"{col:<15}" for col in cols])
    with open(datoutf, "w") as of:
        print(f"{header_str}", file=of)
        for ls in lookat:
            tmps = ls.replace("(s)", "")
            for idx in range(len(cell_rads)):
                print(
                    f"{tmps:<15}{cell_ids[idx]:<15}{cell_rads[idx]:<15.8E}{cell_nds[idx][ls]:<15.8E}{cell_rvgs[idx][ls]:<15.8E}{cell_masses[idx][ls]:<15.8E}",
                    file=of,
                )

    # sys.exit()

    solm_gs = 1.9891e33
    totmass = dict.fromkeys(lookat, 0.0)
    for cid, cm in zip(cell_ids, cell_masses):
        for k, v in cm.items():
            totmass[k] += v

    allmass = 0.0
    for k, v in totmass.items():
        print(f"{k} dust mass = {v / solm_gs} Msol")
        allmass = allmass + v

    print(f"total dust mass = {allmass / solm_gs} Msol")

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    marker_style = dict(linestyle="solid", markersize=5, markevery=1.0)

    def plot_qn(sp, domain, quan, mark="o"):
        xs = []
        ys = []
        for cr, cq in zip(domain, quan):
            if cq[sp] < 1.0e-20:
                continue
            xs.append(cr)
            ys.append(cq[sp])
        plt.semilogy(xs, ys, label=sp, marker=mark, **marker_style)
        # plt.loglog(xs, ys, label=sp, marker=mark, **marker_style)

    #    plot_qn("C(s)", cell_nds)
    #    plot_qn("Al2O3(s)", cell_nds)
    #
    for n, sp in enumerate(lookat):
        if totmass[sp] < 1.0e2:
            continue
        #        plot_qn(sp, cell_times, cell_rvgs, list(Line2D.filled_markers)[n])
        plot_qn(sp, cell_times, cell_masses, list(Line2D.filled_markers)[n])
    #    plot_qn("C(s)", cell_rvgs)
    # plot_qn("Al2O3(s)", cell_rvgs)

    # plt.semilogy(cell_ids, cell_rhos, label="density", linestyle="dashed")
    plt.title(os.path.basename(settings.input.particle_filepath))
    plt.ylim(bottom=1e-22)
    plt.legend()
    plt.show()


def plotter(settings, network, particles, gases):
    lookat = [s for s in net.species if "(" in s]
    resd = dict.fromkeys(lookat, [])
    print(f"plotting...")
    with open(settings.output.filepath, "rb") as simf:
        # sim data header
        np.load(simf, allow_pickle=True)
        with open(os.path.join(settings.output.directory, "Cdust.dat"), "w") as cf:
            with open(
                os.path.join(settings.output.directory, "AL2O3dust.dat"), "w"
            ) as al2o3f:
                with open(
                    os.path.join(settings.output.directory, "SIO2dust.dat"), "w"
                ) as sio2f:
                    with open(
                        os.path.join(settings.output.directory, "MGSIO3dust.dat"), "w"
                    ) as mgsio3f:
                        with open(
                            os.path.join(settings.output.directory, "SICdust.dat"), "w"
                        ) as sicf:
                            for i in range(1700):
                                p, sol = np.load(simf, allow_pickle=True)
                                if sol is None:
                                    continue
                                if sol.status < 0:
                                    continue
                                    # print(f"{sol.t[-1]}")
                                try:
                                    for s in lookat:
                                        sidx = net.sidx(s)
                                        if s == "C(s)":
                                            print(
                                                sidx,
                                                i,
                                                sol.y[sidx, -1],
                                                sol.y[sidx + 1, -1],
                                                sol.y[sidx + 2, -1],
                                                sol.y[sidx + 3, -1],
                                                file=cf,
                                            )
                                        if s == "Al2O3(s)":
                                            print(
                                                sidx,
                                                i,
                                                sol.y[sidx, -1],
                                                sol.y[sidx + 1, -1],
                                                sol.y[sidx + 2, -1],
                                                sol.y[sidx + 3, -1],
                                                file=al2o3f,
                                            )
                                        if s == "SiO2(s)":
                                            print(
                                                sidx,
                                                i,
                                                sol.y[sidx, -1],
                                                sol.y[sidx + 1, -1],
                                                sol.y[sidx + 2, -1],
                                                sol.y[sidx + 3, -1],
                                                file=sio2f,
                                            )
                                        if s == "MgSiO3(s)":
                                            print(
                                                sidx,
                                                i,
                                                sol.y[sidx, -1],
                                                sol.y[sidx + 1, -1],
                                                sol.y[sidx + 2, -1],
                                                sol.y[sidx + 3, -1],
                                                file=mgsio3f,
                                            )
                                        if s == "SiC(s)":
                                            print(
                                                sidx,
                                                i,
                                                sol.y[sidx, -1],
                                                sol.y[sidx + 1, -1],
                                                sol.y[sidx + 2, -1],
                                                sol.y[sidx + 3, -1],
                                                file=sicf,
                                            )
                                            # print(f"{s}({sidx})  {sol.y[sidx,-1]} {sol.y[sidx+3,-1]}")
                                        if sol.y[sidx, -1] > 0:
                                            resd[s].append((i, sol.y[sidx, -1]))

                                #                    for s in lookat:
                                #                        resd[s].append(sol.y[net.sidx(s),-1])
                                except AttributeError:
                                    print(f"idx {i} no valid data")
                                except EOFError:
                                    break

        # for k,v in resd.items():
        #    print(f"{k}: {v}")
        #    print(f"{k}: {sum(v)}")
        import matplotlib.pyplot as plt

        #        for s in lookat:
        xs = []
        ys = []
        for x, y in resd["C(s)"]:
            xs.append(x)
            ys.append(y)
        plt.semilogy(resd["C(s)"][0], resd["C(s)"][1], label=s)
        # print(xs)

        # tt = np.linspace(spec.time_start, spec.time_bound, 2000)
        # z = sol.sol(tt)
        # plt.semilogy(tt, z[7].T)
        #
        # plt.semilogy(sol.t_events[1], sol.y_events[1], 'bo ')
        #
        # plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-keys", action="store_true")
    ap.add_argument("-c", "--configfile", type=str, required=True)
    ap.add_argument("-N", "--ncpu", type=int, default=3)
    ap.add_argument("-P", "--plot", action="store_true")
    ap.add_argument("-S", "--show_parameters", action="store_true")

    args = ap.parse_args()

    simpar = SimulationParameters(args.configfile)
    #    simpar.load(args.configfile)

    net, particles, gases = initialize(simpar)

    if args.show_parameters:
        print(simpar)

    elif args.plot:
        plotter_3(simpar, net, particles, gases)

    else:
        simulation(simpar, net, particles, gases)

        #   if 0:
        # TODO: better schedualing
        # with MPIPoolExecutor(max_workers=args.ncpu) as pool:
        #    for result in pool.map(duster, it.repeat(settings), it.repeat(model_id), zone_ids):
        #        print(result)
        # else:
        # res_msg = duster(settings, model_id, 200)
        # print(res_msg)
        # for iz in zone_ids:
        #     duster(settings, model_id, iz)
    # with open(simpar.output.filepath, 'rb') as simf:
    #     while True:
    #         try:
    #             print(np.load(simf, allow_pickle=True))
    #         except (EOFError, UnpicklingError):
    #             print("All data read")
    #             break
    #
    print("done")
