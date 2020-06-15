import numpy as np
import sys, os, json, argparse, glob, itertools, pickle
from typing import List
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.interpolate import griddata
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d

import h5py as h5

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tracer import Tracer, read_tracerpick
from fluid_particle import FluidParticle, find_fp_volume
#import matplotlib.gridspec as gridspec
#from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#colors = cm.rainbow(np.linspace(0,1, np.maximum(len(plot_mol), len(plot_atm))+1))

ybound = [1.0E-12, 1.2E0]

def generate_plot(input_hdf: str, settings: dict):

    plot_atm = settings["plot"]["elements"].split()
#    plot_mol = settings["plot"]["molecules_small"].split()
#    plot_molbig = settings["plot"]["molecules_large"].split()

    plt.gca()

    #colors = cm.rainbow(np.linspace(0,1, np.maximum(len(plot_mol), len(plot_atm))+1))
    colors_0 = cm.rainbow(np.linspace(0,1, len(plot_mol)+1))
    colors_1 = cm.rainbow(np.linspace(0,1, len(plot_molbig)+1))

    
    with h5.File(input_hdf5, "r") as hf:
        for tsp in hf["root"]:
            step = tsp
        temperatures 
#    headers = None
#    with open(inputf, "r") as f:
#        lines = f.readlines()
#        headers = lines[0].split(",")

#    headers = [h.replace("Y_","").strip() for h in headers]

    n2idx = 0
    for i, h in enumerate(headers):
        if h == "N2":
            n2idx = i
            break

    headers = headers[:n2idx] + headers[n2idx+1:] + [headers[n2idx]]

    cols = dict([h, i] for i, h in enumerate(headers))

    chemdat = np.loadtxt(inputf, delimiter=',', skiprows=1)
    times = chemdat[:, cols["t"]]
    temperatures = chemdat[:, cols["T"]] / 1.0E3
    densities = chemdat[:, cols["density"]]


    Ymol = [chemdat[:, cols[s]] for s in plot_mol]
    Ymolbig = [chemdat[:, cols[s]] for s in plot_molbig]

    plt.rcParams['figure.figsize'] = 18, 9
    plt.rcParams.update({"font.size": 38})
    plt.rcParams.update({"legend.fontsize": 'large'})

#    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig, ax = plt.subplots(nrows=1, ncols=1)

    i_short = 0
    try:
        while times[i_short] < 5.0:
            i_short += 1
    except:
        i_short = -1

    tshort = times[:i_short]

    for i in range(len(plot_mol)):
        ax.semilogy(times, Ymol[i], label=plot_mol[i], color=colors_0[i], linewidth=3.4, linestyle='solid', marker='o', markevery=200, markersize=7.0)
#        ax[0,0].semilogy(times, Ymol[i], label=plot_mol[i], color=colors_0[i], linewidth=2.2, linestyle='solid', marker='o', markevery=180, markersize=4.2)
#        ax[0,1].semilogy(tshort, Ymol[i][:i_short], label=plot_mol[i], color=colors_0[i], linewidth=2.2, linestyle='solid', marker='o', markevery=180, markersize=4.2)
#    for i in range(len(plot_molbig)):
#        ax[1,0].semilogy(times, Ymolbig[i], label=plot_molbig[i], color=colors_1[i])
#        ax[1,1].semilogy(tshort, Ymolbig[i][:i_short], label=plot_molbig[i], color=colors_1[i])
    #for i in range(len(plot_atm)):
    #    ax.semilogy(times, Yatm[i], label=plot_atm[i], linestyle='--', color=colors[i])

    for i in range(1):
        for j in range(1):
#            a = ax[i,j]
            a = ax
            a.set_ylim(*ybound)
            a.grid()
            if j == 0:
                a.set_ylabel("Y")
                #a.spines['left'].set_visible(False)

            ax2 = a.twinx()
            if j == 1 or j == 0:
                ax2.plot(tshort, temperatures[:i_short], linestyle=':', color='r', linewidth=3.3, label="T")
#                a.set_yticks([])
            else:
                ax2.plot(times, temperatures, linestyle=':', color='r', linewidth=3.3, label="T")
                #ax2.spines['right'].set_visible(False)
                ax2.set_yticks([])
                a.set_yticks([])

            if j == 1 or j == 0:
                ax2.set_ylabel(r"temperature (K) / $10^3$")

                box = a.get_position()
                a.set_position([box.x0, box.y0, box.width * 0.9, box.height])

                # Put a legend to the right of the current axis
#                a.legend(loc='center left', bbox_to_anchor=(1.20, 0.5))
                a.legend(loc='upper right', fontsize='x-small')

                ax2.legend(loc='lower right', fontsize='x-small')

            if i == 1 or i == 0:
                a.set_xlabel("time (s)")
            a.set_xlabel("time (s)")
            
            new_tick_label = ['{:2.1f}'.format(x) for x in [2, 4, 6,8,10]] 
            ax2.set_yticklabels(new_tick_label)

    #fig.legend(loc=7)
    fig.tight_layout()
    ax.set_xlim(left=0.05)
    #ax[1].set_xlabel("time (s)")
    fig.subplots_adjust(right=0.74)

    return fig
    #fig.suptitle(f"chemdat", x=0.88, y=0.98)
    #plt.show()
#    outputf =f"{outdir}//tracer_{tid}.png"
#    print(f"saving {outputf}")
#    fig.savefig(outputf, dpi=200)
    #return

def plot_particle_chem(pfile: str, title: str, settings: dict, output = "screen") -> None:

    fig = generate_plot(pfile, settings)
    #fig.suptitle(f"{title}")
    if output == "screen":
        plt.show()
    else:
        fig.savefig(output, dpi=440)
        plt.close(fig)
    return


def main(input: str, config_f: str, output: str, ff: str, batch: bool = False) -> None:
    settings = dict()
    with open(config_f, 'r') as cf:
        settings = json.load(cf)

    if batch:
        if not os.path.isdir(input):
            raise NotADirectoryError("input is not a directory on batch plot")
        for datf in os.listdir(input):
            prefix = os.path.splitext(os.path.basename(datf))[0]
            outputf = os.path.join(output, f"{prefix}.png")
            inputf = os.path.join(input, datf)
            if os.path.isfile(inputf):
                xy = get_xy0(ff, prefix)
                title=f"{prefix} x_0=[{xy[0]/3.0E6:.2f},{xy[1]/5.0E6:.2f}]"
                print(f"plotting {datf}")
                plot_particle_chem(inputf, title, settings, output=outputf)
    else:
        prefix = os.path.splitext(os.path.basename(input))[0]
        xy = get_xy0(ff, prefix)
        title=f"{prefix} x_0=[{xy[0]/3.0E6:.2f},{xy[1]/5.0E6:.2f}]"
        plot_particle_chem(input, title, settings, output=output)

def get_xy0(pickf: str, tlbl: str):
        with open(pickf, "rb") as pf:
            while True:
                try:
                    t = pickle.load(pf)
                    if t.label == tlbl:
                        return t.xyz[0][0], t.xyz[0][1]
                except:# EOFError:
                    break
        return -1, -1

def file_2d(fp_pick: str, output: str, res_dir: str, follow=["H2O", "H2", "CO", "NO", "NO2", "NNH", "NH3", "HNO"]):

    #fps = [FluidParticle(tr) for tr in read_tracerpick(fp_pick)]
    #find_fp_volume(fps)

    avail_files = os.listdir(res_dir)

    with open(output, "w") as ouf:
        print("preparing file...")
        fslb = " ".join([f"{s:10s}" for s in follow])
        print(f"{'tid':8s} {'time':10s} {'x':10s} {'y':10s} {fslb}", file=ouf)

        with open(fp_pick, "rb") as pf:
            while True:
                try:
                    fp = FluidParticle(pickle.load(pf))
                except EOFError:
                    break

                expected_filename = f"{fp.tlbl}.out"
                print(f"searching for chemistry on {fp.tlbl}...")
                if expected_filename not in avail_files:
                    print(f"could not find data for {fp.tlbl}")
                    continue

                headers = None
                with open(os.path.join(res_dir, expected_filename), "r") as f:
                    line = f.readline()
                    headers = line.split(",")

                headers = [h.replace("Y_","").strip() for h in headers]

                n2idx = 0
                for i, h in enumerate(headers):
                    if h == "N2":
                        n2idx = i
                        break

                headers = headers[:n2idx] + headers[n2idx+1:] + [headers[n2idx]]

                cols = dict([h, i] for i, h in enumerate(headers))
                chemdat = np.loadtxt(os.path.join(res_dir, expected_filename), delimiter=',', skiprows=1)
                times = chemdat[:, cols["t"]]
                Ymol = [chemdat[:, cols[s]] for s in follow]

                for i in range(times.size):
                    pos = fp.XYZ(times[i])
                    fslb = " ".join([f"{y[i]:10.8E}" for y in Ymol])
                    print(f"{fp.tid:8d} {times[i]:10.8E} {pos[0]:10.8E} {pos[1]:10.8E} {fslb}", file=ouf)

def iter_loadtxt(filename, skiprows=0):
    cid = 1
    with open(filename, 'r') as infile:
        for _ in range(skiprows):
            next(infile)

        line = infile.readline()
        headers = line.split()
        yield headers
        blk = None
        for line in infile:
            tok = line.split()
            tid = int(tok[0])
            if tid != cid:
                if blk is not None:
                    yield blk
                blk = { h : [] for h in headers}
                cid = tid

            blk["tid"].append(tid)
            for h, t in zip(headers[1:], tok[1:]):
                blk[h].append(float(t))

    #data = np.fromiter(iter_func(), dtype=dtype)
    #data = data.reshape((-1, iter_loadtxt.rowlength))
    #return data


class animate_helper:
    def __init__(self, dat2d: str, npix: int=1000):
        self._datf = dat2d
        self._npix = npix

        #dat = np.genfromtxt(self._datf, names=True)

        #self._ids = np.unique(dat["tid"])[::4]
        #self._ntids = self._ids.size

        #self._fields = list(dat.dtype.fields.keys())[2:]
        dat = iter_loadtxt(dat2d)
        self._fields = next(dat)[2:]
        self._intrps = []
        for blk in dat:
            print(f"constructing intrp for {blk['tid'][0]}")
            self._intrps.append({ fld : interp1d(blk["time"], blk[fld], bounds_error=False, fill_value=-20, copy=False) for fld in self._fields })

        self._ntids=len(self._intrps)
    def __call__(self, time: float, mol: str):

        p, z = self._get_pntz(time, mol)
        return p, z
        #gridx, gridy = np.mgrid[np.min(p[:,0]):np.max(p[:,0]):1000j, np.min(p[:,1]):np.max(p[:,0]):1000j]

        #grid = griddata(p, z, (gridx, gridy), method='cubic', rescale=False)
        #return grid

    def _get_pntz(self, time, mol):
        p = np.zeros((self._ntids, 2))
        z = np.zeros(self._ntids)

        for n in range(self._ntids):
            p[n,0] = self._intrps[n]["x"](time)
            p[n,1] = self._intrps[n]["y"](time)
            z[n] = np.maximum(self._intrps[n][mol](time), 1.0E-20)

        #idz = np.where([z > 0])[0]
        #z[idz] = np.log10(z[idz])
        z = np.log10(z)

        return p, z

    #fp_x0 = np.array([ fp.XYZ(fp.t_left) for fp in fps ])[:,0:2]

    #x = fp_x0[:, 0]
    #y = fp_x0[:, 1]



    #ds = np.log10(np.array([ fp.D(fp.t_left) for fp in fps]))

    #gridx, gridy = np.mgrid[np.min(x):np.max(x):1000j, np.min(y):np.max(y):1000j]

    #grid = griddata(fp_x0, ds, (gridx, gridy), method='nearest', rescale=True)

    #plt.imshow(grid.T, extent=(0,1,0,1), origin="lower")
    #plt.show()

if __name__ == "__main__":
    par = argparse.ArgumentParser()
    par.add_argument("-i", "--input", type=str, help="input file or directory", required=True)
    par.add_argument("-c", "--config", type=str, help="plot configurations", default=None)
    par.add_argument("-o", "--output", help="output locate ('screen', or directory)", type=str, default="screen")
    par.add_argument("-b", "--batch", action="store_true", help="run on all files in directory")
    par.add_argument("--make2Ddat", action="store_true", help="make the data file for 2d viewing")
    par.add_argument("--fluidfile", type=str, default=None)
    par.add_argument("--movie", action="store_true")


    args = par.parse_args()

    if args.make2Ddat:
        if args.fluidfile is None:
            print("error, need --fluidfile")
        file_2d(args.fluidfile, args.output, args.input)
    elif args.movie:

        plt.rcParams['figure.figsize'] = 28, 4
        plt.rcParams.update({"font.size": 14})

        ahpf="__ah_tmp.pickle"
        if not os.path.exists(ahpf):
            ah = animate_helper(args.input)
            with open(ahpf, "wb") as pf:
                pickle.dump(ah, pf)
        with open(ahpf, "rb") as pf:
            ah = pickle.load(pf)

        #g = ah(10.0E0, "H2O")
        cm = plt.cm.get_cmap('Reds')
        mol = "CO"
        ts = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0 ]
        fig, axs = plt.subplots(nrows=1, ncols=len(ts))
        divider = make_axes_locatable(axs[-1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = None
        for i, t, ax in zip(range(len(ts)), ts, axs):
            p, z = ah(t, mol)

            im = ax.scatter(p[:,0] * 1.0E-5, p[:,1] * 1.0E-5, c=z, vmin=-16.0, vmax=0.0, s=10, cmap=cm, alpha=0.3)
            #gridx, gridy = np.mgrid[0:3*10E5:2000j,0:5.5*10E5:2000j]

            #grid = griddata(p, z, (gridx, gridy), method='cubic', rescale=False, fill_value=np.min(z))

            #ax.imshow(grid.T, extent=(0,1,0,1), origin="lower")

            ax.set_title(f"t = {t} s")
            if i != 0:
                ax.set_yticks([])

        fig.colorbar(im, cax=cax, orientation='vertical')
        fig.suptitle(f"{mol}")
        plt.show()
        #plt.imshow(g.T, extent=(0,1,0,1), origin="lower", cmap='jet', vmin=-10, vmax=0)
        #plt.colorbar()
        #plt.show()
    else:
        main(args.input, args.config, args.output, args.fluidfile, args.batch)
