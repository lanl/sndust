# sndust

### Overview

sndust is a collection of python scripts that model the formation of dust grains in a cooling, expanding vapor. Any environment that has time-series temperature and density data can be used as input, but it was orignally designed to study dust production in core-collapse supernovae (CCSNe).

sndust was developed at Los Alamos National Lab (LANL C19146)

### Important runtime options
The top-level file `runtime_settings.json` includes several parameters that control the integration of the ODE that models the physics.
- `abs_tol`, `rel_tol` control the tolerance of the integration step
- `max_dt` is the maximum allowed timestep
- `*_every` are the output options. `screen_every` is the very much the slowest

### Dependancies
The python packages necessary to build are:
- python v3.7
- numpy
- scipy
- numba
- mpi4py
- matplotlib
- h5py
- periodictable

The provided `environment.yml` file defines an environment that installs these packages using `conda`


```
$> conda env create -f environment.yml
```
### Running
`sndust` can run in either serial or parallel. The runtime configuration must be provided, see `runtime_settings.json` for an example setup

```
$> python main.py -c runtime_settings.json
```

For parallel, invoke using `mpi4py.futures`

```
$> mpiexec -n 4 python -m mpi4py.futures main.py -N 4 -c runtime_settings.json
```


