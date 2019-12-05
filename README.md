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
sndust needs numpy, scipy, numba, and mpi4py. A conda package is being put together to make this explicit

