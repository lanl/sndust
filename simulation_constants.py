import numpy as np

# numerical constants
onethird = 1. / 3.
twothird = 2. / 3.
twopi = 2. * np.pi
fourpi = 2. * twopi
fourpiover3 = fourpi / 3.
fourover27 = 4. / 27.
onehalf = 1. / 2.

div6 = 1./6.
div12 = 1./12.
div20 = 1./20.

MIN_CONCENTRATION = 1.0E-20
MAX_REACTANTS = 4
MAX_PRODUCTS  = 4
N_MOMENTS = 4

# controls if @jit compile in debug mode
jit_kwargs = {
    "debug" : True,
    "nopython": False,
    "parallel": False,
    "fastmath": False,
    "boundscheck": False,
    "error_model": "python"
}

# testing files
hydro_testing_file = "/home/mauneyc/scratch/students/dust/utils/binary_chk/mdl2.dat"
comp_testing_file = "/home/mauneyc/scratch/students/dust/utils/composition_PM1.50E+01_XE1.69E+00_ID002.dat"
dust_testing_file = "temporary_dust_data.dat"
