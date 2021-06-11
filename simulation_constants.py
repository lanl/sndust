import numpy as np

# numerical constants
onethird = 1. / 3.
twothird = 2. / 3.
twopi = 2. * np.pi
fourpi = 2. * twopi
fourpiover3 = fourpi / 3.
fourover27 = 4. / 27.

div6 = 1./6.
div12 = 1./12.
div20 = 1./20.

MIN_CONCENTRATION = 1.0E-20
MAX_REACTANTS = 4
MAX_PRODUCTS  = 4
N_MOMENTS = 4

# testing files
hydro_testing_file = "/home/mauneyc/scratch/students/dust/utils/binary_chk/mdl2.dat"
comp_testing_file = "/home/mauneyc/scratch/students/dust/utils/composition_PM1.50E+01_XE1.69E+00_ID002.dat"
dust_testing_file = "temporary_dust_data.dat"

dTime = 0 # global timestep

numBins = 20 # number of size bins
bin_low = 0 # 10^x in angstroms
bin_high = 5 # 10^x in angstroms
_edges = np.logspace(bin_low, bin_high, numBins + 1) # the 1st bin is _edges[1-1] to _edges[1], the last is _edges[numBins-1] to _edges[numBins] 

NDust = 0
