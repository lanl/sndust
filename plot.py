import matplotlib.pyplot as plt
import numpy as np
import glob, os
from simulation_constants import *
import h5py as h
import matplotlib.mlab as mlab

files = glob.glob("*.hdf5")

bins = np.zeros(numBins)

for zone in files:
	dataF = h.File(zone,'r')
	keys = list(dataF['root'].keys())
	data = dataF['root'][keys[-1]][-1]
	y = data[10] # or data[-1]
	bins += y
	dataF.close()

print(bins)


