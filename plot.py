import matplotlib.pyplot as plt
import numpy as np
import glob, os
from simulation_constants import *
import h5py as h
import matplotlib.mlab as mlab

files = glob.glob("output_M002/*1541.hdf5")

bins = np.zeros(numBins)

for zone in files:
	dataF = h.File(zone,'r')
	keys = list(dataF['root'].keys())
	data = dataF['root'][keys[-1]][-1]
	y = data[10] # or data[-1]
	bins += y
	dataF.close()

sizes = np.zeros(numBins)

fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])

for i,val in enumerate(sizes):
	sizes[i] = (edges[i] + edges[i+1])/2.0

tot = np.sum(bins)
bins = bins/tot

plt.xscale('log')
plt.bar(sizes, bins)

plt.show()

