##
## Code used to calculate the size distribution of dust grains from the output of the dust code. This version has been optimized and structured to run all models on a single node with 36 cores until all models are run. The outputs are saved to individual files. An additional file is needed to stich the output of this code into one file.
## Sarah Stangl 2019
##

import h5py as h5
from dust_utils import *
from IPython import embed
import numpy as np
from multiprocessing import Pool




# Make sure the path for this is correct, otherwise you're going to have problems
nozawa = 'nozawa2003.dat'


# read in dust dictionary to get information, such as key species list for each grain
# dust grain species list, etc.
dustdict = read_in_nozawa(nozawa)


# Get list of key species, remove some unnecessary substring --> '(g)'
ks = dustdict['key_species']
key_species = np.array([s.replace('(g)','') for s in ks])


# List of all grain species
grain_species = dustdict['grains']

grain_N = grain_species.shape[0]


# Why did I do this?
grain_plot = np.array([s.replace('(s)','') for s in grain_species])


counter = 0
# Need to fix this in dust code saving part, but grain species list needs to be
# corrected for Mg2SiO4 which comes in 2 flavors, one for Mg key species and the
# other for Si key species.......
for i in range(grain_N):
	if grain_species[i] == 'Mg2SiO4(s)' and counter == 0:
		grain_species[i] = grain_species[i]+'-Mg'
		grain_plot[i] = grain_plot[i]+'-Mg'
		counter += 1
    
	elif grain_species[i] == 'Mg2SiO4(s)' and counter == 1:
		grain_species[i] = grain_species[i]+'-SiO'
		grain_plot[i] = grain_plot[i]+'-SiO'
		counter += 1


# Need to remember to set this up in dust code to save this
# Save everything as dictionaries?
gas_species = np.array(['C','Si','SiO','O',"Mg","Al","Fe","S","Ti","V",'Cr','Co','Ni','Cu'])
gas_N = gas_species.shape[0]




def run(gid):
	
	grain_sizes = np.array([])
	if gid == 21:
		return
	fn = '/net/scratch4/sarahstangl/dust_data/'+str(gid)+'_stripped.hdf5'
	#fn = '/turquoise/users/sarahstangl/dust_data/'+str(gid)+'_model_num_dust.hdf5'
	if os.path.isfile(fn):
		h = h5.File('/net/scratch4/sarahstangl/output/'+str(gid)+'_size_dist_dataTEST.hdf5','w')
		id_set = h.create_group(str(gid))
		dset = h5.File(fn,'r')
		for grain_index,grain_name in enumerate(grain_species):
			cell_sizes = np.array([])
			for idx,pid in enumerate(dset):
				
				pset = dset[pid]
				times = pset['time'][()]
				k1_values = pset[grain_name]['grain_radius'][()]
				dadt = pset['dadt']
				n_crit = pset['ncrit']
				Is = pset['Js']
				i = np.nonzero(k1_values)
				if len(i[0]) == 0:
					continue
				start = i[0][0]
				grow = np.where(k1_values != np.roll(k1_values,1))
				if k1_values[0] == 0:
					grow = np.delete(grow,0)
				deltaT = [times[x] -times[x-1] for x in range(len(times))]
				deltaT[0]=0.0
				sizes = np.zeros(len(times))
			
				for spot in grow:
					sizes[spot] = Is[int(spot)][grain_index] * (n_crit[int(spot)][grain_index] ** (1/3))
				#need to grow this for each previous set
				
				#for index in xrange(start,len(times)):
				#	sizes[:index] = [x + dadt[index][grain_index] * deltaT[index] if x !=0 else 0 for x in sizes[:index]]
				#[np.put(sizes,[np.arange(0,index,1)],[x + dadt[index][grain_index] * deltaT[index] if x !=0 else 0 for x in sizes[:index]]) for index in range(start,len(times))]
				#cell_sizes = np.append(cell_sizes,sizes)


				for index in range(len(times)):
					np.put(sizes,np.arange(0,index,1),np.add(sizes[:index],dadt[index][grain_index] * deltaT[index]))
				
				for sizeIdx in range(len(sizes)):
					if np.isin(sizeIdx,grow):
						cell_sizes = np.append(cell_sizes,sizes[sizeIdx])


			id_set.create_dataset(str(grain_name),data=cell_sizes)
		dset.close()
		h.close()


if __name__ == '__main__':
     p = Pool(processes=36)
     p.map(run, list(range(72)))

