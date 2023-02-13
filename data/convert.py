import h5py as h
import numpy as np

species = ['fake', 'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At']

datFile = h.File('TEMP_initdata.hdf5','r')
data = datFile['002']['initial_nd']

outFile = open('model_2.dat','w')
spec = '\t'.join(species)+'\n'
outFile.write('grid\t'+spec)

numID = 1542

for i in range(numID):
	pid_dat = data[i]
	num_density = [str("{:.6e}".format(dat)) for dat in pid_dat]
	num_data = '\t'.join(num_density)
	outFile.write(str(i+1)+'\t'+num_data+'\n')

datFile.close()
outFile.close()
	
