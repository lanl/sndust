import numpy as np

file_name = 'TEMP_hydrorundata.dat'
data = np.genfromtxt(file_name,skip_header=1)

num_st = int(len(data)/2048)
temp = np.split(data[:,8],num_st)
volume = np.split(data[:,6]/data[:,7],num_st)
time = np.unique(data[:,3])
mass = np.split(data[:,6],num_st)
velo = np.split(data[:,5],num_st)
rho  = np.split(data[:,7],num_st)

outfile = open('star_2.dat','w')

for idx,val in enumerate(time):
	outfile.write(str("{:.7e}".format(val))+'\n')
	for pidx in list(range(1542)):
		outfile.write(str(pidx+1)+'\t'+str("{:.7e}".format(temp[idx][pidx]))+'\t'+str("{:.7e}".format(volume[idx][pidx]))+'\t'+str("{:.7e}".format(mass[idx][pidx]))+'\t'+str("{:.7e}".format(rho[idx][pidx]))+'\t'+str("{:.7e}".format(velo[idx][pidx]))+'\n')

outfile.close()	
	
