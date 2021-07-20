import numpy as np
from findShock import *


data = np.genfromtxt('TEMP_hydrorundata.dat',skip_header=1,)

nc = data[:,0]
nt = data[:,1]
ID = data[:,2]
time = data[:,3]
xc = data[:,4]
vc = data[:,5]
mass = data[:,6]
rho = data[:,7]
temp = data[:,8]

numCells = int(nc[0])

time = np.unique(data[:,3])

# we only need the pressure and velocity to find a shock
shock = []

for i in list(range(len(time))):
	rho_t = rho[i*numCells:i*numCells+numCells]
	velo_t = vc[i*numCells:i*numCells+numCells]
	shock.append(list(find_shocks(test(rho_t,velo_t),0.5)))

fout=open('shock_mod2.dat','w')
fout.write('      nc       nt       id            time              xc              vc            mass             rho     temperature    shock\n')
for i in list(range(len(time))):
	for j in list(range(2048)):
		isShock = 0
		idx = i*2048 +j
		if int(j) in shock[i]:
			isShock = 1
		fout.write(str(int(nc[idx]))+'        '+str(int(nt[idx]))+'      '+str(int(ID[idx]))+'      '+str(time[i])+'      '+str(xc[idx])+'      '+str(vc[idx])+'      '+str(mass[idx])+'      '+str(rho[idx])+'      '+str(temp[idx])+'     '+str(isShock)+'\n')

fout.close()

