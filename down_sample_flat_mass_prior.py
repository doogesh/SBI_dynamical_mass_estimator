# Script to generate a training set with flat mass distribution
# Contributors: Radek Wojtak

import numpy as np
from numpy import random
from scipy.interpolate import interp1d

M200c = np.loadtxt("mock/SAG_final/clusters_M200c_SAG_v3.0", skiprows=0, usecols = (1))
M200c = np.array(M200c)
nn_size=np.shape(M200c)[0]

# histogram ~ mass function (dN/dlog M200c)
mf=np.histogram(M200c,bins=30)
mfy=np.log10(mf[0])
mfx=np.zeros(30)
for i in range(0,30):
    mfx[i]=0.5*(mf[1][i]+mf[1][i+1])
mfx[0]=mf[1][0]
mfx[29]=mf[1][30]
mfi=interp1d(mfx, mfy, kind='cubic') # interpolation to evaluate dN/dlogM200c at any mass

# higher th is, bigger subsample is (for th=10, we have ~25 000)
th=6.5

prob=np.random.rand(nn_size)
subid=np.zeros(nn_size)

# drawing subsamples: all selected clusters have subid[]=1., unselected have subid[]=0.
for i in range(0,nn_size):
   prob_loc=prob[i]*10.**(mfi(M200c[i])-2.)
   if prob_loc < th: subid[i]=1.

number=np.sum(subid)

print('The number of selected clusters: ',number.astype(int))

# checking the distribution

newmass=subid*M200c

check=np.histogram(newmass,bins=10,range=[mfx[0],mfx[29]])

print(check)

# get IDs of the selected clusters

selectID=np.zeros((number.astype(int),),dtype=int)
ii=0
for i in range(0,nn_size):
    if subid[i] > 0.5:
        selectID[ii]=i
        ii=ii+1
# saving vector of IDs
np.savez('clusters_ID_flat_train_final.npz', selectID=selectID)
