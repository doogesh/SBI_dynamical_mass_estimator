# # Script to generate 3D KDE representation of SDSS clusters

import numpy as np
import tqdm
from tools import *

# Load cluster and galaxy data
M200c = np.loadtxt("SDSSDR13-GALWCLs19_4Mpcph/clusters_M200c_SDSS_GALWCAT19_4Mpcph", skiprows=0, usecols = (1))
M200c = np.array(M200c)

M200c_error = np.loadtxt("SDSSDR13-GALWCLs19_4Mpcph/clusters_M200c_SDSS_GALWCAT19_4Mpcph", skiprows=0, usecols = (2))
M200c_error = np.array(M200c_error)

clusters_dynamics = np.loadtxt("SDSSDR13-GALWCLs19_4Mpcph/galaxies_XYvZ_SDSS_GALWCAT19_4Mpcph", skiprows=0, usecols = (0,1,2,3))

clusters_dynamics[:,0] -= 1

NumClusters = M200c.size

print("Total number of clusters in SDSS-III catalogue")
print(NumClusters)

# Extents of 3D dynamical phase space
x = np.linspace(-4.0, 4.0, 50)
y = np.linspace(-4.0, 4.0, 50)
v = np.linspace(-2200, 2200, 50)

mesh3D = np.meshgrid(x, y, v, indexing='ij')
mesh3D_ = (np.array(mesh3D).reshape(3,-1))


### SDSS DATA SET

phase_space_3D_KDE = []
M200c_list = []
M200c_error_list = []

for i in tqdm.tqdm(range(NumClusters)):

    IDs = np.where(clusters_dynamics[:,0] == i)
    x_ = clusters_dynamics[IDs,1]
    y_ = clusters_dynamics[IDs,2]
    v_ = clusters_dynamics[IDs,3]
    
    KDE_i = compute_KDE(x_[0,:], y_[0,:], v_[0,:], mesh3D_)
    phase_space_3D_KDE.append(KDE_i)
    
    M200c_list.append(M200c[i])
    M200c_error_list.append(M200c_error[i])

np.savez("SDSS_Normalized_4Mpc.npz", phase_space_3D_KDE=phase_space_3D_KDE)

np.savez("SDSS_abdullah_et_al_2019_results_4Mpc.npz", M200c_list=M200c_list, M200c_error_list=M200c_error_list)

NewNumClusters = np.array(M200c_list).size

print("Final number of clusters in SDSS-III data set")
print(NewNumClusters)

print("3D SDSS-III generation completed successfully!")
