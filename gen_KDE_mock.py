# Script to generate training, validation and tests sets

import numpy as np
import tqdm
from tools import *

### Load cluster and galaxy data
M200c = np.loadtxt("mock/SAG_final_4Mpch/clusters_M200c_SAG_v3.0_4Mpch", skiprows=0, usecols = (1))
M200c = np.array(M200c)

clusters_dynamics = np.loadtxt("mock/SAG_final_4Mpch/galaxies_XYvZ_SAG_v3.0_4Mpch", skiprows=0, usecols = (0,1,2,3))

clusters_dynamics[:,0] -= 1

NumClusters = M200c.size

print("Total number of clusters in catalogue")
print(NumClusters)

trainIDs = np.load('clusters_ID_flat_train_final.npz')["selectID"]
print("Number of clusters in training set")
print(trainIDs.size)

# Extents of 3D dynamical phase space
x = np.linspace(-4.0, 4.0, 50)
y = np.linspace(-4.0, 4.0, 50)
v = np.linspace(-2200, 2200, 50)

mesh3D = np.meshgrid(x, y, v, indexing='ij')
mesh3D_ = (np.array(mesh3D).reshape(3,-1))


### TRAINING SET

phase_space_3D_KDE = []
M200c_list = []

for i in tqdm.tqdm(trainIDs):

    IDs = np.where(clusters_dynamics[:,0] == i)
    x_ = clusters_dynamics[IDs,1]
    y_ = clusters_dynamics[IDs,2]
    v_ = clusters_dynamics[IDs,3]
    
    KDE_i = compute_KDE(x_[0,:], y_[0,:], v_[0,:], mesh3D_)
    phase_space_3D_KDE.append(KDE_i)
    
    M200c_list.append(M200c[i])

np.savez("flat_train_set_Normalized_h0_175_final_4Mpc.npz", M200c_list=M200c_list, phase_space_3D_KDE=phase_space_3D_KDE)

print("Final number of clusters in training set")
print(np.array(M200c_list).size)


### VALIDATION SET

validationIDs = np.load('clusters_ID_validation_final.npz')["validationIDs"]
print("Number of clusters in validation set")
print(validationIDs.size)

phase_space_3D_KDE = []
M200c_list = []

for i in tqdm.tqdm(validationIDs):

    IDs = np.where(clusters_dynamics[:,0] == i)
    x_ = clusters_dynamics[IDs,1]
    y_ = clusters_dynamics[IDs,2]
    v_ = clusters_dynamics[IDs,3]
    
    KDE_i = compute_KDE(x_[0,:], y_[0,:], v_[0,:], mesh3D_)
    phase_space_3D_KDE.append(KDE_i)
    
    M200c_list.append(M200c[i])

np.savez("validation_set_Normalized_h0_175_final_4Mpc.npz", M200c_list=M200c_list, phase_space_3D_KDE=phase_space_3D_KDE)

print("Final number of clusters in validation set")
print(np.array(M200c_list).size)


### TEST SET

testIDs = np.load('clusters_ID_test_final.npz')["testIDs"]

phase_space_3D_KDE = []
M200c_list = []

for i in tqdm.tqdm(testIDs):

    IDs = np.where(clusters_dynamics[:,0] == i)
    x_ = clusters_dynamics[IDs,1]
    y_ = clusters_dynamics[IDs,2]
    v_ = clusters_dynamics[IDs,3]

    KDE_i = compute_KDE(x_[0,:], y_[0,:], v_[0,:], mesh3D_)
    phase_space_3D_KDE.append(KDE_i)

    M200c_list.append(M200c[i])

np.savez("test_set_Normalized_h0_175_final_4Mpc.npz", M200c_list=M200c_list, phase_space_3D_KDE=phase_space_3D_KDE)

print("Final number of clusters in test set")
print(np.array(M200c_list).size)

print("3D generation completed successfully!")
