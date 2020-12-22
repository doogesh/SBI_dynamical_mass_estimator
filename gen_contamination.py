# # Script to generate the contamination for each cluster

# Saves pickle files that contain for each cluster the
# - number of interloper clusters
# - total mass of interloper clusters
# - coordinates of the interloper clusters
# - IDs of the interloper clusters
# - phase-space distance of interloper clusters
# Use these pickle files as input for the notebook 'plot_contamination' to generate figure 7 from the paper.

# Notes:
# Define interloper cluster as cluster inside a cylinder with radius R and length V.
# For clusters below M = 10**14; LOS is the z-direction.
# For clusters above M = 10**14; use the LOS vectors.
# Count a cluster only as an interloper cluster, if its mass is higher than that of the original cluster.
# Running this script might take a few hours.


import numpy as np
import pickle
import time

# Constants
Om = 0.3
G = 4.30091 * 10**(-9)  # Mpc M_sun^-1 (km/s)^2

# Set threshold values for R and V
R = 4.0        # Mpc/h
V = 2200       # km/s
H0 = 70        # km/s/Mpc

# Load the data
clusters = np.loadtxt("Data/clusters_M200c_SAG_v3.0_4Mpch")
ID = clusters[:, 0]
mass = clusters[:, 1]
distance = clusters[:, 2]
x = clusters[:, 3]
y = clusters[:, 4]
z = clusters[:, 5]
los = clusters[:, [6, 7, 8]]
a_snapshot = clusters[:, 9]

# Load the rockstar halo data
halos_93570 = np.loadtxt("Mocks/hlist_0.93570.list_M200c_13.4")
halos_95670 = np.loadtxt("Mocks/hlist_0.95670.list_M200c_13.4")
halos_97810 = np.loadtxt("Mocks/hlist_0.97810.list_M200c_13.4")


def rockstar_velocity_radius(cluster_index):
    """
    Function that takes the cluster index, finds the corresponding halo in the Rockstar catalogue
    with the same coordinates, and returns its velocity vector, r_200c and v_200c.
    """
    a = a_snapshot[cluster_index]
    if a == 0.93570:
        halos = halos_93570
    elif a == 0.95670:
        halos = halos_95670
    elif a == 0.97810:
        halos = halos_97810

    x_halo = np.around(halos[:, 17], 3)
    y_halo = np.around(halos[:, 18], 3)
    z_halo = np.around(halos[:, 19], 3)
    vx_halo = halos[:, 20]
    vy_halo = halos[:, 21]
    vz_halo = halos[:, 22]
    m200_halo = halos[:, 40]  # M_sun / h
    rho_crit = 2.775 * 10 ** 11 * (Om * a**(-3) + (1 - Om))  # h^2 M_sun / Mpc^3
    rho_mean = Om * a**(-3) * rho_crit
    r200_halo = (3 * m200_halo[cluster_index] / (4 * np.pi * 200* rho_crit))**(1/3)  # Mpc / h
    v200_halo = (G * m200_halo[cluster_index] / r200_halo)**0.5  # km/s

    # Find index in Rockstar catalogue
    rockstar_index = (x_halo == x[cluster_index]) & (y_halo == y[cluster_index]) & (z_halo == z[cluster_index])
    # If no halo is found, try combinations of 2 matching coordinates
    if not np.any(rockstar_index):
        if np.any((x_halo == x[cluster_index]) & (y_halo == y[cluster_index])):
            rockstar_index = (x_halo == x[cluster_index]) & (y_halo == y[cluster_index])
        elif np.any((x_halo == x[cluster_index]) & (z_halo == z[cluster_index])):
            rockstar_index = (x_halo == x[cluster_index]) & (z_halo == z[cluster_index])
        elif np.any((y_halo == y[cluster_index]) & (z_halo == z[cluster_index])):
            rockstar_index = (y_halo == y[cluster_index]) & (z_halo == z[cluster_index])
        else:
            if np.any(x_halo == x[cluster_index]):
                rockstar_index = x_halo == x[cluster_index]
            elif np.any(y_halo == y[cluster_index]):
                rockstar_index = y_halo == y[cluster_index]
            elif np.any(z_halo == z[cluster_index]):
                rockstar_index = z_halo == z[cluster_index]
            else:
                return np.nan, np.nan, np.nan

    # Double check if masses agree
    if np.around(np.log10(m200_halo[rockstar_index]), 3) != mass[cluster_index]:
        print("Masses don't agree!")
        return np.nan, np.nan, np.nan

    return np.array([vx_halo[rockstar_index][0], vy_halo[rockstar_index][0], vz_halo[rockstar_index][0]]), \
           r200_halo, v200_halo


# Start timing
start = time.time()

# Create empty arrays
number_interlopers = []
mass_interlopers = []
coordinates_interlopers = []
IDs_interlopers = []
PS_distance = []

# - Loop over every cluster,
# - loop over every cluster again and check whether they are inside the first cluster's cylinder.

# First loop
for c1 in range(len(clusters)):
    # print progress
    if (c1 % 1000) == 0.0:
        print((c1 / len(clusters)) * 100, "%")

    # ---------------------------------------
    # Calculate the new x,y and z axes aligned with the LOS of the cluster
    # For clusters below M = 10**14: LOS = z-direction
    if mass[c1] < 14.0:
        x_new = np.array([1, 0, 0])
        y_new = np.array([0, 1, 0])
        z_new = np.array([0, 0, 1])
    else:
        z_new = los[c1]
        z_new /= np.linalg.norm(z_new)
        # Create random vector
        x_new = np.random.randn(3)
        # Apply Graham-Schmidt process
        x_new -= x_new.dot(z_new) * z_new / np.linalg.norm(z_new) ** 2
        x_new /= np.linalg.norm(x_new)
        # Use cross product to get third orthogonal vector
        y_new = np.cross(z_new, x_new)
        y_new /= np.linalg.norm(y_new)

    # Reset counter and make empty lists for the masses, coordinates and IDs
    interlopers_temp = 0
    mass_temp, coordinates_temp, IDs_temp, PS_temp = [], [], [], []

    # Second loop: Loop over all other clusters to check if they're inside the cylinder
    for c2 in range(len(clusters)):
        # If this is the cluster c1 itself; skip this step
        if x[c2] == x[c1] and y[c2] == y[c1] and z[c2] == z[c1]:
            continue

        # Determine x, y and z distance projected along the line of sight
        difference = np.array([x[c2] - x[c1], y[c2] - y[c1], z[c2] - z[c1]])
        delta_x = np.dot(difference, x_new)
        delta_y = np.dot(difference, y_new)
        delta_z = np.dot(difference, z_new)

        # If distance clusters < R and if velocity difference clusters < V
        if ((delta_x**2 + delta_y**2)**0.5) < R and np.abs(delta_z * H0) < V:

            # Include peculiar velocities from the Rockstar catalogue and calculate their difference
            vz_difference = rockstar_velocity_radius(c2)[0] - rockstar_velocity_radius(c1)[0]
            if np.any(np.isnan(vz_difference)):
                print("NAN for peculiar velocities!")
                continue

            # Projected peculiar velocity difference
            delta_vz = np.dot(vz_difference, z_new)
            # Total velocity difference (including Hubble flow)
            total_velocity = delta_z * H0 + delta_vz

            # Velocity still smaller than threshold V?
            if np.abs(total_velocity) < V:
                if mass[c2] > mass[c1] and [x[c2], y[c2], z[c2]] not in coordinates_temp:
                    # Add to all temporary lists
                    interlopers_temp += 1
                    mass_temp.append(mass[c2])
                    coordinates_temp.append([x[c2], y[c2], z[c2]])
                    IDs_temp.append(ID[c2])

                    # For this interloper, determine the phase-space distance and add to list
                    r200_av = (rockstar_velocity_radius(c1)[1] + rockstar_velocity_radius(c2)[1]) / 2
                    v200_av = (rockstar_velocity_radius(c1)[2] + rockstar_velocity_radius(c2)[2]) / 2
                    PS_temp.append(((total_velocity/v200_av)**2 + (delta_x/r200_av)**2 + (delta_y/r200_av)**2)**0.5)

    if interlopers_temp == 0:
        mass_temp = 0
        coordinates_temp = 0
        IDs_temp = 0
        PS_temp = np.nan

    number_interlopers.append(interlopers_temp)
    mass_interlopers.append(mass_temp)
    coordinates_interlopers.append(coordinates_temp)
    IDs_interlopers.append(IDs_temp)
    PS_distance.append(PS_temp)

# Stop timing and print duration
end = time.time()
duration = end - start
print("Done! Duration: ", np.around(duration/3600, 2), "hours")

# Print lists
print(number_interlopers)
print(mass_interlopers)
print(coordinates_interlopers)
print(IDs_interlopers)
print(PS_distance)

print(" ")

# Save lists as pickle files to preserve their structure
with open("data/interlopers/new_r4_distance_interlopers_number.pickle", 'wb') as p1:
    pickle.dump(number_interlopers, p1)
with open("data/interlopers/new_r4_distance_interlopers_mass.pickle", 'wb') as p2:
    pickle.dump(mass_interlopers, p2)
with open("data/interlopers/new_r4_distance_interlopers_coordinates.pickle", 'wb') as p3:
    pickle.dump(coordinates_interlopers, p3)
with open("data/interlopers/new_r4_distance_interlopers_IDs.pickle", 'wb') as p4:
    pickle.dump(IDs_interlopers, p4)
with open("data/interlopers/new_r4_distance_interlopers_PSdistance.pickle", 'wb') as p5:
    pickle.dump(PS_distance, p5)
