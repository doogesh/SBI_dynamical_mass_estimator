"""
Some useful tools for preprocessing the cluster and galaxy data
and miscellaneous postprocessing routines
Contributors: Doogesh Kodi Ramanah, Nikki Arendse
"""

import numpy as np
import scipy as sp
import scipy.stats as spsi
import tqdm as tqdm

__all__ = ["compute_KDE", "compute_PDF_sigma", "compute_posterior_calibration"]

def compute_KDE(x_, y_, v_, mesh3D_, N=50, bandwidth=0.175, normalized=True):
    """
    Compute the 3D (NxNxN) Gaussian KDE representation of the phase-space distribution
    from the x & y projected galaxy positions and v line-of-sight velocities,
    for a cluster containing D galaxies
    Args:
       x_, y_, v_ (np.array): [D]
    Returns:
       3D KDE representation (np.array): [NxNxN]
    """
    d_ = np.stack([x_, y_, v_], axis=0)
    kde_ = sps.gaussian_kde(d_, bandwidth)
    mesh3D_i = mesh3D_.copy()
    mesh3D_density_i = kde_.evaluate(mesh3D_i)
    mesh3D_density_i = mesh3D_density_i.reshape(N,N,N)

    if normalized:
        mesh3D_density_i /= np.sum(mesh3D_density_i)

    return mesh3D_density_i

def compute_PDF_sigma(cluster_PDF, mass_range):
    """
    Compute the peak and sigma from arbitrary PDFs.
    Must specify the range of values. Both inputs are 
    arrays of same dimension D
    Args:
       cluster_PDF (np.array): [D]
       mass_range (np.array): [D]
    Returns:
       peak (float): max a posteriori estimate
       sig_left (float): lower sigma limit
       sig_right (float): upper sigma limit
    """
    # Normalize PDF and find peak
    cluster_PDF /= np.sum(cluster_PDF)
    peak = mass_range[np.argmax(cluster_PDF)]

    # Define threshold for 1 sigma
    threshold = 0.15865525393

    # Calculate 1 sigma region from the left
    L = 0
    for l in range(len(mass_range)):
        L += cluster_PDF[l]
        if L >= threshold:
            left_index = l
            break

    # Calculate 1 sigma region from the right
    R = 0
    for r in reversed(range(len(mass_range))):
        R += cluster_PDF[r]
        if R >= threshold:
            right_index = r
            break

    sig_left = peak - mass_range[left_index]
    sig_right = mass_range[right_index] - peak

    return peak, sig_left, sig_right

def compute_posterior_calibration(m_true, PDF_, M_list):
    """
    Posterior calibration (arbitrary PDF) for quantile-quantile plots
    for mass estimates of D clusters
    Adapted from Matthew Ho's implementation
    Args:
       m_true (np.array): array [D] containing ground truth masses
       PDF_ (np.array): array [D,M] containing PDFs of all D clusters over mass range M
       M_list (np.array): array [M] encoding range of masses
    Returns:
       empirical percentile (np.array): array [num_p] for num_p percentile bins
    """
    # Number of predictive percentile bins
    num_p = 100
    perc_bins = np.linspace(0, 0.999, num_p)

    # Compute CDF (cumulative distribution function) from PDFs
    CDF_ = []
    for k in range(m_true.size):
        CDF_.append(np.cumsum(PDF_[k,:]/PDF_[k,:].sum()))

    M_perc_bins = np.zeros((np.array(CDF_)[:,0].size, num_p))

    # Get the masses that we're predicting for each percentile bin
    for i in tqdm.tqdm(range(num_p)):
        k = perc_bins[i]
        for m in np.arange(np.array(CDF_)[:,0].size):
            M_perc_bins[m, i] = (M_list[np.where(CDF_[m,:] > k)][0])
    print(M_perc_bins.shape)
    
    # Enumerate how many times true mass falls below the predicted percentile
    num_M_less_p = (np.sum(np.repeat(m_true.reshape(-1,1), num_p, axis=1) < M_perc_bins, axis=0)).astype('float')
    
    return num_M_less_p/len(M_perc_bins) # i.e. divide by N
