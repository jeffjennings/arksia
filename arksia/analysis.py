"""This module contains functions for some basic analysis and associated plotting of pipeline results 
(written by Jeff Jennings)."""

import json 
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d

def h_distribution(h, logev):
    # interpolate
    h_interp = interp1d(h, logev, kind='quadratic')
    hgrid = np.logspace(np.log10(h[0]), np.log10(h[-1]), 300)
    logev_fine = h_interp(hgrid)

    # normalize
    logev -= logev_fine.max()
    logev_fine -= logev_fine.max()

    # h is log-spaced 
    dh = hgrid * (hgrid[1] / hgrid[0] - 1)

    # cumulative distribution
    cdf = np.cumsum(10 ** logev_fine * dh)
    cdf /= cdf.max()
    cdf, good_idx = np.unique(cdf, return_index=True) # prevent repeat entries of 1.0

    # cumulative dist percentiles
    pct = interp1d(cdf, hgrid[good_idx], kind='quadratic')
    h16, h50, h84 = pct([0.16, 0.5, 0.84])

    logp_fine = 10 ** logev_fine
    logp = 10 ** logev

    # point estimate of h 
    hmax = hgrid[logp_fine.argmax()]

    return hgrid, good_idx, logp, logp_fine, cdf, h16, h50, h84, hmax


def resolving_belt_width_figure(source_par_f="./pars_source.json",
                                para_sol_f=None,
                                vis_f=None,
                                save_f="./frank_resolving_belts.png"):
    """
    Generate a figure showing the previously fitted Gaussian width of 
      belts in frank brightness profiles vs a proxy for the baseline resolution.
    
    Parameters
    ----------
    source_par_f : str
        Path to source parameter file containing entries for sources with Gaussian fits
    para_sol_f : str
        Path to parametric fit '.obj' file. If None, default pipeline save location is used.
    vis_f : str
        Path to '.npz' visibility data file. If None, default pipeline save location is used.
    save_f: str
        Path to save the figure to

    Returns
    -------
    fig : `plt.figure` instance
        The generated figure
    """    

    source_pars = json.load(open(source_par_f, 'r'))

    disk_names = []
    sigma = []
    bl_80 = []

    for dd in source_pars:
        disk_names.append(dd)

        # load the Gaussian fit to the frank profile
        if para_sol_f is None:
            para_sol_f = f"./{dd}/parametric/parametric_fit_gauss.obj"
        with open(para_sol_f, 'rb') as f: 
            para_sol = pickle.load(f)
        sigma.append(para_sol.bestfit_params['sigma'])

        # load the corresponding project visibility dataset
        if vis_f is None:
            vis_f = f"{dd}/vis_combined.npz"
        dat = np.load(vis_f)
        u, v = [dat[i] for i in ['u', 'v']]

        projected_baselines = np.hypot(u, v)
        percentile_80 = np.percentile(projected_baselines, 80)
        bl_80.append(percentile_80)

    fwhm = np.sqrt(8 * np.log(2)) * np.array(sigma)
    bl_80 = np.array(bl_80) / 1e6
    
    fig, ax = plt.subplots()
    fig.suptitle('Belt width vs. baseline resolution proxy')

    ax.scatter(bl_80, fwhm)
    for i, lab in enumerate(disk_names):
        ax.annotate(lab, (bl_80[i], fwhm[i]), fontsize=6)

    ax.set_yscale('log')
    ax.set_xlabel(r'80th percentile of projected baseline dist. [M$\lambda$]')
    ax.set_ylabel(r'FWHM Gauss fit to frank [arcsec]')

    print(f"    saving figure to {save_f}")
    plt.savefig(save_f, dpi=300)

    return fig 

