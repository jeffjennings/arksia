"""This module contains functions for some basic analysis and associated plotting of pipeline results 
(written by Jeff Jennings)."""

import os
import json 
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d

def h_distribution(h, logev):
    """
    Calculate PDF and CDF distributions of aspect ratio for frank 1+1D fits to a source.
    
    Parameters
    ----------
    h : array
        Aspect ratios sampled in frank 1+1D fits
    logev : array
        log(evidence) values for the corresponding fits

    Returns
    -------
    logp : array
        log probability corresponding to input 'logev'
    [hgrid, logp_fine] : list of arrays
        Dense grids corresponding to 'h' and 'logp', used for interpolation
    [cdf, good_idx] : list of arrays
        Cumulative distribution function of aspect ratios 'h' and indices corresponding
        to 'hgrid' where CDF has unique values
    [h16, h50, h84] : list of float
        16th, 50th, 84th, 99.7th percentiles of 'cdf'
    hmax : float
        Point esimate of best estimate for 'h'
    """    

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
    h16, h50, h84, h99_7 = pct([0.16, 0.5, 0.84, 0.997])

    logp_fine = 10 ** logev_fine
    logp = 10 ** logev

    # point estimate of h 
    hmax = hgrid[logp_fine.argmax()]

    return logp, [hgrid, logp_fine], [cdf, good_idx], [h16, h50, h84, h99_7], hmax


def resolving_belt_width_figure(source_par_f="./pars_source.json",
                                results_base_dir=".",
                                save_f="./frank_resolving_belts.png"):
    """
    Generate a figure showing the previously fitted Gaussian width of 
      belts in frank brightness profiles vs a proxy for the baseline resolution.
    
    Parameters
    ----------
    source_par_f : str
        Path to source parameter file containing entries for sources with Gaussian fits
    results_base_dir : str
        Path to parent directory for pipeline results (pipeline file structure assumed)        
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
        para_sol_f = os.path.join(results_base_dir, f"{dd}/parametric/parametric_fit_gauss.obj")
        with open(para_sol_f, 'rb') as f: 
            para_sol = pickle.load(f)
        sigma.append(para_sol.bestfit_params['sigma'])

        # load the corresponding project visibility dataset
        vis_f = os.path.join(results_base_dir, f"{dd}/vis_combined.npz")
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


def aspect_ratio_trend_figure(fit_summary='./frank_scale_heights.txt',
                              save_f='./frank_aspect_ratio_trends.png'
                              ):
    """
    Generate a figure showing source aspect ratios inferred with frank as a function
      of physical and observational parameters
    
    Parameters
    ----------
    fit_summary : str
        Path to .txt file with aspect ratio fit results       
    save_f: str
        Path to save the figure to

    Returns
    -------
    fig : `plt.figure` instance
        The generated figure
    """     

    names, *_ = np.genfromtxt(fit_summary, dtype='str').T
    _, dist, inc, h16, h50, h84, h99_7 = np.genfromtxt(fit_summary).T

    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    axs = axs.ravel()
    fig.suptitle("Aspect ratio inference with frank 1+1D and Gaussian fits to frank radial belts", fontsize=10)
    kwargs = {'fmt':'.', 'c':'k', 'ecolor':'#a4a4a4'}

    for ii, dd in enumerate(names):
        # load the Gaussian fits to the frank radial brightness profiles
        para_sol_f = f"./{dd}/parametric/parametric_fit_gauss.obj"
        with open(para_sol_f, 'rb') as f: 
            para_sol = pickle.load(f)

        rc = para_sol.bestfit_params['Rc']
        centroid = float(rc / dist[ii])

        sigma = para_sol.bestfit_params['sigma']
        fwhm = np.sqrt(8 * np.log(2)) * sigma
        frac_width = float(fwhm / rc)

        yerr = [np.array([h50[ii] - h16[ii]]), np.array([h84[ii] - h50[ii]])]

        # plot upper limit for fits where P(h) doesn't go to 0 at h=0
        if dd in ['HD109573', 'HD131488', 'HD161868']:
            axs[0].plot(frac_width, h99_7[ii], 'kv')
            axs[1].plot(frac_width, h99_7[ii], 'kv')
            axs[1].annotate(dd, (frac_width + 0.02, h99_7[ii]), fontsize=6)
            axs[2].plot(centroid, h99_7[ii], 'kv')
            axs[3].plot(inc[ii], h99_7[ii], 'kv')
        else:
            axs[0].errorbar(frac_width, h50[ii], yerr, **kwargs)
            axs[1].errorbar(frac_width, h50[ii], yerr, **kwargs)
            axs[1].annotate(dd, (frac_width + 0.02, h50[ii]), fontsize=6)
            axs[2].errorbar(centroid, h50[ii], yerr, **kwargs)
            axs[3].errorbar(inc[ii], h50[ii], yerr, **kwargs)
    
    axs[0].set_xlabel(r"Fractional width = FWHM / $r_{centroid}$")
    axs[1].set_xlabel(r"Fractional width = FWHM / $r_{centroid}$")
    axs[2].set_xlabel(r"Centroid [au]")
    axs[3].set_xlabel(r"Inclination [deg]")    

    for aa in axs:
        aa.set_yscale('log')
        aa.set_ylabel(r"Vertical aspect ratio $h=H/r$")        
    
    print(f"    saving figure to {save_f}")
    plt.savefig(save_f, dpi=300, bbox_inches='tight')

    return fig 