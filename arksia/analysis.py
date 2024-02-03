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


