"""Routines for input/ouput (loading/saving products used in pipeline)."""

import os
import numpy as np

import astropy.io.fits as pyfits

from frank.utilities import generic_dht, get_fit_stat_uncer
from frank.io import load_sol, load_uvtable

def concatenate_vis(in1, in2, out=None):
    """
    Concatenate the visibilities from two ascii files; save as one output .npz.

    Parameters
    ----------
    in1, in2 : string
        Filenames of the input visibility tables
    out : string, optional
        Filename of the output table. If not supplied, the name will be set as
        vis_combined.npz

    Returns
    -------
    uv_data : list
        concatenated dataset: u-coordinates, v-coordinates, visibility amplitudes 
        (Re(V) + Im(V) * 1j), weights
    """
    u1, v1, re1, im1, weights1, _ = np.genfromtxt(in1).T
    u2, v2, re2, im2, weights2, _ = np.genfromtxt(in2).T

    u = np.concatenate((u1, u2))
    v = np.concatenate((v1, v2))
    re = np.concatenate((re1, re2))
    im = np.concatenate((im1, im2))
    vis = re + im * 1j
    weights = np.concatenate((weights1, weights2))

    if out is None:
        parts1 = in1.split('/')
        parts2 = in2.split('/')
        base1 = os.path.splitext(parts1[4])[0]
        base2 = os.path.splitext(parts2[4])[0]
        out = "{}/vis_combined.npz".format(os.path.join(*parts1[:4]))

    np.savez(out,
             u=u, v=v, V=vis, weights=weights,
             units={'u': 'lambda', 'v': 'lambda',
                    'V': 'Jy', 'weights': 'Jy^-2'})

    return [u, v, vis, weights]
