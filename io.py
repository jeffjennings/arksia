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


def get_vis(model):
    """
    Load (or generate if it does not exist) an ARKS visibility dataset.

    Parameters
    ----------
    model : dict
        Dictionary containing pipeline parameters

    Returns
    -------
    uv_data : list
        dataset: u-coordinates, v-coordinates, visibility amplitudes 
        (Re(V) + Im(V) * 1j), weights
    """

    combined_vis_path = "{}/vis_combined.npz".format(model["base"]["frank_dir"])
    if os.path.isfile(combined_vis_path):
        u, v, re, im, weights, _ = np.genfromtxt(combined_vis_path).T
        vis = re + im * 1j
        uv_data = [u, v, vis, weights] 

    else:
        visACA = "{}/{}.ACA.continuum.fav.tav.{}corrected.txt".format(
            model["base"]["frank_dir"], model["base"]["disk"], model["base"]["SMG_sub"]
            ) 
        vis12m = "{}/{}.12m.continuum.fav.tav.{}corrected.txt".format(
            model["base"]["frank_dir"], model["base"]["disk"], model["base"]["SMG_sub"]
            ) 

        # join visibility datasets from ACA and 12m obs.
        # account for sources missing ACA dataset
        if os.path.isfile(visACA):
            uv_data = concatenate_vis(visACA, vis12m)
        else:
            print("  ACA vis dataset missing --> loading only 12m data")
            u, v, re, im, weights, _ = np.genfromtxt(vis12m).T
            vis = re + im * 1j
            uv_data = [u, v, vis, weights]        

    return uv_data 
