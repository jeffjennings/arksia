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


def load_fits_image(fits_image, aux_image=False):
    """
    Load an image from a .fits file.

    Parameters
    ----------
    fits_image : string
        Path to the .fits file
    aux_image : bool, default=False
        Whether the .fits image is an 'auxillary' image (such as a dirty image) 
        or a standard clean output image
        
    Returns
    -------
    image, [bmaj, bmin] : 2D array, list of floats
        The image and its major and min axis widths (only `image` is returned 
        if `aux_image=True`)
    """

    ff = pyfits.open(fits_image)
    image = get_last2d(ff[0].data) 

    print('image {}, brightness unit {}'.format(fits_image, ff[0].header['BUNIT']))

    if not aux_image:
        header = ff[0].header
        bmaj = float(header['BMAJ'])
        bmin = float(header['BMIN'])
        # pixel_scale = float(header['CDELT2']) 
        # npix = len(im[:,0])

        return image, np.array([bmaj, bmin])

    return image 


def get_last2d(image):
    "Get last 2 dimensions of an N-dimensional image"
    if image.ndim <= 2:
        return image
    slc = [0] * (image.ndim - 2) + [slice(None), slice(None)]
    return image[tuple(slc)]


def load_bestfit_frank_uvtable(model, resid_table=False):
    """
    Load the frank best-fit 2D reprojected visibilities.

    Parameters
    ----------
    model : dict
        Dictionary containing pipeline parameters
    resid_table : bool, default=False
        Whether the visibilities being loaded are the frank fit residuals 
        or the frank fit itself

    Returns
    -------
    uv_data : list
        frank visibilities: u-coordinates, v-coordinates, visibility amplitudes 
        (Re(V) + Im(V) * 1j), weights    
    """    
    if resid_table:
        suffix = 'resid'
    else:
        suffix = 'fit'

    # enforce the best-fit has 0 scale height
    path = "{}/{}_alpha{}_w{}_h0.000_fstar{:.0f}uJy_method{}_frank_uv_{}.npz".format(
                        model["base"]["frank_dir"], 
                        model["base"]["disk"], 
                        model["frank"]["bestfit"]["alpha"],
                        model["frank"]["bestfit"]["wsmooth"], 
                        model["frank"]["fstar"] * 1e6,
                        model["frank"]["bestfit"]["method"],
                        suffix
    )
    
    u, v, V, weights = load_uvtable(path)

    return [u, v, V, weights]

