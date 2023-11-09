"""This module contains functions for loading and saving products in the pipeline 
(written by Jeff Jennings)."""

import os
import numpy as np

import astropy.io.fits as pyfits

from frank.utilities import generic_dht, get_fit_stat_uncer
from frank.io import load_sol, load_uvtable

def concatenate_vis(in_paths, out_path):
    """
    Concatenate the visibilities from multiple files; save as one output .npz file.

    Parameters
    ----------
    in_paths : list of string
        Filenames of the input visibility tables
    out_path : string
        Filename of the output table

    Returns
    -------
    uv_data : list
        concatenated dataset: u-coordinates, v-coordinates, visibility amplitudes 
        (Re(V) + Im(V) * 1j), weights
    """
    u, v, re, im, weights = [], [], [], [], []
    for ii in in_paths:
        try: 
            ut, vt, ret, imt, weightst, _ = np.genfromtxt(ii).T
            u.extend(ut)
            v.extend(vt)
            re.extend(ret)
            im.extend(imt)
            weights.extend(weightst)
        except OSError:
            pass

    u, v, weights = np.array(u), np.array(v), np.array(weights)
    vis = np.add(re, np.multiply(im, 1j))

    np.savez(out_path,
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
        print('    loading combined visibility file {}'.format(combined_vis_path))
        dat = np.load(combined_vis_path)
        u, v, vis, weights = [dat[i] for i in ['u', 'v', 'V', 'weights']]
        uv_data = [u, v, vis, weights] 

    else:
        print('    combined visibility file {} not found. Creating it by combining .txt files.'.format(combined_vis_path))

        # input filenames
        vis_paths = []
        for ii in ['ACA', '12m', '12mSB', '12mLB']:
            vis_paths.append("{}/{}.{}.continuum.fav.tav.{}corrected.txt".format(
                model["base"]["frank_dir"], model["base"]["disk"], ii, model["base"]["SMG_sub"]
                )
            )

        uv_data = concatenate_vis(vis_paths, combined_vis_path)

    return uv_data 


def load_fits_image(fits_image, aux_image=False, verbose=0):
    """
    Load an image from a .fits file.

    Parameters
    ----------
    fits_image : string
        Path to the .fits file
    aux_image : bool, default=False
        Whether the .fits image is an 'auxillary' image (such as a dirty image) 
        or a standard clean output image
    verbose : int, default=0
        Level of verbosity for print statements
        
    Returns
    -------
    image, [bmaj, bmin] : 2D array, list of floats
        The image and its major and min axis widths (only `image` is returned 
        if `aux_image=True`)
    """

    ff = pyfits.open(fits_image)
    image = get_last2d(ff[0].data) 

    if verbose > 0:
        print('        loading .fits image {}, brightness unit {}'.format(fits_image, ff[0].header['BUNIT']))

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


def parse_rave_filename(model, file_type='rave_fit'):
    """Interpret rave filenames for generalizing the loading of rave results.

    Parameters
    ----------
    model : dict
        Dictionary containing pipeline parameters
    file_type : str, default='rave_fit'
        Which type of rave file to load: 
          - 'rave_fit' for a rave brightness profile
          - 'rave_residual_image' for a 2d rave residual image

    Returns
    -------
    file_path : str
        Path to the desired rave file
    """    

    if model["clean"]["robust"] == 0.5:
        rave_str = "1"
    else:
        rave_str = "2"

    # one source has different rave parameter choices
    if model['base']['disk'] == "HD161868" and rave_str == "2":
        raveN = 7
    else: 
        raveN = 5

    if file_type == 'rave_fit':
        # adjust rave fit paths here if neeeded
        file_path = "{}/{}-{}_inc=90_N={}_radial_{}0arcsec.npy".format(
            model["base"]["rave_dir"], 
            model["base"]["disk"], 
            rave_str,
            raveN,
            model["rave"]["pixel_scale"]
            )
    
    elif file_type == 'rave_residual_image':
        file_path = "{}/{}-{}_inc=90_N={}_2Dresiduals.npy".format(
            model["base"]["rave_dir"], 
            model["base"]["disk"], 
            rave_str,
            raveN
            )        

    else: 
        raise ValueError("'file_type' {} must be one of 'rave_fit', 'rave_residual_image'".format(file_type))
    
    return file_path


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


def load_bestfit_profiles(model):
    """
    Load the clean, rave and frank best-fit radial brightness profiles and 
    radial visibility profiles

    Parameters
    ----------
    model : dict
        Dictionary containing pipeline parameters

    Returns
    -------
    profiles : nested list
        - For clean: the brightness profile radial points, brightness values and 
        uncertainties; baselines and visibility amplitudes of the 1D Fourier 
        transform of the brightness profile.
        - For rave: the brightness profile radial points, brightness values and 
        unique lower and upper uncertainties; baselines and visibility 
        amplitudes of the 1D Fourier transform of the brightness profile.  
        - For frank: the brigthness profile radial points, brightness values and 
        uncertainties; baselines and visibility amplitudes of the visibility
        fit; the fit solution object.      
    """

    clean_bestfit = "{}/clean_profile_robust{}.txt".format(
        model["base"]["clean_dir"], model["clean"]["bestfit"]["robust"])
    rc, Ic, Iec = np.genfromtxt(clean_bestfit).T

    rave_bestfit = "{}/rave_profile_robust{}.txt".format(
        model["base"]["rave_dir"], model["clean"]["bestfit"]["robust"])
    rr, Ir, Ier_lo, Ier_hi = np.genfromtxt(rave_bestfit).T

    # enforce the best-fit has 0 scale height
    frank_bestfit = "{}/{}_alpha{}_w{}_h0.000_fstar{:.0f}uJy_method{}_frank_sol.obj".format(
                        model["base"]["frank_dir"], 
                        model["base"]["disk"], 
                        model["frank"]["bestfit"]["alpha"],
                        model["frank"]["bestfit"]["wsmooth"],
                        model["frank"]["fstar"] * 1e6,
                        model["frank"]["bestfit"]["method"],
    )
    sol = load_sol(frank_bestfit)
    rf, If, Ief = sol.r, sol.I, get_fit_stat_uncer(sol)

    # dense grid for visibility profiles
    grid = np.logspace(np.log10(1e3), np.log10(1e6), 10**3)

    # clean visibility profile
    _, Vc = generic_dht(rc, Ic, Rmax=sol.Rmax, N=sol._info["N"], grid=grid,
                            inc=0) # 'inc=0' passed to enforce optically thin assumption
    # rave visibility profile
    _, Vr = generic_dht(rr, Ir, Rmax=sol.Rmax, N=sol._info["N"], grid=grid,
                            inc=0)
    # frank visibility profile
    Vf = sol.predict_deprojected(grid)

    return [[rc, Ic, Iec], [grid, Vc]], \
        [[rr, Ir, Ier_lo, Ier_hi], [grid, Vr]], \
        [[rf, If, Ief], [grid, Vf], sol]