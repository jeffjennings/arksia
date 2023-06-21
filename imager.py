
"""Routines for generating images from visibilities (originally written by Jeff Jennings)."""
import numpy as np 

from mpol.gridding import DirtyImager

def dirty_image(uv_data, model, robust=None):
    """
    Produce a dirty image (2D array) given visibilities

    Parameters
    ----------
    uv_data : list
        visibilities: u-coordinates, v-coordinates, visibility amplitudes 
        (Re(V) + Im(V) * 1j), weights   
    model : dict
        Dictionary containing pipeline parameters
    robust : float, default=None
        Robust weighting parameter. If None, 'model["clean"]["robust"]' will be used
        
    Returns
    -------
    im : array
        Dirty image
    """    
    if robust is None:
        robust = model["clean"]["robust"]

    # generate dirty image at same pixel scale as clean image
    imager = DirtyImager.from_image_properties( 
        cell_size=model["clean"]["pixel_scale"],
        npix=model["clean"]["npix"],
        uu=uv_data[0] / 1e3,
        vv=uv_data[1] / 1e3,
        weight=uv_data[3],
        data_re=uv_data[2].real,
        data_im=uv_data[2].imag,
        )
    im, _ = imager.get_dirty_image(weighting="briggs", robust=robust)

    return np.squeeze(im)