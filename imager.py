
"""Routines for generating images from visibilities (originally written by Jeff Jennings)."""
import numpy as np 

from mpol.gridding import DirtyImager

def dirty_image(vis, model, robust, npix=None, pixel_scale=None, casa_vis=True):
    """
    Produce a dirty image (2D array) given visibilities

    Parameters
    ----------
    vis : list
        visibilities: u-coordinates, v-coordinates, visibility amplitudes 
        (Re(V) + Im(V) * 1j), weights   
    model : dict
        Dictionary containing pipeline parameters
    robust : float
        Robust weighting parameter.
    npix : int, default=None
        Number of pixels in image. If None, 'model["clean"]["npix"]' will be used
    pixel_scale : float, default=None
        Pixel width [arcsec]. If None, 'model["clean"]["pixel_scale"]' will be used
    casa_vis : bool, default=True
        Whether the 'vis' were produced with CASA, in which case their 
        complex conjugate will be taken to image with MPoL
        
    Returns
    -------
    im : array
        Dirty image
    """    
    if npix is None:
        npix = model["clean"]["npix"]
    if pixel_scale is None:
        pixel_scale = model["clean"]["pixel_scale"]

    if casa_vis:
        # MPoL uses the standard baseline convention, CASA doesn't
        vis[2] = np.conj(vis[2])

    # generate dirty image at same pixel scale as clean image
    imager = DirtyImager.from_image_properties( 
        cell_size=pixel_scale,
        npix=npix,
        uu=vis[0] / 1e3,
        vv=vis[1] / 1e3,
        weight=vis[3],
        data_re=vis[2].real,
        data_im=vis[2].imag,
        )
    im, _ = imager.get_dirty_image(weighting="briggs", robust=robust, unit='Jy/arcsec^2')

    return np.squeeze(im)