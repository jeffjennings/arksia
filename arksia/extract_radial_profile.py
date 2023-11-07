"""This module contains functions to obtain a radial brightness profile from a CLEAN image 
(written by Seba Marino and Jeff Jennings)."""

import numpy as np
from scipy.optimize import fsolve
from frank.utilities import jy_convert 

def radial_profile_from_image(image, geom, rmax, Nr, phis, 
                            npix, pixel_scale, bmaj, bmin,
                            image_rms, error_std=False, arcsec2=True, 
                            simple_ellipse=False, pb_image=None, 
                            model_image=False, verbose=0, **kwargs):
    """
    Function to extract radial profile from an image 
    
    image: 2D array
        clean image
    geom : dict
        Source geometry. Keys:
        dRA: float, RA offset in arcsec
        dDec: float, Dec offset in arcsec
        PA: float, PA of the disc in degrees
        inc: float, inclination of the disc in degrees
    rmax: float, maximum radius in arcsec
    Nr: int, number of radial bins
    phis: ndarray, array containing sample of PA's that will be use to calculate average intensity
    image_rms: float, image rms in Jy/beam
    error_std: boolean, whether to use the dispersion to calculate uncertainty or the image rms
    arcsec2: boolean, whether to return profile in units of Jy/arcsec2 or Jy/beam
    simple_ellipse: Whether the arc length is calculated with an ellipse or simple ellipse. 
    pb_image: 2D array
        primary beam image. If None, the primary beam will be considered constant and equal to 1.    
    """
    inc, PA, dRA, dDec = geom["inc"], geom["PA"], geom["dRA"], geom["dDec"]

    if pb_image is None:
        pb_image = np.ones((npix,npix))        
        
    if model_image:
        if verbose > 0:
            print('      model_image = True --> setting image_rms = 0')
        image_rms = 0
        bmaj = 0
        if arcsec2:
            # convert CLEAN .model image from [Jy / pixel] to [Jy / arcsec^2] 
            image = image / pixel_scale ** 2
    else:
        # convert to arcsec
        bmaj = bmaj * 3600
        bmin = bmin * 3600
        beam_area = np.pi * bmaj * bmin / (4 * np.log(2))
        if verbose > 0:
            print("      beam = {:.2f} x {:.2f} arcsec".format(bmaj, bmin))

        if arcsec2:
            if verbose > 0:
                print('converting image to [Jy / arcsec^2]')
            # convert CLEAN image from [Jy / CLEAN beam] to [Jy / arcsec^2]
            image = image / beam_area
            image_rms = image_rms / beam_area
    
    return radial_profile(image, pb_image, geom, rmax, Nr, phis, image_rms, 
                        bmaj, pixel_scale, npix, error_std, arcsec2, 
                        simple_ellipse, model_image)


def radial_profile(image, pb_image, geom, rmax, Nr, phis, image_rms, 
                    bmaj, pixel_scale, npix, error_std=False, arcsec2=True, 
                    simple_ellipse=False, model_image=False, rescale_flux=True, 
                    verbose=0):

    """
    dRA, dDec are RA DEC offsets in arcsec
    PA and inc are PA and inc of the disc in deg
    rmax [arcsec] is the maximum deprojected radius at which to do the azimuthal averaging
    Nr is the number of radial points to calculate
    phis [deg] is an array with uniform spacing that sets the range of PA at which to do the interpolation (0 is north)

    """    
    inc, PA, dRA, dDec = geom["inc"], geom["PA"], geom["dRA"], geom["dDec"] 

    # polar grid
    rs = np.linspace(0, rmax, Nr)
    if PA < 0: 
        PA = PA + 180
    PA_rad = PA * np.pi / 180
    phis_rad = phis * np.pi / 180
    dphi_rad = abs(phis_rad[1] - phis_rad[0])
    Nphi_rad = len(phis_rad)

    ecc = np.sin(inc * np.pi / 180)
    # aspect ratio between major and minor axis (>=1)
    chi = 1 / (np.sqrt(1 - ecc ** 2))
    

    # calculate radial profile 
    Is = np.zeros((Nr, Nphi_rad))
    Is_pb = np.zeros((Nr, Nphi_rad))

    err_count = 0
    err_count_pb = 0 

    for ii in range(Nr):
        for jj in range(Nphi_rad):
            xx, yy = ellipse(dRA, dDec, phis_rad[jj], chi, rs[ii], PA_rad)
            
            ip = -int(xx / pixel_scale) + npix // 2
            jp = int(yy / pixel_scale) + npix // 2
            try:
                Is[ii, jj] = image[jp, ip]
            except IndexError:
                if err_count == 0:
                    print('Warning: radial profile extends beyond image. Padding with nan.')
                err_count += 1
                Is[ii, jj] = np.nan
            if not model_image:
                try:
                    Is_pb[ii, jj] = pb_image[jp, ip]
                except IndexError:
                    if err_count_pb == 0:
                        print('Warning: radial profile extends beyond PB image. Padding with nan.')
                    err_count_pb += 1
                    Is_pb[ii, jj] = np.nan

    if err_count > 0:
        print('  Image padded with {} nan'.format(err_count))
    if err_count_pb > 0:
        print('  PB image padded with {} nan'.format(err_count_pb))

    # radial intensity [Jy/arcsec]
    Ir = np.nanmean(Is, axis=1) 
    if model_image: 
        if arcsec2:
            Ir = jy_convert(Ir, 'arcsec2_sterad')
        return rs, Ir

    Ir_pb = np.zeros(Nr)
    
    for i in range(Nphi_rad):
        Ir_pb = Ir_pb + (image_rms / Is_pb[:,i]) ** 2
    Ir_pb = np.sqrt(Ir_pb / Nphi_rad)

    # number of independent points 
    if simple_ellipse:
        # normalize
        arclength = (Nphi_rad - 1) * dphi_rad * np.sqrt((1 + (1 / chi) ** 2 ) / 2 ) 
    else:
        arclength, _ = arc_length(1, 1 / chi, phis_rad[0] - PA_rad, phis_rad[-1] - PA_rad)

    if verbose > 0:
        print('      arc length = {:.2f} deg'.format(arclength * 180 / np.pi))
    Nindep = rs * arclength / bmaj
    Nindep[Nindep < 1] = 1
    
    if error_std:
        I_err = np.nanstd(Is, axis=1) / np.sqrt(Nindep)
    else:
        I_err = Ir_pb / np.sqrt(Nindep)

    if arcsec2:
        Ir = jy_convert(Ir, 'arcsec2_sterad')        
        I_err = jy_convert(I_err, 'arcsec2_sterad')

    if rescale_flux:
        Ir = Ir * np.cos(inc * np.pi / 180)
        I_err = I_err * np.cos(inc * np.pi / 180)

    return rs, Ir, I_err


def arc_length(a, b, phi1, phi2, Nint=1000000):
    # a is semi-major axis (=1)
    # b is the semi-minor axis (=1/aspect_ratio)
    # returns arc length from phi1 to phi2 which are defined as 0 at the disc PA and grow anticlockwise in the sky

    # works with delta y / delta x instead of dy/dx
   
    # translates phi1 and phi2 to angles between 0 and 2pi
    phi1 = simple_phi(phi1)
    phi2 = simple_phi(phi2)
    # catches error when phi1 and phi2 are the same if one wants to integrate over whole circumpherence
    if phi2 == phi1: 
        phi1 = 0.0
        phi2 = 2 * np.pi

    # Figure out the right ranges of xs to integrate. Is phi=0.0 or phi=180 contained in range?
    phis = separate_intervals(phi1, phi2)
    
    Nph = len(phis)
    Arc_length = 0.0

    for i in range(Nph - 1):
        phi_i = phis[i]
        phi_ii = phis[i + 1]  
 
        x1 = x_phi(phi_i, a, b)
        x2 = x_phi(phi_ii, a, b)
        dx = (x2 - x1) / (Nint - 1)
        xs = np.arange(x1, x2, dx)

        Arc_length += np.sum(Delta_s(xs, a, b))
        
    return Arc_length, phis

    
def separate_intervals(phi1, phi2):
    # Figure out the right ranges of xs to integrate. Does phi=0.0 pr phi=180 is contained in range?
    phis = [phi1]

    if phi1 > phi2: # passes through zero
        # we need to figure out if it passes through 180 first
        if phi1 < np.pi:
            phis.append(np.pi)
        phis.append(0.0)

    if phi2 > np.pi and phis[-1] < np.pi:
        phis.append(np.pi)
    phis.append(phi2)

    return np.array(phis)


def ellipse(x0, y0, phi, chi, a, PA):
    ## phi is a position angle in the plane of the sky

    # x0,y0 ellipse center
    # phi pa at which calculate x,y phi=0 is +y axis
    # chi aspect ratio of ellipse with chi>1
    # a semi-major axis
    # a/chi semi-minor axis
    # PA  pa of ellipse 0 is north and pi/2 is east

    if a==0.:
        return x0*np.ones_like(phi), y0*np.ones_like(phi)
    
    phipp = phi - PA
    xpp = x_phi(np.pi / 2 - phipp, a/chi, a)
    ypp = y_phi(np.pi / 2 - phipp, a/chi, a)

    xp = xpp * np.cos(PA) + ypp * np.sin(PA)
    yp = -xpp * np.sin(PA) + ypp * np.cos(PA)
    
    xc = xp + x0
    yc = yp + y0
    
    return xc, yc


def simple_phi(phi): 
    # returns phi between 0 and 2pi
    if abs(phi) != 2 * np.pi:
        phir = phi % (2 * np.pi)
    else: 
        phir = phi

    if phir < 0:
        phir = 2 * np.pi + phir

    return phir


def x_phi(phi, a, b):
    phi=simple_phi(phi)
    
    if phi <= np.pi / 2 or phi >= 3 / 2 * np.pi:
        sign = 1.0
    else:
        sign = -1.0

    return sign / np.sqrt(np.tan(phi) ** 2 / b ** 2 + 1 / a ** 2)


def y_phi(phi, a, b):    
    phi=simple_phi(phi)
    if phi >= 0 and phi <= np.pi:
        sign = 1.0
    else:
        sign = -1.0
    
    return sign / np.sqrt(1 / b ** 2 + 1 / (a ** 2 * np.tan(phi) ** 2))


def Delta_s(xs, a, b): 
    # delta s over an ellipse
    ys = np.zeros(len(xs))
    # avoid imaginary numbers 
    mask=((xs <= a ) & (xs >= -a)) 

    if xs[-1] < xs[0]:
        ys[mask]= b * np.sqrt(1 - xs[mask] ** 2 / a ** 2)
    else:
        ys[mask]= -b * np.sqrt(1 - xs[mask] ** 2 / a ** 2)

    return np.sqrt((ys[:-1] - ys[1:]) ** 2 + (xs[:-1] - xs[1:]) ** 2)


# We want to calculate the critical angle phi along an ellipse (from its major axis)
# when the radius r(phi) becomes smaller than the semi-major axis (a) by a certain factor f.

# We can find phi by starting from the definition of the ellipse:
#     x2/a2 + y2/b2 =1 
    
# and rewrite x as l(phi)*cos(phi) and y as l(phi)*sin(phi). We can also write b as
# a*cos(inc) (inclined ring). Then the ratio between a and l is 

# a/l(phi)=sqrt(cos(phi)**2+sin(phi)**2/cos(inc)**2)
def stretching(phi, inc):
    """
    phi and inc in rad
    """
    return np.sqrt(np.cos(phi)**2+np.sin(phi)**2/np.cos(inc)**2)

def find_phic(inc, f):
    """
    find critical phi (phic) when the ratio a/l(phi) = f (f>1).
    inc in radians.
    f>1. 
    """
    
    if 1./np.cos(inc)<f:
        print('All phis satisfy condition, using phic=pi/2')
        return np.pi/2. 
              
    # solve equation stretching-f=0
    func= lambda phi: stretching(phi, inc)-f
    phic = fsolve(func, np.pi/4.) # initial guess phic=45deg
    return phic[0]
