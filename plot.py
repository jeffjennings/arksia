"""Routines for plotting pipeline results."""

import json 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d 

from frank.utilities import UVDataBinner, sweep_profile, generic_dht, jy_convert
from mpol.plot import get_image_cmap_norm
from mpol.gridding import DirtyImager

from io import get_vis, load_fits_image, load_bestfit_profiles, load_bestfit_frank_uvtable
from image_radial_profile import radial_profile_from_image
from radial_pipeline import model_setup

def plot_image(image, extent, ax, norm=None, cmap='inferno', title=None, 
                cbar_label=None, cbar_pad=0.1):
    """Plot a 2D image, optionally with colorbar. Args follow `plt.imshow`."""
    if norm is None:
        norm = get_image_cmap_norm(image)

    im = ax.imshow(
        image,
        origin="lower",
        interpolation="none",
        extent=extent,
        cmap=cmap,
        norm=norm
    )    
    
    if cbar_label is not None:
        cbar = plt.colorbar(im, ax=ax, location="left", pad=cbar_pad, shrink=0.7)
        cbar.set_label(cbar_label)
    ax.set_title(title)  


def profile_comparison_figure(fits, model):
    """
    Generate a figure comparing clean, rave, frank radial brightness and visibility profiles.

    Parameters
    ----------
    fits : nested list
        Clean, rave, frank profiles to be plotted. Output of `io.load_bestfit_profiles`
    model : dict
        Dictionary containing pipeline parameters
        
    Returns
    -------
    fig : `plt.figure` instance
        The generated figure
    """
        
    print('making profile comparison figure')

    # load best-fit profiles
    [[rc, Ic, Iec], [grid, Vc]], [[rr, Ir, Ier_lo, Ier_hi], [grid, Vr]], [[rf, If, Ief], [grid, Vf], sol] = fits

    fig = plt.figure(figsize=(10,6))
    fig.suptitle("{} -- robust = {} for clean and rave".format(
        model["base"]["disk"],
        model["base"]["bestfit_robust"])
        )

    gs = GridSpec(4, 2, figure=fig, hspace=0, left=0.09, right=0.97, top=0.94, bottom=0.09)

    ax0 = fig.add_subplot(gs[:3, 0])
    ax3 = fig.add_subplot(gs[3, 0])

    ax1 = fig.add_subplot(gs[:3, 1])
    ax2 = fig.add_subplot(gs[3, 1])

    cols, marks, labs = ['C1', 'C3', 'C2'], ['.', 'x', '+'], ['clean', 'rave', 'frank']
    # plot clean, rave, frank I(r) in mJy / arcsec^2
    Is_jy_sr = [Ic, Ir, If]
    # rave fits have dfft lower and upper uncertainties; clean and frank don't
    Ies_jy_sr = [[Iec, Iec], [Ier_lo, Ier_hi], [Ief, Ief]]
    rs = [rc, rr, rf]

    for ii, jj in enumerate(Is_jy_sr):     
        I_mjy_as2 = jy_convert(jj, 'sterad_arcsec2') * 1e3
        ax0.plot(rs[ii], I_mjy_as2, c=cols[ii], label=labs[ii])
    
        Ie_lo_mjy_as2 = jy_convert(Ies_jy_sr[ii][0], 'sterad_arcsec2') * 1e3
        Ie_hi_mjy_as2 = jy_convert(Ies_jy_sr[ii][1], 'sterad_arcsec2') * 1e3
        ax0.fill_between(rs[ii], I_mjy_as2 - Ie_lo_mjy_as2, I_mjy_as2 + Ie_hi_mjy_as2, 
                         color=cols[ii], alpha=0.4)

    # plot clean, rave, frank Re(V), binned vis.
    [u, v, vis, weights] = get_vis(model)

    # deproject vis
    up, vp, Vp = sol.geometry.apply_correction(u, v, vis)
    bls = np.sqrt(up**2 + vp**2)

    # plot 1d rave residual brightness 
    if model["clean"]["robust"] == 0.5:
        rave_str = "1"
    else:
        rave_str = "2"    
    if model['base']['disk'] == "HD161868" and rave_str == "2":
        raveN = 7
    else: 
        raveN = 5        
    rave_resid_im_path = "{}/{}-{}_inc=90_N={}_2Dresiduals.npy".format(
        model["base"]["rave_dir"], 
        model["base"]["disk"], 
        rave_str,
        raveN
        )    
    
    rave_resid_im = np.load(rave_resid_im_path)
    # convert Jy / pixel to Jy / arcsec
    rave_resid_im /= model["rave"]["pixel_scale"] ** 2 

    phis_mod = np.linspace(model["base"]["geom"]["PA"] - 180, 
                                  model["base"]["geom"]["PA"] + 180,
                                  model["clean"]["Nphi"] 
                                  )
    
    rave_resid_r, rave_resid_I = radial_profile_from_image( 
        rave_resid_im, geom=model["base"]["geom"], 
        rmax=max(rr), Nr=len(rr), phis=phis_mod, 
        npix=rave_resid_im.shape[0], pixel_scale=model["rave"]["pixel_scale"],
        bmaj=0, bmin=0, image_rms=0, model_image=True, arcsec2=False
        )

    ax3.plot(rave_resid_r, rave_resid_I * 1e3, c='C3') 
            
    # plot 1d frank residual brightness at same pixel scale, robust weighting value as clean image
    [ufres, vfres, Vfres, wfres] = load_bestfit_frank_uvtable(model, resid_table=True)

    imager = DirtyImager.from_image_properties(
        cell_size=model["clean"]["pixel_scale"],
        npix=model["clean"]["npix"],
        uu=ufres / 1e3,
        vv=vfres / 1e3,
        weight=wfres,
        data_re=Vfres.real,
        data_im=Vfres.imag,
        )
    frank_resid_im, _ = imager.get_dirty_image(weighting="briggs", 
                                               robust=model["base"]["bestfit_robust"])
    frank_resid_im = np.squeeze(frank_resid_im)

    phis_mod = np.linspace(model["base"]["geom"]["PA"] - 180, 
                                  model["base"]["geom"]["PA"] + 180,
                                  model["clean"]["Nphi"] 
                                  )
    
    frank_resid_r, frank_resid_I = radial_profile_from_image(
        frank_resid_im, geom=model["base"]["geom"], 
        rmax=model["frank"]["Rmax"], Nr=model["frank"]["N"], phis=phis_mod, 
        npix=model["clean"]["npix"], pixel_scale=model["clean"]["pixel_scale"],
        bmaj=0, bmin=0, image_rms=0, model_image=True, arcsec2=False
        )
           
    ax3.plot(frank_resid_r, frank_resid_I * 1e3, c='C2')

    # bin observed visibilities
    bin_cols, bin_marks = ['k', "#a4a4a4"], ["+", "x"]
    for ii, jj in enumerate(model["plot"]["bin_widths"]):
        bin_vis = UVDataBinner(bls, Vp, weights, jj)
        
        # plot binned Re(V)
        ax1.plot(bin_vis.uv / 1e6, bin_vis.V.real * 1e3, c=bin_cols[ii],
                    marker=bin_marks[ii], ls='None',
                    label=r'Obs., {:.0f} k$\lambda$ bins'.format(jj / 1e3)
                    )
        
    # plot vis fits
    for ii, jj in enumerate([Vc, Vr, Vf]):
        ax1.plot(grid / 1e6, jj * 1e3, c=cols[ii], label=labs[ii])

    # plot clean, rave, frank binned residuals.
    # bin vis fits for residual calculation
    bin_vis = UVDataBinner(bls, Vp, weights, model["plot"]["bin_widths"][-1])    
    _, bin_Vc = generic_dht(rc, Ic, Rmax=sol.Rmax, N=sol._info["N"], 
        grid=bin_vis.uv, inc=0)
    _, bin_Vr = generic_dht(rr, Ir, Rmax=sol.Rmax, N=sol._info["N"], 
        grid=bin_vis.uv, inc=0)
    bin_Vf = sol.predict_deprojected(bin_vis.uv, I=If)

    resid_yscale_guess = []
    for ii, jj in enumerate([bin_Vc, bin_Vr, bin_Vf]):
        # bin vis fit residuals
        resid = bin_vis.V.real - jj
        rmse = np.sqrt(np.mean(resid ** 2))

        resid_yscale_guess.append(resid.mean() + resid.std())

        # plot fit residuals
        ax2.plot(bin_vis.uv / 1e6, resid * 1e3, c=cols[ii], 
                          marker=marks[ii], ls='None',
                        label=r'RMSE {:.3f} mJy'.format(rmse * 1e3))

    # symmetric y-bounds for residuals
    resid_yscale = np.array([resid_yscale_guess]) * 1e3
    ax2.set_ylim(-resid_yscale.max(), resid_yscale.max())

    resid_yscale_I = np.max((np.abs(frank_resid_I).max(), np.abs(rave_resid_I).max())) * 1e3
    ax3.set_ylim(-resid_yscale_I * 1.1, resid_yscale_I * 1.1)

    ax3.set_xlim(ax0.get_xbound())
    ax3.set_ylim()
    ax0.legend()
    ax3.set_xlabel(r'r [arcsec]')
    ax0.set_ylabel(r'I [mJy arcsec$^{-2}$]')
    ax3.set_ylabel(r'Resid. I [mJy arcsec$^{-2}$]')

    ax1.legend(loc=1)
    ax2.legend(fontsize=6, loc=1)
    ax2.set_xlabel(r'Baseline [M$\lambda$]')
    ax1.set_ylabel('Re(V) [mJy]')
    ax2.set_ylabel('Resid. [mJy]')
    
    for ax in [ax1, ax2]:
        ax.set_xlim(0, max(bls) * 1.05 / 1e6)
    ax1.tick_params(labelbottom=False)   

    for ax in [ax0, ax1, ax2, ax3]:
        ax.axhline(0, c='c', ls='--', zorder=10)

    plt.savefig('{}/profile_compare_robust{}.png'.format(
        model["base"]["save_dir"], model["base"]["bestfit_robust"]), dpi=300
        )
    
    return fig
