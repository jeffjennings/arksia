"""This module contains functions for plotting pipeline results 
(written by Jeff Jennings)."""

import json 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d 

from frank.utilities import UVDataBinner, make_image, sweep_profile, generic_dht, jy_convert
from mpol.plot import get_image_cmap_norm

from arksia.input_output import get_vis, load_bestfit_frank_uvtable, load_fits_image, parse_rave_filename
from arksia.extract_radial_profile import radial_profile_from_image
from arksia.imager import dirty_image

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


def clean_diagnostic_figure(model, clean_image, clean_profile, 
                            model_image=None, model_profile=None):

    print('  Figures: making clean diagnostic figure')

    fig = plt.figure(figsize=(10,6))
    fig.suptitle("{} -- robust = {}".format(model["base"]["disk"],
                                            model["clean"]["robust"])
                                            )
    
    gs = GridSpec(2, 2, figure=fig, hspace=0, left=0.09, right=0.97, top=0.94, bottom=0.19)
    gs1 = GridSpec(2, 1, figure=fig, hspace=0, left=0.09, right=0.97, top=0.9, bottom=0.09)

    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs1[1])
    
    # plot clean image
    clean_im_xmax = model["clean"]["pixel_scale"] * model["clean"]["npix"] / 2
    clean_extent = [clean_im_xmax, -clean_im_xmax, -clean_im_xmax, clean_im_xmax]

    plot_image(clean_image * 1e3, clean_extent, ax0, 
            #    norm=get_image_cmap_norm(clean_image * 1e3, stretch='asinh'),               
               cbar_label='$I_{clean}$ [mJy arcsec$^{-2}$]'
               )

    # plot clean model image
    if model_image is not None:
        plot_image(model_image * 1e3, clean_extent, ax1, cmap="Reds",
                norm=get_image_cmap_norm(model_image * 1e3, stretch='asinh'), 
                cbar_label='$I_{clean\ model}$ [mJy arcsec$^{-2}$]'
                )    
    
    # plot radial brightness profiles
    if model_profile is not None:
        ax2.plot(model_profile[0], model_profile[1], '#a4a4a4', label='.model')

    ax2.plot(clean_profile[0], clean_profile[1], 'r', label='.image')
    ax2.fill_between(clean_profile[0], 
                     clean_profile[1] - clean_profile[2], 
                     clean_profile[1] + clean_profile[2], 
                     color='r', alpha=0.4
                     )
    
    ax2.set_xlabel('r ["]')
    ax2.set_ylabel('I [Jy / sr]')
    ax2.legend()
    
    ff = f"{model['base']['clean_dir']}/clean_profiles_robust{model['clean']['robust']}.png"
    print('    saving figure to {}'.format(ff))
    plt.savefig(ff, dpi=300)

    return fig 


def profile_comparison_figure(fits, model, resid_im_robust=2.0, npix=1000, include_rave=True):
    """
    Generate a figure comparing clean, frank, and optionally rave radial brightness and visibility profiles.

    Parameters
    ----------
    fits : nested list
        Clean, frank, rave profiles to be plotted. Output of `input_output.load_bestfit_profiles`
    model : dict
        Dictionary containing pipeline parameters
    resid_im_robust : float, default=2.0
        Robust weighting parameter used for imaging frank residual visibilities.
    npix : int
        Number of pixels along one axis used to make frank images
    include_rave : bool, default=True
        Whether to include rave results in figure 

    Returns
    -------
    fig : `plt.figure` instance
        The generated figure
    """
        
    print('  Figures: making profile comparison figure')
    
    # load best-fit profiles
    if include_rave is True:
        [[rc, Ic, Iec], [grid, Vc]], [[rr, Ir, Ier_lo, Ier_hi], [grid, Vr]], [[rf, If, Ief], [grid, Vf], sol] = fits
    else:
        [[rc, Ic, Iec], [grid, Vc]], _, [[rf, If, Ief], [grid, Vf], sol] = fits

    fig = plt.figure(figsize=(10,6))
    fig.suptitle("{} -- robust = {} for clean and rave".format(
        model["base"]["disk"],
        model["clean"]["bestfit"]["robust"])
        )

    gs = GridSpec(4, 2, figure=fig, hspace=0, left=0.09, right=0.97, top=0.94, bottom=0.09)

    ax0 = fig.add_subplot(gs[:3, 0])
    ax3 = fig.add_subplot(gs[3, 0])

    ax1 = fig.add_subplot(gs[:3, 1])
    ax2 = fig.add_subplot(gs[3, 1])

    cols, marks, labs = ['C1', 'C2'], ['.', '+'], ['clean', 'frank']
    if include_rave is True:
        cols, marks, labs = cols.append('C3'), marks.append('x'), labs.append('rave')

    # brightness profiles (clean, frank, rave)
    rs = [rc, rf]
    if include_rave is True:
        rs.append(rr)

    Is_jy_sr = [Ic, If]
    if include_rave is True:
        Is_jy_sr.append(Ir)
    
    # brightness uncertainties
    Ies_jy_sr = [[Iec, Iec], [Ief, Ief]]
    if include_rave is True:
        # rave fits have dfft lower and upper uncertainties
        Ies_jy_sr.append([Ier_lo, Ier_hi])

    for ii, jj in enumerate(Is_jy_sr):     
        # convert Jy / sterad to mJy / arcsec^2
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

    phis_mod = np.linspace(model["base"]["geom"]["PA"] - 180, 
                                  model["base"]["geom"]["PA"] + 180,
                                  model["clean"]["Nphi"] 
                                  )
    
    if include_rave is True:
        # plot 1d rave residual brightness   
        rave_resid_im_path = parse_rave_filename(model, file_type='rave_residual_image')
        rave_resid_im = np.load(rave_resid_im_path)
        
        # convert Jy / pixel to Jy / arcsec
        rave_resid_im /= model["rave"]["pixel_scale"] ** 2 
        
        rave_resid_r, rave_resid_I = radial_profile_from_image( 
            rave_resid_im, geom=model["base"]["geom"], 
            rmax=max(rr), Nr=len(rr), phis=phis_mod, 
            npix=rave_resid_im.shape[0], pixel_scale=model["rave"]["pixel_scale"],
            bmaj=0, bmin=0, image_rms=0, model_image=True, arcsec2=False
            )

        ax3.plot(rave_resid_r, rave_resid_I * 1e3, c='C3') 

    # load frank residual visibilities (at projected data u,v)
    frank_resid_vis = load_bestfit_frank_uvtable(model, resid_table=True)
    # get pixel scale
    xf, _, _ = make_image(sol, npix, project=True)
    frank_pixel_scale = np.diff(xf).mean() / 2
    # plot 1d frank residual brightness
    frank_resid_im = dirty_image(frank_resid_vis, robust=resid_im_robust, npix=npix, pixel_scale=frank_pixel_scale)
    
    frank_resid_r, frank_resid_I = radial_profile_from_image(
        frank_resid_im, geom=model["base"]["geom"], 
        rmax=model["frank"]["rout"], Nr=model["frank"]["N"], phis=phis_mod, 
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
    vis_profiles = [Vc, Vf]
    if include_rave is True:
        vis_profiles.append(Vr)

    for ii, jj in enumerate(vis_profiles):
        ax1.plot(grid / 1e6, jj * 1e3, c=cols[ii], label=labs[ii])

    # plot binned visibility residuals
    bin_vis = UVDataBinner(bls, Vp, weights, model["plot"]["bin_widths"][-1])    
    _, bin_Vc = generic_dht(rc, Ic, Rmax=sol.Rmax, N=sol._info["N"], 
        grid=bin_vis.uv, inc=0)
    bin_Vf = sol.predict_deprojected(bin_vis.uv, I=If)
    if include_rave is True:
        _, bin_Vr = generic_dht(rr, Ir, Rmax=sol.Rmax, N=sol._info["N"], 
            grid=bin_vis.uv, inc=0)

    resid_yscale_guess = []
    binned_vis_profiles = [bin_Vc, bin_Vf]
    if include_rave is True:
        binned_vis_profiles.append(bin_Vr)
        
    for ii, jj in enumerate(binned_vis_profiles):
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
    ax2.set_ylim(-np.nanmax(resid_yscale), np.nanmax(resid_yscale))

    resid_yscale_I = np.nanmax(abs(frank_resid_I)) * 1e3
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

    ff = '{}/profile_compare_cleanRobust{}_frankResidRobust{}.png'.format(
        model["base"]["save_dir"], model["clean"]["robust"], resid_im_robust)
    print('    saving figure to {}'.format(ff))
    plt.savefig(ff, dpi=300)
    
    return fig


def image_comparison_figure(fits, model, resid_im_robust=2.0, npix=1000, xy_bounds=[-7,7], include_rave=True):
    """
    Generate a figure comparing clean, frank, and optionally rave 2d images

    Parameters
    ----------
    fits : nested list
        Clean, frank, rave profiles to be plotted. Output of `input_output.load_bestfit_profiles`
    model : dict
        Dictionary containing pipeline parameters
    resid_im_robust : float, default=2.0
        Robust weighting parameter used for imaging frank residual visibilities.
    npix : int
        Number of pixels along one axis used to make frank images
    xy_bounds : list of float, default=[-7,7]
        Plot axis bounds for (assumed square) images        
    include_rave : bool, default=True
        Whether to include rave results in figure 

    Returns
    -------
    fig : `plt.figure` instance
        The generated figure
    """

    print('  Figures: making image comparison figure')

    # load best-fit profiles
    if include_rave is True:
        [[rc, Ic, Iec], [grid, Vc]], [[rr, Ir, Ier_lo, Ier_hi], [grid, Vr]], [[rf, If, Ief], [grid, Vf], sol] = fits
    else:
        [[rc, Ic, Iec], [grid, Vc]], _, [[rf, If, Ief], [grid, Vf], sol] = fits

    # get clean images
    base_path = "{}/{}.combined.{}corrected.briggs.{}.{}.{}".format(
        model["base"]["input_dir"], 
        model["base"]["disk"], 
        model["base"]["SMG_sub"],
        model["clean"]["robust"], 
        model["clean"]["npix"], 
        model["clean"]["pixel_scale"]
        )
    clean_fits = base_path + ".pbcor.fits" 
    model_fits = base_path + ".model.fits"

    clean_image, clean_beam = load_fits_image(clean_fits)
    # convert clean image from Jy / beam to Jy / arcsec^2
    bmaj, bmin = clean_beam * 3600
    print('    clean beam: bmaj {} x bmin {} arcsec'.format(bmaj, bmin))
    beam_area = np.pi * bmaj * bmin / (4 * np.log(2))
    clean_image = clean_image / beam_area 

    # convert clean model image from Jy / pixel to Jy / arcsec^2
    model_image = load_fits_image(model_fits, aux_image=True)    
    model_image = model_image / model["clean"]["pixel_scale"] ** 2

    # set clean image pixel size (assuming square image)
    clean_im_xmax = model["clean"]["pixel_scale"] * model["clean"]["npix"] / 2
    clean_extent = [clean_im_xmax, -clean_im_xmax, -clean_im_xmax, clean_im_xmax]

    if include_rave is True:
        # make rave pseudo-2d image
        rave_im_xmax = model["rave"]["pixel_scale"] * len(rr) / 2 
        rave_extent = [rave_im_xmax, -rave_im_xmax, -rave_im_xmax, rave_im_xmax]    
        rave_image, _, _ = sweep_profile(rr, Ir, project=True,
            xmax=rave_im_xmax, ymax=rave_im_xmax, dr=model["rave"]["pixel_scale"], 
            phase_shift=True, geom=sol.geometry
            )
        rave_image = jy_convert(rave_image, 'sterad_arcsec2')

        # make rave residual image (again assuming square images)
        rave_resid_im_path = parse_rave_filename(model, file_type='rave_residual_image')
        rave_resid_im = np.load(rave_resid_im_path)

        # convert Jy / pixel to Jy / arcsec
        rave_resid_im /= model["rave"]["pixel_scale"] ** 2 
        rave_resid_Imax = np.nanmax(abs(rave_resid_im))

    # make frank pseudo-2d image
    xf, yf, frank_image = make_image(sol, npix, project=True)
    frank_image = frank_image.T
    frank_image = jy_convert(frank_image, 'sterad_arcsec2')
    frank_extent = [xf[-1], xf[0], yf[-1], yf[0]]
    frank_pixel_scale = np.diff(xf).mean()

    frank_resid_vis = load_bestfit_frank_uvtable(model, resid_table=True)
    # generate frank residual image
    frank_resid_im = dirty_image(frank_resid_vis, robust=resid_im_robust, npix=npix, pixel_scale=frank_pixel_scale)
    frank_resid_Imax = np.nanmax(abs(frank_resid_im))
    frank_resid_extent = [xf[-1], xf[0], yf[0], yf[-1]]

    # make figure
    fig = plt.figure(figsize=(10,6))
    fig.suptitle("{} -- robust = {} for clean (and rave if included); {} for frank imaged residuals".format(
        model["base"]["disk"],
        model["clean"]["bestfit"]["robust"],
        resid_im_robust)
        )
    gs = GridSpec(2, 3, figure=fig, hspace=0.01, wspace=0.2, left=0.04, right=0.97, top=0.98, bottom=0.01)

    ax4 = fig.add_subplot(gs[0, 0])
    ax5 = fig.add_subplot(gs[1, 0])
    ax6 = fig.add_subplot(gs[0, 2]) 
    ax7 = fig.add_subplot(gs[1, 2])
    ax8 = fig.add_subplot(gs[0, 1])
    ax9 = fig.add_subplot(gs[1, 1])

    # plot clean image
    norm = Normalize(vmin=np.nanmin(clean_image) * 1e3, vmax=np.nanmax(clean_image) * 1e3)
    plot_image(clean_image * 1e3, clean_extent, ax4, norm=norm,
               cbar_label='$I_{clean}$ [mJy arcsec$^{-2}$]'
               )

    # plot clean model image
    plot_image(model_image * 1e3, clean_extent, ax5, cmap="Reds",
               norm=get_image_cmap_norm(model_image * 1e3, stretch='asinh'), 
               cbar_label='$I_{clean\ model}$ [mJy arcsec$^{-2}$]'
               )            

    if include_rave is True:
        # plot rave pseudo-image
        plot_image(rave_image * 1e3, rave_extent, ax6, 
                cbar_label='$I_{rave}$ [mJy arcsec$^{-2}$]'
                )

        # plot (clean - rave) image using symmetric colormap
        rave_resid_norm = Normalize(vmin=-rave_resid_Imax * 1e3, 
                                    vmax=rave_resid_Imax * 1e3)
        plot_image(rave_resid_im * 1e3, rave_extent, ax7, cmap="RdBu_r",
                norm=rave_resid_norm, 
                cbar_label='$I_{clean - rave}$ [mJy arcsec$^{-2}$]'
                )

    # plot frank pseudo-image 
    plot_image(frank_image * 1e3, frank_extent, ax8,
               cbar_label='$I_{frank}$ [mJy arcsec$^{-2}$]'
               )   

    # plot frank imaged residuals using symmetric colormap
    frank_resid_norm = Normalize(vmin=-frank_resid_Imax * 1e3, 
                                 vmax=frank_resid_Imax * 1e3)
    plot_image(frank_resid_im * 1e3, frank_resid_extent, ax9, cmap="RdBu_r", 
               norm=frank_resid_norm, 
               cbar_label='$\mathcal{F}(V_{frank\ resid.}$) [mJy arcsec$^{-2}$]'
               ) 
    ax9.set_title('Pixel scale {:.1f} mas'.format(frank_pixel_scale * 1e3))

    for ax in [ax4, ax5, ax6, ax7, ax8, ax9]:
        ax.set_xlim(xy_bounds[1], xy_bounds[0])
        ax.set_ylim(xy_bounds[0], xy_bounds[1])

    ff = '{}/image_compare_cleanRobust{}_frankResidRobust{}.png'.format(
        model["base"]["save_dir"], model["clean"]["robust"], resid_im_robust)
    print('    saving figure to {}'.format(ff))
    plt.savefig(ff, dpi=300)
    
    return fig


def frank_image_diag_figure(model, sol, frank_resid_vis, resid_im_robust=2.0, 
                            npix=1800, xy_bounds=[-7,7], save_prefix=None
                            ):
    """
    Generate a figure showing frank brightness profile reprojected and swept 
    over 2\pi in azimuth, and reprojected dirty image of frank residual visibilities

    Parameters
    ----------
    model : dict
        Dictionary containing pipeline parameters
    sol : frank _HankelRegressor object
        Reconstructed profile using maximum a posteriori power spectrum
        (see frank.radial_fitters.FrankFitter)     
    frank_resid_vis : list
        frank residual visibilities: u-coordinates, v-coordinates, visibility amplitudes 
        (Re(V) + Im(V) * 1j), weights             
    resid_im_robust : float, default=2.0
        Robust weighting parameter used for imaging frank residual visibilities.
    npix : int, default=2400
        Number of pixels along one axis used to make frank images
    xy_bounds : list of float, default=[-7,7]
        Plot axis bounds for (assumed square) images
    save_prefix : string, default = None
        Prefix for saved figure name. If None, the figure won't be saved

    Returns
    -------
    fig : `plt.figure` instance
        The generated figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    
    # make frank pseudo-2d image
    xf, yf, frank_image = make_image(sol, npix, project=True)
    frank_image = frank_image.T
    frank_image = jy_convert(frank_image, 'sterad_arcsec2')
    frank_extent = [xf[-1], xf[0], yf[-1], yf[0]]
    frank_pixel_scale = np.diff(xf).mean()

    # make frank residual image
    frank_resid_im = dirty_image(frank_resid_vis, robust=resid_im_robust, npix=npix, pixel_scale=frank_pixel_scale)
    frank_resid_Imax = np.nanmax(abs(frank_resid_im))    
    frank_resid_extent = [xf[-1], xf[0], yf[0], yf[-1]]

    # plot frank pseudo-image 
    plot_image(frank_image * 1e3, frank_extent, axes[0], 
               cbar_label='$I_{frank}$ [mJy arcsec$^{-2}$]'
               )   

    # plot frank imaged residuals using symmetric colormap
    frank_resid_norm = Normalize(vmin=-frank_resid_Imax * 1e3, 
                                 vmax=frank_resid_Imax * 1e3)
    plot_image(frank_resid_im * 1e3, frank_resid_extent, axes[1], cmap="RdBu_r", 
               norm=frank_resid_norm, 
               cbar_label='$\mathcal{F}(V_{frank\ resid.}$) [mJy arcsec$^{-2}$]'
               ) 

    for ax in axes:
        ax.set_xlim(xy_bounds[1], xy_bounds[0])
        ax.set_ylim(xy_bounds[0], xy_bounds[1])

    fig.suptitle("{} -- robust {} for imaged residuals. Pixel scale {:.1f} mas".format(model["base"]["disk"],
                                                                                   resid_im_robust,
                                                                                   frank_pixel_scale * 1e3)
                                                                                   )
    
    if save_prefix:
        plt.savefig(save_prefix + f"_frank_images_residRobust{resid_im_robust}.png", dpi=300,
                    bbox_inches='tight')
        plt.close()

    return fig, axes


def frank_multifit_figure(model, sols, plot_var, single_panel=False, save_prefix=None):
    """
    Generate a figure showing results of vertical inference with frank 1+1D

    Parameters
    ----------
    model : dict
        Dictionary containing pipeline parameters
    sols : list of frank _HankelRegressor objects
        frank fits performed over multiple hyperparameter values        
    plot_var : str
        One of ["I", "V"]: Whether to plot brightness profiles "I" 
        or visibility fits "V"
    single_panel : bool
        Whether to plot all fits on a single panel
    save_prefix : str, default = None
        Prefix for saved figure name. If None, the figure won't be saved
    """
    if single_panel is True:
        fig_layout = "single_panel"
    else:
        fig_layout = "grid"
    print(f"  Figures: making multifit figure: {fig_layout} of {plot_var}")

    if single_panel is True:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
        axes = [axes]
    else:
        fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(10,8))
        axes = axes.ravel()

    if single_panel is True:
        plot_kwargs = {"c":None}
    else:
        plot_kwargs = {"c":"r"}
        
    title = model["base"]["disk"]

    if plot_var == "V":
        [u, v, vis, weights] = get_vis(model)

        # deproject observed vis
        up, vp, Vp = sols[0].geometry.apply_correction(u, v, vis) # assuming all sols have same geometry
        bls = np.sqrt(up**2 + vp**2)

        # bin observed vis
        bin_width = model["plot"]["bin_widths"][-1] # forcing single bin width
        bin_vis = UVDataBinner(bls, Vp, weights, bin_width)

        grid = np.logspace(np.log10(min(bin_vis.uv)), np.log10(max(bin_vis.uv)), 10**3)

        title += f"\nData shown in {bin_width/1e3} k$\\lambda$ bins"

    xs, ys = [], []
    for ii, ss in enumerate(sols):
        if single_panel is True:
            ii = 0

        lab = f"$\\alpha$ {ss.info['alpha']}, $w$ {ss.info['wsmooth']}"

        if plot_var == "I":
            axes[ii].plot(ss.r * model["base"]["dist"], ss.I / 1e6, label=lab, **plot_kwargs)

        else:
            if (single_panel is True and ii == 0) or single_panel is False:
                # plot binned observed Re(V)
                axes[ii].plot(bin_vis.uv / 1e6, bin_vis.V.real * 1e3, 'k.', ls='None', zorder=-50)

            # get frank vis fit
            Vf = ss.predict_deprojected(grid)
            axes[ii].plot(grid / 1e6, Vf * 1e3, label=lab, **plot_kwargs)

        axes[ii].legend(loc='upper right', fontsize=6)

        if plot_var == "V":
            axes[ii].set_ylim(-1.1 * max(Vf * 1e3), 1.1 * max(Vf * 1e3))

        xs.append(axes[ii].get_xlim())
        ys.append(axes[ii].get_ylim())

    min_x, max_x = min(xx[0] for ii,xx in enumerate(xs)), max(xx[1] for ii,xx in enumerate(xs))
    min_y, max_y = min(yy[0] for ii,yy in enumerate(ys)), max(yy[1] for ii,yy in enumerate(ys))
    
    for ii, aa in enumerate(axes):
        aa.xaxis.set_ticks_position("both")
        aa.yaxis.set_ticks_position("both")

        if single_panel is False:
            if ii == 0:
                aa.xaxis.tick_top()
                aa.xaxis.set_label_position("top")
            else:
                aa.xaxis.set_ticklabels([])    
                aa.yaxis.set_ticklabels([])

        aa.set_ylim(min_y, max_y)
        # aa.set_xlim(min_x, max_x)
        if plot_var == "I":
            aa.set_xlim(0, max_x)
        else:
            aa.set_xlim(0, max(bls) * 1.05 / 1e6)

    if plot_var == "I":
        axes[0].set_ylabel(r'I [MJy sr$^{-1}$]')
        axes[0].set_xlabel(r'r [AU]')
    else:
        axes[0].set_ylabel(r'Re(V) [mJy]')
        axes[0].set_xlabel(r'Baseline [M$\lambda$]')

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(title)

    if save_prefix:
        prefix = save_prefix + f"_frank_multifit_{plot_var}"
        if single_panel is False:
            prefix += "_grid"
        
        plt.savefig(prefix + ".png", dpi=300, bbox_inches='tight')

        plt.close()

    return fig, axes


def aspect_ratio_figure(model):
    """
    Generate a figure showing results of vertical inference with frank 1+1D

    Parameters
    ----------
    model : dict
        Dictionary containing pipeline parameters
        
    Returns
    -------
    fig : `plt.figure` instance
        The generated figure
    """
    
    print('  Figures: making aspect ratio figure')

    # log evidence results over grid of alpha, wsmooth, h values
    alphas, wsmooths, hs, logevs = np.genfromtxt("{}/vertical_inference_frank.txt".format(model["base"]["save_dir"])).T

    # size of h grid
    nh = len(np.unique(hs))
    # first idx of each unique (alpha, wsmooth) pair (used to group results by priors)
    idx = np.arange(0, len(logevs), nh)

    fig, axes = plt.subplots(ncols=1, nrows=len(idx), figsize=(7.5,5))
    fig.suptitle(model["base"]["disk"])

    for ii, jj in enumerate(idx):
        alpha, wsmooth = alphas[jj], wsmooths[jj]
        if len(idx) == 1:
            h, logev = hs * 1, logevs * 1
            axes = [axes]
        else:
            h, logev = hs[jj:jj + np.diff(idx)[0]], logevs[jj:jj + np.diff(idx)[0]]
        
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

        axes[ii].plot(h, logp, 'r.', label='PDF samples', zorder=10)
        axes[ii].plot(hgrid, logp_fine, 'k', label='PDF interp.')

        axes[ii].plot(hgrid[good_idx], cdf, 'c', label='CDF')

        axes[ii].axvline(h16, ls='--', c='g', label='$\mu$')
        axes[ii].axvline(h50, ls='--', c='c', label='$-1\sigma$')
        axes[ii].axvline(h84, ls='--', c='r', label='$+1\sigma$')
        axes[ii].axvline(hmax, ls='--', c='m', label='point estimate')

        axes[ii].set_xscale('log')

        axes[ii].set_title(r'$\alpha$ = {}, w$_{{smooth}}$ = {:.0e}, h={:.3f}$_{{-{:.3f}}}^{{+{:.3f}}}$'.format(alpha, wsmooth, h50, h50 - h16, h84 - h50)) 

    # show labels on last panel
    plt.legend()
    plt.xlabel('h = H / r')
    plt.ylabel(r'Normalized P(h|V, $\beta$)')

    ff = '{}/vertical_inference_frank.png'.format(
        model["base"]["save_dir"])
    print('    saving figure to {}'.format(ff))
    plt.savefig(ff, dpi=300)

    return fig


def parametric_fit_figure(fit, reference, model):
    """
    Generate a figure showing results of parametric fits to nonparametric frank
      brightness profiles.

    Parameters
    ----------
    fit: dict
        Dictionary containing initial and final fit parameters for parametric
        model, and loss values

    reference : list
        Reference profile radial points, brightness, 1 sigma uncertainty

    model : dict
        Dictionary containing pipeline parameters
        
    Returns
    -------
    fig : `plt.figure` instance
        The generated figure
    """    

    print('  Figures: making parametric fit comparison figure')
    
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10,7.5))
    axes = axes.ravel()
    fig.delaxes(axes[3])

    fig.suptitle(f"{model['base']['disk']}: {fit.functional_form} fit to frank profile")

    # print formatted bestfit values in fig
    st = 'Best fit values:\n'
    for i,j in fit.bestfit_params.items():
        if j is not None:
            st += f"{i} = {j:.2f}\n"
    st += f"\nSource distance: {model['base']['dist']:.0f} pc"
    fig.text(0.6, 0.15, st)

    # reference profile radial points, brightness, 1 sigma
    rr, II, ss = reference

    fit_initial = fit.parametric_model(fit.initial_params, rr)
    fit_final = fit.parametric_model(fit.bestfit_params, rr)
    resid = II - fit_final

    axes[0].plot(rr, II / 1e6, 'k', label='frank')
    # frank 1 sigma
    axes[0].fill_between(rr, (II - ss) / 1e6, (II + ss) / 1e6, color='k', alpha=0.4)    

    axes[0].plot(rr, fit_initial / 1e6, 'c', label="initial guess")
    axes[0].plot(rr, fit_final / 1e6, 'r', label="best fit")
    
    axes[2].plot(rr, resid / 1e6, '.', ms=2, c='#a4a4a4', 
                 label = f"mean {np.mean(resid) / 1e6:.4f} MJy sr$^{-1}$"
                 )
    axes[2].axhline(y=0, c='c', ls='--')

    axes[1].semilogy(fit.loss_history, 'b', label=f"final loss {fit.loss_history[-1]:.2f}")

    axes[0].legend()
    axes[0].set_ylabel(r'I [MJy sr$^{-1}$]')

    axes[2].set_xlabel(r'r [arcsec]')
    axes[2].set_ylabel(r'Resid. I [MJy sr$^{-1}$]')

    # make y-axis symmetric about 0
    bound = max(abs(np.asarray(axes[2].get_ylim())))
    axes[2].set_ylim(-bound, bound)
    axes[2].legend()

    axes[1].legend()
    axes[1].set_xlabel(r'Iteration')    
    axes[1].set_ylabel(r'Loss')    

    ff = f"{model['base']['parametric_dir']}/parametric_fit_{fit.functional_form}.png"
    print(f"    saving figure to {ff}")
    plt.savefig(ff, dpi=300)

    return fig