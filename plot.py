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

from input_output import get_vis, load_fits_image, load_bestfit_frank_uvtable, parse_rave_filename
from image_radial_profile import radial_profile_from_image

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
        
    print('  Figures: making profile comparison figure')

    # load best-fit profiles
    [[rc, Ic, Iec], [grid, Vc]], [[rr, Ir, Ier_lo, Ier_hi], [grid, Vr]], [[rf, If, Ief], [grid, Vf], sol] = fits

    fig = plt.figure(figsize=(10,6))
    fig.suptitle("{} -- robust = {} for clean and rave".format(
        model["base"]["disk"],
        model["clean"]["robust"])
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
                                               robust=model["clean"]["robust"])
    frank_resid_im = np.squeeze(frank_resid_im)

    phis_mod = np.linspace(model["base"]["geom"]["PA"] - 180, 
                                  model["base"]["geom"]["PA"] + 180,
                                  model["clean"]["Nphi"] 
                                  )
    
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

    ff = '{}/profile_compare_robust{}.png'.format(
        model["base"]["save_dir"], model["clean"]["robust"])
    print('    saving figure to {}'.format(ff))
    plt.savefig(ff, dpi=300)
    
    return fig


def image_comparison_figure(fits, model):
    """
    Generate a figure comparing clean, rave, frank 2d image

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

    print('  Figures: making image comparison figure')
    [[rc, Ic, Iec], [grid, Vc]], [[rr, Ir, Ier_lo, Ier_hi], [grid, Vr]], [[rf, If, Ief], [grid, Vf], sol] = fits

    # get clean images
    base_path = "{}/{}.combined.{}corrected.briggs.{}.{}.{}".format(
        model["base"]["clean_dir"], 
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

    # make rave pseudo-2d image
    rave_im_xmax = model["rave"]["pixel_scale"] * len(rr) / 2 
    rave_extent = [rave_im_xmax, -rave_im_xmax, -rave_im_xmax, rave_im_xmax]    
    rave_image, _, _ = sweep_profile(rr, Ir, project=True,
        xmax=rave_im_xmax, ymax=rave_im_xmax, dr=model["rave"]["pixel_scale"], 
        phase_shift=True, geom=sol.geometry
        )
    rave_image = jy_convert(rave_image, 'sterad_arcsec2')

    # make rave residual image (again assuming square images)
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
    rave_resid_Imax = abs(rave_resid_im).max()

    # make frank pseudo-2d image
    frank_image, xfim, yfim = sweep_profile(rf, If, project=True, 
        phase_shift=True, geom=sol.geometry
        )

    frank_image = jy_convert(frank_image, 'sterad_arcsec2')
    frank_extent = [xfim, -xfim, -yfim, yfim]

    # make frank imaged residual vis, using same robust as clean image.
    # load frank residual visibilities (at projected data u,v)
    [ufres, vfres, Vfres, wfres] = load_bestfit_frank_uvtable(model, resid_table=True)
    
    # generate dirty image of frank residual vis at same pixel scale, robust weighting value as clean image
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
                                               robust=model["clean"]["robust"])
    frank_resid_im = np.squeeze(frank_resid_im)
    
    frank_resid_Imax = abs(frank_resid_im).max()

    # same colormap for clean, rave, frank images
    maxI = max(np.nanmax(rave_image), np.nanmax(frank_image)) 
    uniform_norm = Normalize(vmin=0 * 1e3, vmax=maxI * 1e3)
    
    # make figure
    fig = plt.figure(figsize=(10,6))
    fig.suptitle("{} -- robust = {} for clean".format(
        model["base"]["disk"],
        model["clean"]["robust"])
        )
    gs = GridSpec(2, 3, figure=fig, hspace=0.01, wspace=0.2, left=0.04, right=0.97, top=0.98, bottom=0.01)

    ax4 = fig.add_subplot(gs[0, 0])
    ax5 = fig.add_subplot(gs[1, 0])
    ax6 = fig.add_subplot(gs[0, 2]) 
    ax7 = fig.add_subplot(gs[1, 2])
    ax8 = fig.add_subplot(gs[0, 1])
    ax9 = fig.add_subplot(gs[1, 1])

    # plot clean image
    plot_image(clean_image * 1e3, clean_extent, ax4, norm=uniform_norm,
               cbar_label='$I_{clean}$ [mJy arcsec$^{-2}$]'
               )

    # plot clean model image
    plot_image(model_image * 1e3, clean_extent, ax5, cmap="Reds",
               norm=get_image_cmap_norm(model_image * 1e3, stretch='asinh'), 
               cbar_label='$I_{clean\ model}$ [mJy arcsec$^{-2}$]'
               )            

    # plot rave pseudo-image
    plot_image(rave_image * 1e3, rave_extent, ax6, norm=uniform_norm, 
               cbar_label='$I_{rave}$ [mJy arcsec$^{-2}$]'
               )

    # plot (clean - rave) image using symmetric colormap
    rave_resid_norm = Normalize(vmin=-rave_resid_Imax * 1e3, 
                                 vmax=rave_resid_Imax * 1e3)
    plot_image(rave_resid_im * 1e3, rave_extent, ax7, cmap="RdBu",
               norm=rave_resid_norm, 
               cbar_label='$I_{clean - rave}$ [mJy arcsec$^{-2}$]'
               )

    # plot frank pseudo-image 
    plot_image(frank_image * 1e3, frank_extent, ax8, norm=uniform_norm,
               cbar_label='$I_{frank}$ [mJy arcsec$^{-2}$]'
               )   

    # plot frank imaged residuals using symmetric colormap
    frank_resid_norm = Normalize(vmin=-frank_resid_Imax * 1e3, 
                                 vmax=frank_resid_Imax * 1e3)
    plot_image(frank_resid_im * 1e3, frank_extent, ax9, cmap="RdBu", 
               norm=frank_resid_norm, 
               cbar_label='$\mathcal{F}(V_{frank\ resid.}$) [mJy arcsec$^{-2}$]'
               ) 

    for ax in [ax4, ax5, ax6, ax7, ax8, ax9]:
        ax.set_xlim(7,-7)
        ax.set_ylim(-7,7)

    ff = '{}/image_compare_robust{}.png'.format(
        model["base"]["save_dir"], model["clean"]["robust"])
    print('    saving figure to {}'.format(ff))
    plt.savefig(ff, dpi=300)
    
    return fig


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
    alphas, wsmooths, hs, logevs = np.genfromtxt("{}/frank/vertical_inference.txt".format(model["base"]["save_dir"])).T

    # size of h grid
    nh = len(np.unique(hs))
    # first idx of each unique (alpha, wsmooth) pair (used to group results by priors)
    idx = np.arange(0, len(logevs), nh)

    fig, axes = plt.subplots(ncols=1, nrows=len(idx), figsize=(15,10))
    fig.suptitle(model["base"]["disk"])

    for ii, jj in enumerate(idx):
        alpha, wsmooth = alphas[jj], wsmooths[jj]
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

        for ll in [h16, h50, h84]:
            axes[ii].axvline(ll, ls='--', c='g')
        axes[ii].axvline(hmax, ls='--', c='m', label='point estimate')

        axes[ii].set_xscale('log')

        axes[ii].set_title(r'$\alpha$ = {}, w$_{{smooth}}$ = {:.0e}, h={:.3f}$_{{-{:.3f}}}^{{+{:.3f}}}$'.format(alpha, wsmooth, h50, h50 - h16, h84 - h50)) 

    # show labels on last panel
    plt.legend()
    plt.xlabel('h = H / r')
    plt.ylabel(r'Normalized P(h|V, $\beta$)')

    plt.savefig('{}/vertical_inference_frank.png'.format(
        model["base"]["save_dir"]), dpi=300
    )

    return fig


def survey_summary(profiles_txt=True, profiles_fig=True, 
                   gen_par_f='default_gen_pars.json', 
                   source_par_f='default_source_pars.json', 
                   robust=2.0
                   ):
    """
    Generate radial profile results across all survey sources.

    Parameters
    ----------
    profiles_txt : bool, default=True
        Whether to produce a .txt file per source containting the 
        clean, rave, frank brightness profiles (sampled at same radii)
    profiles_fig : bool, default=True
        Whether to produce a single figure showing brightness profiles for 
        all sources
    gen_par_f : string, default='default_gen_pars.json'
        Path to the general parameter file
    source_par_f : string, default='default_source_pars.json'
        Path to the parameter file with custom values for each source
    robust : float, default=2.0
        Robust weighting value to use for retrieving clean, rave results
    
    Returns
    -------
    figs : `plt.figure` instance
        The generated figures, produced if `profiles_fig` is True
    """

    # get all source names
    source_pars = json.load(open(source_par_f, 'r'))
    disk_names = []
    for ii in source_pars:
        disk_names.append(ii)

    for ii, jj in enumerate(disk_names):
        class parsed_args():
            base_parameter_filename = gen_par_f
            disk = jj
            source_parameter_filename = source_par_f

        # generate model for each source
        model = model_setup(parsed_args)

        # best-fit clean, rave, frank profile for each source
        [[rc, Ic, Iec], [grid, Vc]], [[rr, Ir, Ier_lo, Ier_hi], [grid, Vr]], [[rf, If, Ief], [grid, Vf], sol] = load_bestfit_profiles(model, robust)

        # interpolate clean and rave profiles onto frank radial points 
        Ic_interp = np.interp(rf, rc, Ic)
        Iec_interp = np.interp(rf, rc, Iec)
        Ir_interp = np.interp(rf, rr, Ir)
        Ier_lo_interp = np.interp(rf, rr, Ier_lo)
        Ier_hi_interp = np.interp(rf, rr, Ier_hi)

        Is_jy_sr = [Ic_interp, Ir_interp, If]
        Ies_jy_sr = [[Iec_interp, Iec_interp], [Ier_lo_interp, Ier_hi_interp], [Ief, Ief]]


        if profiles_txt:
            # save .txt file per source with clean,rave,frank profiles
            np.savetxt('./{}_radial_profiles.txt'.format(jj), 
                    np.array([rf * model["base"]["dist"], Ic_interp, Iec_interp, 
                              If, Ief, Ir_interp, Ier_lo_interp, Ier_hi_interp
                              ]).T,
                    header="dist={} [au].\nAll brightnesses in [Jy/steradian].\nUncertainties not comparable across models. " 
                    header += "Rave uncertainties have unique lower and upper bounds.\nColumns: "
                    header += "r [au]\t\tI_clean\t\tsigma_clean\t\tI_frank\t\tsigma_frank\t\tI_rave\t\tsigma_lower_rave\t\tsigma_upper_rave".format(model["base"]["dist"])
                    )


        figs = []
        if profiles_fig:
            # generate, save figures for profiles of all sources and profiles with uncertainties.
            for hh in range(2):
                fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(10, 10), squeeze=True)
                cols, labs = ['C1', 'C3', 'C2'], ['clean', 'rave', 'frank']

                for kk, ll in enumerate(Is_jy_sr):     
                    # plot profile
                    axes[ii].plot(rf * model["base"]["dist"], ll, c=cols[kk], label=labs[kk])
                
                    if hh == 1:
                        # plot 1 sigma uncertainty band
                        axes[ii].fill_between(rf * model["base"]["dist"], 
                                            ll - Ies_jy_sr[kk][0], ll + Ies_jy_sr[kk][1], 
                                        color=cols[kk], alpha=0.4)
                
                axes[ii].axhline(y=0, ls='--', c='k')

                axes[ii].text(0.6, 0.9, jj, transform=axes[ii].transAxes)

                if ii == len(disk_names) - 1:
                    axes[ii].legend(loc='center right')
                    axes[ii].set_xlabel('r [au]')
                    axes[ii].set_ylabel(r'I [Jy sterad$^{-1}$]')

                axes[ii].set_title('$f_*=${:.0f} $\mu$Jy'.format(model["frank"]["fstar"] * 1e6))

                if hh == 1:
                    suptitle = r'$1\sigma$ uncertainties do not include systematic unc., and are not comparable across models'
                    fig.suptitle(suptitle)
                
                fig.tight_layout()

                if hh == 0:
                    plt.savefig('./survey_profile_summary.png')
                else:
                    plt.savefig('./survey_profile_summary_unc.png')

                figs.append(fig)

        return figs