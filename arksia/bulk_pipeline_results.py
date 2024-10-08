"""This module contains a function to obtain results from fits/analysis of multiple sources
(written by Jeff Jennings)."""

import os
import json
import numpy as np 
import matplotlib.pyplot as plt 

from frank.geometry import FixedGeometry
from frank.utilities import UVDataBinner

from arksia.pipeline import model_setup, process_rave_fit
from arksia.input_output import load_bestfit_profiles, get_vis

# sources for which cleaan profile extraction routine is invalid
# (edge-on sources without a cavity; see `arksia.extract_radial_profile`)
no_clean = ['HD32297', 'HD15115', 'HD14055', 'HD197481', 'HD39060']

def survey_summary(gen_par_f='./pars_gen.json', 
         source_par_f='./pars_source.json',
         phys_par_f='./summary_disc_parameters.csv',
         profiles_txt=True, profiles_fig=True, robust=0.5,
         include_rave=True
         ):
    """
    Generate summary radial profile results across multiple survey sources.

    Parameters
    ----------       
    gen_par_f : string, default='pars_gen.json'
        Path to the general parameters file  
    source_par_f : string, default='pars_source.json'
        Path to the parameter file with custom values for each source         
    phys_par_f : string, default='pars_gen.json'
        Path to the physical parameters file
    profiles_txt : bool, default=True
        Whether to produce a .txt file per source containting the 
        clean, rave, frank brightness profiles (sampled at same radii)
    profiles_fig : bool, default=True
        Whether to produce a single figure showing brightness profiles for 
        all sources
    robust : float, default=2.0
        Robust weighting value to use for retrieving clean, rave results
    include_rave : bool, default=True
        Whether to include rave results in summary

    Returns
    -------
    figs : `plt.figure` instance
        The generated figures, produced if `profiles_fig` is True
    """

    # get all source names
    source_pars = json.load(open(source_par_f, 'r'))
    disk_names = []
    for dd in source_pars:
        disk_names.append(dd)

    fig0, axs0 = plt.subplots(nrows=5, ncols=4, figsize=(10, 10), squeeze=True)
    fig1, axs1 = plt.subplots(nrows=5, ncols=4, figsize=(10, 10), squeeze=True)
    fig2, axs2 = plt.subplots(nrows=5, ncols=4, figsize=(10, 10), squeeze=True)
    fig3, axs3 = plt.subplots(nrows=5, ncols=4, figsize=(10, 10), squeeze=True)
    figs, axs = [fig0, fig1, fig2, fig3], [axs0, axs1, axs2, axs3]

    gen_pars = json.load(open(gen_par_f, 'r'))
    source_pars = json.load(open(source_par_f, 'r'))

    for ii, jj in enumerate(disk_names):
        # update generic parameters that vary by source
        gen_pars['base']['input_dir'] = f"{os.path.dirname(gen_pars['base']['input_dir'])}/{jj}"
        # gen_pars['base']['input_dir'] = f"./{jj}"
        disk_pars = source_pars[jj]
        gen_pars['clean']['robust'] = disk_pars["clean"]["bestfit"]["robust"]

        # save updated gen_pars
        gen_pars_current = os.path.join(os.path.dirname(gen_par_f), 'pars_gen_temp.json')
        with open(gen_pars_current, 'w') as f:
            json.dump(gen_pars, f)

        # generate model for each source
        class parsed_args():
            base_parameter_filename = gen_pars_current
            source_parameter_filename = source_par_f
            physical_parameter_filename = phys_par_f
            disk = jj
        model = model_setup(parsed_args)

        # process_rave_fit(model)

        # best-fit clean, rave, frank profile for each source
        fits = load_bestfit_profiles(model, robust, include_rave=include_rave)
        [[rc, Ic, Iec], [grid, Vc]] = fits[0]
        [[rf, If, Ief], [grid, Vf], sol] = fits[2]
        if include_rave:
            [[rr, Ir, Ier_lo, Ier_hi], [grid, Vr]] = fits[1]        

        # interpolate clean and rave profiles onto frank radial points 
        Is_interp = [If]
        Ies_interp = [[Ief, Ief]]
        Vs_interp = [Vf]

        Ic_interp = np.interp(rf, rc, Ic)
        Iec_interp = np.interp(rf, rc, Iec)
        if jj in no_clean:
            dummy = np.array([np.nan] * len(rc))
            Is_interp.append(dummy)
            Ies_interp.append([dummy, dummy])
            Vs_interp.append(np.array([np.nan] * len(grid)))
        else:            
            Is_interp.append(Ic_interp)
            Ies_interp.append([Iec_interp, Iec_interp])
            Vs_interp.append(Vc)

        if include_rave:
            Ir_interp = np.interp(rf, rr, Ir)
            Ier_lo_interp = np.interp(rf, rr, Ier_lo)
            Ier_hi_interp = np.interp(rf, rr, Ier_hi)

            Is_interp.append(Ir_interp)
            Ies_interp.append([Ier_lo_interp, Ier_hi_interp])
            Vs_interp.append(Vr)

        # bin observed vis
        u, v, V, weights = get_vis(model)
        V = V - model["frank"]["fstar"]
        frank_geom = FixedGeometry(**model["base"]["geom"]) 
        u, v, V = frank_geom.apply_correction(u, v, V)
        bls = np.hypot(u, v)
        bin_obs = UVDataBinner(bls, V, weights, 20e3)
        bin_obs2 = UVDataBinner(bls, V, weights, 50e3)
            
        if profiles_txt:
            ff = f'{model["base"]["save_dir"]}/{jj}_radial_profiles.txt'
            print('  Survey summary: saving radial profiles to {}'.format(ff))

            # save .txt file per source with clean,rave,frank profiles
            header=f"dist={model['base']['dist']} [au].\nAll brightnesses in [Jy/steradian].\nUncertainties not comparable across models.\nColumns: r [au]\t\tI_frank\t\tsigma_frank\t\t"
            profiles = np.array([rf * model["base"]["dist"], If, Ief])

            if jj not in no_clean:
                header += "I_clean\t\tsigma_clean\t\t"
                profiles = np.append(profiles, [Ic_interp, Iec_interp], axis=0)

            if include_rave:
                header += "I_rave\t\tsigma_lower_rave\t\tsigma_upper_rave\nRave uncertainties have unique lower and upper bounds."
                profiles = np.append(profiles, [Ir_interp, Ier_lo_interp, Ier_hi_interp], axis=0)
            
            np.savetxt(ff, profiles.T, header=header)


        if profiles_fig:
            # generate, save figures for brightness profiles of all sources with and without uncertainties, visibility models
            for hh in range(4):
                fig = figs[hh]
                ax = axs[hh]

                # flatten axes
                ax = [bb for aa in ax for bb in aa]
                cols, labs = ['C2', 'C1'], ['frank', 'clean']
                if include_rave: 
                    cols.append('C3')
                    labs.append('rave')                    

                for kk, ll in enumerate(Is_interp):   
                    if hh < 2:  
                        # plot profile
                        ax[ii].plot(rf * model["base"]["dist"], ll / 1e6, c=cols[kk], label=labs[kk], zorder=-kk)
                    else:
                        # plot observed Re(V)
                        ax[ii].plot(bin_obs.uv / 1e6, bin_obs.V.real * 1e3, c='#a4a4a4', marker='x', ls='None', zorder=-9)
                        ax[ii].plot(bin_obs2.uv / 1e6, bin_obs2.V.real * 1e3, c='k', marker='+', ls='None', zorder=-8)

                        # plot visibility model
                        ax[ii].plot(grid / 1e6, Vs_interp[kk] * 1e3, c=cols[kk], label=labs[kk], zorder=-kk)

                        ax[ii].set_xlim(0, max(bls / 1e6))
                        if hh == 3: 
                            if include_rave:
                                global_min = min(min(Vf), np.nanmin(Vr), np.nanmin(Vc)) * 1e3
                            else:
                                global_min = min(min(Vf), np.nanmin(Vc)) 
                            ax[ii].set_ylim(global_min * 1e3, 1.5 * Vf.mean() * 1e3)

                    if hh == 1:
                        # 1 sigma uncertainty band
                        band = ax[ii].fill_between(rf * model["base"]["dist"], 
                                            (ll - Ies_interp[kk][0]) / 1e6, (ll + Ies_interp[kk][1]) / 1e6, 
                                        color=cols[kk], alpha=0.4, zorder=-kk)
                        # prevent 1 sigma band from altering y-limits
                        band.remove()
                        ax[ii].relim()
                        ax[ii].add_collection(band, autolim=False)

                ax[ii].axhline(y=0, ls='--', c='k', zorder=-10)

                fstar_ujy = model["frank"]["fstar"] * 1e6
                ax[ii].set_title(f"{jj}, " + r"$F_* =$ " + f"{fstar_ujy:.0f} uJy", fontsize=10)

                if ii == 0:
                    ax[ii].legend(loc='upper right', fontsize=8)
                    if hh < 2:
                        ax[ii].set_xlabel('r [au]')
                        ax[ii].set_ylabel(r'I [MJy sterad$^{-1}$]')
                    else:
                        ax[ii].set_xlabel(r'Baseline [M$\lambda$]')
                        ax[ii].set_ylabel(r'Re(V) [mJy]')                        

    if profiles_fig:
        ff0 = f'{model["base"]["save_dir"]}/../survey_profile_summary.png'
        ff1 = f'{model["base"]["save_dir"]}/../survey_profile_summary_unc.png'
        ff2 = f'{model["base"]["save_dir"]}/../survey_visibility_summary.png'
        ff3 = f'{model["base"]["save_dir"]}/../survey_visibility_summary_zoom.png'
        print('  Survey summary: saving figures to:\n      {},\n {},\n {},\n{}'.format(ff0, ff1, ff2, ff3))

        fig1.suptitle(r'$1\sigma$ uncertainties do not include systematic unc., and are not comparable across models')    

        plt.figure(fig0); plt.tight_layout(); plt.savefig(ff0, dpi=300)
        plt.figure(fig1); plt.tight_layout(); plt.savefig(ff1, dpi=300)
        plt.figure(fig2); plt.tight_layout(); plt.savefig(ff2, dpi=300)
        plt.figure(fig3); plt.tight_layout(); plt.savefig(ff3, dpi=300)

        os.remove(gen_pars_current)

        return figs
    
if __name__ == "__main__":
    survey_summary()