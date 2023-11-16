"""This module is the main file for running radial pipeline fits and analysis 
(written by Jeff Jennings)."""

import os; os.environ.get('OMP_NUM_THREADS', '1')
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import multiprocess
import pickle 

from frank.constants import deg_to_rad
from frank.utilities import jy_convert
from frank.debris_fitters import FrankDebrisFitter
from frank.io import save_fit
from frank.make_figs import make_quick_fig
from frank.geometry import FixedGeometry
from frank.utilities import get_fit_stat_uncer

import arksia 
from arksia import input_output, extract_radial_profile, plot, parametric_fitter

def parse_parameters(*args):
    """
    Read in a .json parameter files to run the pipeline

    Parameters
    ----------
    parameter_filename : str
        Parameter file (.json)

    Returns
    -------
    args : dict
        Dictionary containing model parameters the pipeline uses
    """

    parser = argparse.ArgumentParser("Run the radial profile pipeline for ARKS data")

    parser.add_argument("-d", "--disk",
                        type=str,
                        help="Disk name")

    parser.add_argument("-b", "--base_parameter_filename",
                        type=str,
                        default="./pars_gen.json",
                        help="Parameter file (.json) with generic pars")
    
    parser.add_argument("-s", "--source_parameter_filename",
                        type=str,
                        default="./pars_source.json",
                        help="Parameter file (.json) with source-specific pars")

    args = parser.parse_args(*args)
    
    return args


def model_setup(parsed_args):
    """
    Initialze a set of model parameters to be used in pipeline routines.
    Run checks to catch errors on values of some parameters. 

    Parameters
    ----------
    parsed_args : dict
        Dictionary containing model parameters to be used in the pipeline

    Returns
    -------
    model : dict
        Dictionary containing final model parameters the pipeline uses, with 
        some processing and added parameters relative to `parsed_args`
    """

    # generic parameters
    model = json.load(open(parsed_args.base_parameter_filename, 'r'))
    model["base"]["disk"] = parsed_args.disk

    print('\nRunning radial profile pipeline for {}'.format(model["base"]["disk"]))

    # disk-specific parameters
    source_pars = json.load(open(parsed_args.source_parameter_filename, 'r'))
    disk_pars = source_pars[model["base"]["disk"]]

    # expect input files to be in "<root_dir>/<disk name>/<clean or rave or frank>"
    model["base"]["save_dir"] = os.path.join(model["base"]["root_dir"], "{}".format(model["base"]["disk"]))
    print("  Model setup: setting load/save paths as {}/<clean, rave, frank>. Visibility tables should be in frank path.".format(model["base"]["save_dir"]))

    model["base"]["clean_dir"] = os.path.join(model["base"]["save_dir"], "clean")
    model["base"]["rave_dir"] = os.path.join(model["base"]["save_dir"], "rave")
    model["base"]["frank_dir"] = os.path.join(model["base"]["save_dir"], "frank")
    model["base"]["parametric_dir"] = os.path.join(model["base"]["save_dir"], "parametric")
    
    subdirs = model["base"]["clean_dir"], model["base"]["rave_dir"], \
        model["base"]["frank_dir"], model["base"]["parametric_dir"]
    for dd in subdirs: 
        os.makedirs(dd, exist_ok=True)

    if disk_pars["base"]["SMG_sub"] is True:
        model["base"]["SMG_sub"] = "SMGsub."
    else:
        model["base"]["SMG_sub"] = ""

    model["base"]["dist"] = disk_pars["base"]["dist"]

    model["clean"]["npix"] = disk_pars["clean"]["npix"]
    model["clean"]["pixel_scale"] = disk_pars["clean"]["pixel_scale"]

    # get clean image rms
    image_pars = json.load(open(os.path.join(model["base"]["root_dir"], "pars_image.json"), 'r'))
    disk_image_pars = image_pars[model["base"]["disk"]]
    robusts, rmss = disk_image_pars["clean"]["image_robust"], disk_image_pars["clean"]["image_rms"]
    model["clean"]["image_rms"] = rmss[robusts.index(model["clean"]["robust"])]

    model["clean"]["bestfit"] = {}
    model["clean"]["bestfit"]["robust"] = disk_pars["clean"]["bestfit"]["robust"]

    model["rave"]["pixel_scale"] = disk_pars["rave"]["pixel_scale"]
    
    model["frank"]["bestfit"] = {}
    model["frank"]["bestfit"]["alpha"] = disk_pars["frank"]["bestfit"]["alpha"]
    model["frank"]["bestfit"]["wsmooth"] = disk_pars["frank"]["bestfit"]["wsmooth"]
    model["frank"]["bestfit"]["method"] = disk_pars["frank"]["bestfit"]["method"]

    # get source geom for clean profile extraction and frank fit
    mcmc = json.load(open(os.path.join(model["base"]["save_dir"], "MCMC_results.json"), 'r'))
    try: 
        dRA = mcmc["deltaRA-12mLB.obs1"]["median"]
        dDec = mcmc["deltaDec-12mLB.obs1"]["median"]
    except KeyError:
        dRA = mcmc["deltaRA-12m.obs1"]["median"]
        dDec = mcmc["deltaDec-12m.obs1"]["median"]

    geom = {"inc" : mcmc["i"]["median"],
            "PA" : mcmc["PA"]["median"], 
            "dRA" : dRA,
            "dDec" : dDec
            }
    model["base"]["geom"] = geom 
    print('    source geometry from MCMC {}'.format(geom))

    # stellar flux to remove from visibilities as point-source
    if model["frank"]["set_fstar"] == "custom":
        model["frank"]["fstar"] = disk_pars["frank"]["custom_fstar"] / 1e6
    elif model["frank"]["set_fstar"] == "SED":
        model["frank"]["fstar"] = disk_pars["frank"]["SED_fstar"] / 1e6
    elif model["frank"]["set_fstar"] == "MCMC":
        try:
            model["frank"]["fstar"] = mcmc["fstar"]["median"] / 1e3
        except KeyError:
            model["frank"]["fstar"] = 0.0
            print('        no stellar flux in MCMC file --> setting fstar = 0')
    else:
        raise ValueError("Parameter ['frank']['set_fstar'] is '{}'. It should be one of 'MCMC', 'SED', 'custom'".format(model["frank"]["set_fstar"])) 

    # enforce a Normal fit if finding scale height (LogNormal fit not compatible with vertical inference)
    if model["base"]["run_frank"] is True and model["frank"]["scale_heights"] is not None:
        print("    'scale_heights' is not None in your parameter file -- enforcing frank 'method=Normal' and 'max_iter=2000'")
        model["frank"]["method"] = "Normal"
        model["frank"]["max_iter"] = 2000

    # implemented functional forms 
    valid_funcs = [x for x in dir(arksia.parametric_forms) if not x.startswith('__')]
    if model["base"]["run_parametric"] is True:
        for pp in model["parametric"]["form"]:
            if pp not in valid_funcs:
                raise ValueError(f"{pp} is not one of {valid_funcs}")

    return model


def extract_clean_profile(model):
    """Obtain radial profiles from each of a CLEAN image and CLEAN model 

    Parameters
    ----------
    model : dict
        Dictionary containing pipeline parameters

    Returns
    -------
    tuple
        Radial points `r` [arcsec], brightness `I` [Jy/sr] and brightness 
        uncertainty `I_err` [Jy/sr] for each the CLEAN image profile' `r` and 
        `I` for the CLEAN model profile
    """
    # image filenames 
    base_path = "{}/{}.combined.{}corrected.briggs.{}.{}.{}".format(
        model["base"]["clean_dir"], 
        model["base"]["disk"], 
        model["base"]["SMG_sub"],
        model["clean"]["robust"], 
        model["clean"]["npix"], 
        model["clean"]["pixel_scale"]
        )

    # get image arrays
    clean_fits = base_path + ".pbcor.fits"
    pb_fits = base_path + ".pb.fits"
    model_fits = base_path + ".model.fits"

    clean_image, clean_beam = input_output.load_fits_image(clean_fits)
    bmaj, bmin = clean_beam
    pb_image = input_output.load_fits_image(pb_fits, aux_image=True)
    model_image = input_output.load_fits_image(model_fits, aux_image=True)

    # profile of clean image.
    # for radial profile on east side of disk,
    # range in azimuth (PA +- range) over which to average 
    f = 1.3
    phic_rad = extract_radial_profile.find_phic(model["base"]["geom"]["inc"] * np.pi / 180, f)
    phic_deg = phic_rad / deg_to_rad
    

    print('  Clean profiles: extracting profiles from {} and {} using phi_crit {:.2f} deg'.format(clean_fits, model_fits, phic_deg))

    phis_E = np.linspace(model["base"]["geom"]["PA"] - phic_deg, 
                        model["base"]["geom"]["PA"] + phic_deg, 
                        model["clean"]["Nphi"]
                        ) 
    
    phis_W = phis_E + 180

    # radial profile of east and west sides
    r_E, I_E, I_err_E = extract_radial_profile.radial_profile_from_image(
        clean_image, geom=model["base"]["geom"], phis=phis_E, bmaj=bmaj, 
        bmin=bmin, pb_image=pb_image, **model["clean"])
    r_W, I_W, I_err_W = extract_radial_profile.radial_profile_from_image(
        clean_image, geom=model["base"]["geom"], phis=phis_W, bmaj=bmaj,
        bmin=bmin, pb_image=pb_image, **model["clean"])

    # average of E and W
    r, I, I_err = r_W, np.mean((I_E, I_W), axis=0), np.hypot(I_err_E, I_err_W) / 2


    # profile of CLEAN .model image.
    # average across all azimuths (no need to take separate E and W profiles)
    phis_mod = np.linspace(model["base"]["geom"]["PA"] - 180, 
                                  model["base"]["geom"]["PA"] + 180,
                                  model["clean"]["Nphi"] 
                                  )
    
    r_mod, I_mod = extract_radial_profile.radial_profile_from_image(
        model_image, geom=model["base"]["geom"], phis=phis_mod, bmaj=0, 
        bmin=0, model_image=True, **model["clean"])

    # radial profile save paths
    ciff = "{}/clean_profile_robust{}.txt".format(
        model["base"]["clean_dir"], model["clean"]["robust"])
    cmff = "{}/clean_model_profile_robust{}.txt".format(
        model["base"]["clean_dir"], model["clean"]["robust"])    
    
    print('    saving CLEAN image profile to {} and model profile to {}'.format(ciff, cmff))
    np.savetxt(ciff, 
        np.array([r, I, I_err]).T, 
        header='Extracted from {}\nr [arcsec]\tI [Jy/sr]\tI_err [Jy/sr]'.format(
            clean_fits.split('/')[-1])
        )
    np.savetxt(cmff,
        np.array([r_mod, I_mod]).T,        
        header='Extracted from {}\nr [arcsec]\tI [Jy/sr]'.format(
            model_fits.split('/')[-1])
        )
 

def process_rave_fit(model):
    """Unpack a fitted RAVE radial profile, convert units, save

    Parameters
    ----------
    model : dict
        Dictionary containing pipeline parameters

    Returns
    -------
    list
        Radial points `r` [arcsec], brightness `I` [Jy/sr] and brightness 
        uncertainty `I_err` [Jy/sr] for the RAVE radial profile
    """
    
    fit_path = input_output.parse_rave_filename(model, file_type='rave_fit')
    print('  Rave profiles: processing {}'.format(fit_path))

    r, I_err_lo, I, I_err_hi = np.load(fit_path)
    I_err_lo = I - I_err_lo
    I_err_hi = I_err_hi - I

    I = jy_convert(I, 'arcsec2_sterad')
    I_err_lo = jy_convert(I_err_lo, 'arcsec2_sterad')
    I_err_hi = jy_convert(I_err_hi, 'arcsec2_sterad')    

    ff = "{}/rave_profile_robust{}.txt".format(
        model["base"]["rave_dir"], model["clean"]["robust"])
    print('    saving RAVE profile to {}'.format(ff))

    np.savetxt(ff, 
        np.array([r, I, I_err_lo, I_err_hi]).T, 
        header='Extracted from {}\nr [arcsec]\tI [Jy/sr]\tI_err (lower bound) [Jy/sr]\tI_err (upper bound) [Jy/sr]'.format(
            fit_path.split('/')[-1])
        )


def run_frank(model):
    """Perform a frank fit

    Parameters
    ----------
    model : dict
        Dictionary containing pipeline parameters

    Returns
    -------
    sols : list of _HankelRegressor objects
        frank fit solutions for each set of hyperparameters 
        (see frank.radial_fitters.FrankFitter)
    """
    print(' Frank fit: running fit')

    uv_data = input_output.get_vis(model)

    print('    shifting visibilities down by fstar {} uJy according to {}'.format(model["frank"]["fstar"] * 1e6, model["frank"]["set_fstar"]))
    uv_data[2] = uv_data[2] - model["frank"]["fstar"]
    
    frank_geom = FixedGeometry(**model["base"]["geom"]) 

    # set scale height
    if model["frank"]["scale_heights"] is None:
        hs = [0]
    else:
        hs = np.logspace(*model["frank"]["scale_heights"])
        print("    aspect ratios to be sampled: {}".format(hs))      

    # perform frank fit(s)
    def frank_fitter(priors):
        alpha, wsmooth, h = priors 
        print('        performing fit for alpha {} wsmooth {} h {}'.format(alpha, wsmooth, h))

        if h == 0:
            scale_height = None  
        else:
            def scale_height(R):
                return h * R
            
        # FrankDebrisFitter assumes disk is optically thin
        FF = FrankDebrisFitter(Rmax=model["frank"]["rout"], 
                                N=model["frank"]["N"], 
                                geometry=frank_geom,
                                scale_height=scale_height, 
                                alpha=alpha,
                                weights_smooth=wsmooth,
                                method=model["frank"]["method"],
                                I_scale=model["frank"]["I_scale"],
                                max_iter=model["frank"]["max_iter"],
                                convergence_failure='warn'
                                )
        
        sol = FF.fit(*uv_data)

        # add non-negative brightness profile to sol
        if model["frank"]["method"] == "Normal":
            setattr(sol, '_nonneg', sol.solve_non_negative())

        if scale_height is None:
            logev = None
        else:
            logev = FF.log_evidence_laplace()
        # add evidence to sol
        setattr(sol, 'logevidence', logev)

        # save fit outputs
        save_prefix = "{}/{}_alpha{}_w{}_h{:.3f}_fstar{:.0f}uJy_method{}".format(
                    model["base"]["frank_dir"], model["base"]["disk"], 
                    alpha, wsmooth, h,
                    model["frank"]["fstar"] * 1e6,
                    model["frank"]["method"]
                    )

        print("          saving fit results to {}*".format(save_prefix))
        save_fit(*uv_data, 
                sol=sol,
                prefix=save_prefix,
                save_solution=model["frank"]["save_solution"],
                save_profile_fit=model["frank"]["save_profile_fit"],
                save_vis_fit=model["frank"]["save_vis_fit"],
                save_uvtables=model["frank"]["save_uvtables"]
                )

        # save fit summary figures
        if model["frank"]["make_quick_fig"]:
            qfig, qaxes = make_quick_fig(*uv_data, sol, bin_widths=model["plot"]["bin_widths"],
                        save_prefix=None,
                        )
            # add 1 sigma statistical uncertainty band to plot
            sigmaI = get_fit_stat_uncer(sol)
            qaxes[0].fill_between(sol.r, (sol.I - sigmaI) / 1e10, (sol.I + sigmaI) / 1e10, color='r', alpha=0.2)
            qaxes[1].fill_between(sol.r, (sol.I - sigmaI) / 1e10, (sol.I + sigmaI) / 1e10, color='r', alpha=0.2)
            plt.savefig(save_prefix + '_frank_fit_quick.png', dpi=300,
                        bbox_inches='tight')
            plt.close()            

            # reprojected frank residual visibilities
            frank_resid_vis = [uv_data[0], uv_data[1], uv_data[2] - sol.predict(uv_data[0], uv_data[1]), uv_data[3]]
            plot.frank_image_diag_figure(model, sol, frank_resid_vis, save_prefix=save_prefix)

        return sol

    nfits = len((model["frank"]["alpha"])) * len(model["frank"]["wsmooth"]) * len(hs)
    if nfits == 1:
        # import frank; frank.enable_logging()
        sol = frank_fitter([model["frank"]["alpha"][0], model["frank"]["wsmooth"][0], hs[0]])
        return sol
    
    else: 
        # use as many threads as there are fits, up to a maximum of 'model["frank"]["nthreads"]'
        nthreads = min(nfits, model["frank"]["nthreads"])
        pool = multiprocess.Pool(processes=nthreads)
        print('    {} fits will be performed. Using {} threads (1 thread per fit; {} threads available on CPU).'.format(nfits, 
                                                                                             nthreads, 
                                                                                             multiprocess.cpu_count())
                                                                                             )

        # grids of prior values
        g0, g1, g2 = np.meshgrid(model["frank"]["alpha"], model["frank"]["wsmooth"], hs)
        g0, g1, g2 = g0.flatten(), g1.flatten(), g2.flatten()
        priors = np.array([g0, g1, g2]).T
        # run fits over grids
        sols = pool.map(frank_fitter, priors)

        logevs = []
        for ss in sols:
            logevs.append(ss.logevidence)

        # save scale heights and log evidences
        if np.array(hs).any() != 0:
            ff = "{}/vertical_inference_frank.txt".format(model["base"]["save_dir"])
            print("    saving h and log evidence results to {}".format(ff))
            np.savetxt(ff,
                np.array([g0, g1, g2, logevs]).T, header='alpha\twsmooth\th=H/r\tlog evidence'
            )

        return sols
    

def fit_parametric(fits, model):
    """
    # TODO
    """

    # get frank best-fit profile
    _, _, results = fits
    frank_profile = results[0]

    # run parametric fits
    figs = []
    PFits = []
    for pp in model['parametric']['form']:
        PFit = parametric_fitter.ParametricFit(frank_profile, 
                                            model, 
                                            func_form=pp,
                                            learn_rate=model['parametric']['learn_rate'], 
                                            niter=int(model['parametric']['niter']))
        PFit.fit()
        PFits.append(PFit)

        print(f"    initial params {PFit.initial_params}\n    final {PFit.bestfit_params}\n    loss {PFit.loss_history}")

        ff = f"{model['base']['parametric_dir']}/parametric_fit_{pp}.obj"
        print(f"    saving parametric fit results to {ff}")
        with open(ff, 'wb') as f:
            pickle.dump(PFit, f)

        # plot results
        fig = plot.parametric_fit_figure(PFit, frank_profile, model)
        figs.append(fig)

    return PFits, frank_profile, figs


def main(*args):
    """Run a pipeline to extract/fit/analyze radial profiles using 
    clean/rave/frank for ARKS data.

    Parameters
    ----------
    *args : strings
        Simulates the command line arguments
    """

    parsed_args = parse_parameters(*args)
    model = model_setup(parsed_args)
    
    if model["base"]["extract_clean_profile"] is True:
        extract_clean_profile(model)

    if model["base"]["process_rave_fit"] is True:
        process_rave_fit(model)

    if model["base"]["run_frank"] is True:
        frank_sols = run_frank(model)

    if model["base"]["compare_models_fig"] is True:
        fits = input_output.load_bestfit_profiles(model)   
        fig1 = plot.profile_comparison_figure(fits, model)
        fig2 = plot.image_comparison_figure(fits, model)

    if model["base"]["aspect_ratio_fig"] is True:
        fig3 = plot.aspect_ratio_figure(model)

    if model["base"]["run_parametric"] is True:
        fits = input_output.load_bestfit_profiles(model)
        parametric_fits, frank_profile, figs = fit_parametric(fits, model)

if __name__ == "__main__":
    main()
