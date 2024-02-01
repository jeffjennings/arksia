"""This module is the main file for running radial pipeline fits and analysis 
(written by Jeff Jennings)."""

import os; os.environ.get('OMP_NUM_THREADS', '1')
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import multiprocess
import pickle 

# import frank; frank.enable_logging()
from frank.constants import deg_to_rad
from frank.utilities import jy_convert
from frank.debris_fitters import FrankDebrisFitter
from frank.io import save_fit
from frank.make_figs import make_quick_fig
from frank.geometry import FixedGeometry
from frank.utilities import get_fit_stat_uncer

import arksia 
from arksia import input_output, extract_radial_profile, plot, parametric_fitter

arksia_path = os.path.dirname(arksia.__file__)

def get_default_parameter_file():
    """Get the path to the default parameter file"""
    return os.path.join(arksia_path, 'pars_gen.json')

def load_default_parameters():
    """Load the default parameters"""
    return json.load(open(get_default_parameter_file(), 'r'))

def get_parameter_descriptions():
    """Get the description for parameters"""
    with open(os.path.join(arksia_path, 'description_pars_gen.json')) as f:
        params_gen = json.load(f)

    with open(os.path.join(arksia_path, 'description_pars_source.json')) as f:
        params_source = json.load(f)
    return [params_gen, params_source]

def helper():
    param_descrip = get_parameter_descriptions()

    print(f"""
         Run the modular 'arksia' pipeline with `python -m arksia.pipeline -d <disk name>`,
          where <disk name> is one of the sources in 'pars_source.json'.
          Required inputs are one .csv file containing disk physical parameters (obtained from the
          ARKS survey, by default named 'summary_disc_parameters.csv') and two 
          .json files (containing generic and source-specific 
          pipeline parameters). The default generic parameters 
          file is 'description_pars_gen.json' and is of the form:\n 
          {json.dumps(param_descrip[0], indent=4)}.\n\nThe default source-specific
          parameters file is 'description_pars_source.json' and is of the form:\n
          {json.dumps(param_descrip[1], indent=4)}""")
    
    
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
                        help="Parameter file (.json) with generic pipeline parameterss")
    
    parser.add_argument("-s", "--source_parameter_filename",
                        type=str,
                        default="./pars_source.json",
                        help="Parameter file (.json) with source-specific pipeline parameterss")

    parser.add_argument("-p", "--physical_parameter_filename",
                        type=str,
                        default="./summary_disc_parameters.csv",
                        help="Summary table (.csv) with source-specific physical parameters")    

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

    # generic pipeline parameters
    model = json.load(open(parsed_args.base_parameter_filename, 'r'))
    model["base"]["disk"] = parsed_args.disk

    print('\nRunning radial profile pipeline for {}'.format(model["base"]["disk"]))

    print(f"  Model setup: expecting all input files in {model['base']['input_dir']}")

    if model["base"]["output_dir"] is None:
        model["base"]["save_dir"] = model["base"]["input_dir"]
        print(f"  Model setup: 'output_dir' is None in your parameter file -- setting save path as {model['base']['save_dir']}")
    else:
        model["base"]["save_dir"] = model["base"]["output_dir"]
    print(f"    Results will be saved in {model['base']['save_dir']}/<frank, clean, rave, parametric>")

    model["base"]["clean_dir"] = os.path.join(model["base"]["save_dir"], "clean")
    model["base"]["rave_dir"] = os.path.join(model["base"]["save_dir"], "rave")
    model["base"]["frank_dir"] = os.path.join(model["base"]["save_dir"], "frank")
    model["base"]["parametric_dir"] = os.path.join(model["base"]["save_dir"], "parametric")

    # source-specific pipeline parameters
    source_pars = json.load(open(parsed_args.source_parameter_filename, 'r'))
    disk_pars = source_pars[model["base"]["disk"]]

    # source-specific physical parameters
    phys_pars = None
    with open(parsed_args.physical_parameter_filename) as ff:
        reader = csv.DictReader(ff)
        for row in reader:
            if row['name'].replace(' ', '') == model["base"]["disk"]:
                phys_pars = row
                break
        if phys_pars is None: 
            raise ValueError(f"Disk name {model['base']['disk']} not found in {parsed_args.physical_parameter_filename}")

    for key, val in phys_pars.items():
        if key != "name" and val not in ['', '--']:
            phys_pars[key] = float(val)

    model["base"]["dist"] = phys_pars["dpc"]
    # source geom for clean profile extraction and frank fit
    model["base"]["geom"] = {
        "inc" : phys_pars["i"],
        "PA" : phys_pars["PA"], 
        "dRA" : phys_pars["deltaRA"],
        "dDec" : phys_pars["deltaDec"]
        }
    print(f"  Model setup: provided source geometry {model['base']['geom']}")

    if disk_pars["base"]["SMG_sub"] is True:
        model["base"]["SMG_sub"] = "SMGsub."
    else:
        model["base"]["SMG_sub"] = ""

    model["clean"]["npix"] = disk_pars["clean"]["npix"]
    model["clean"]["pixel_scale"] = disk_pars["clean"]["pixel_scale"]

    if model["base"]["extract_clean_profile"] is True:
        os.makedirs(model["base"]["clean_dir"], exist_ok=True)
        model["clean"]["image_robust"] = disk_pars["clean"]["image_robust"] 
        model["clean"]["image_rms"] = disk_pars["clean"]["image_rms"]
        robusts, rmss = model["clean"]["image_robust"], model["clean"]["image_rms"]
        model["clean"]["image_rms"] = rmss[robusts.index(model["clean"]["robust"])]

    if model["base"]["process_rave_fit"] is True:
        os.makedirs(model["base"]["rave_dir"], exist_ok=True)
        model["rave"]["pixel_scale"] = disk_pars["rave"]["pixel_scale"]
    
    if True in [model["base"]["run_frank"], model["base"]["reproduce_best_frank"]]:
        if model["base"]["reproduce_best_frank"] is True:
            print("\n  Model setup: Overriding 'frank' choices in your parameter file to reproduce best-fit model (without scale-height inference)\n")
      
            model["frank"]["rout"] = disk_pars["frank"]["bestfit"]["rout"]
            model["frank"]["N"] = disk_pars["frank"]["bestfit"]["N"]
            model["frank"]["alpha"] = [disk_pars["frank"]["bestfit"]["alpha"]]
            model["frank"]["wsmooth"] = [disk_pars["frank"]["bestfit"]["wsmooth"]]
            model["frank"]["method"] = disk_pars["frank"]["bestfit"]["method"]
            model["frank"]["I_scale"] = 1e2
            model["frank"]["max_iter"] = 1000
            model["frank"]["set_fstar"] = "MCMC"
            model["frank"]["scale_height"] = None
            model["frank"]["save_solution"] = True
            model["frank"]["save_profile_fit"] = True
            model["frank"]["save_vis_fit"] = True
            model["frank"]["save_uvtables"] = True
            model["frank"]["make_quick_fig"] = True
            model["plot"]["image_xy_bounds"] = [-disk_pars["frank"]["bestfit"]["rout"], disk_pars["frank"]["bestfit"]["rout"]]
            model["plot"]["frank_resid_im_robust"] = 0.5

        else:
            os.makedirs(model["base"]["frank_dir"], exist_ok=True)
            # handle non-list inputs
            if type(model["frank"]["alpha"]) in [int, float]:
                model["frank"]["alpha"] = [model["frank"]["alpha"]]
            if type(model["frank"]["wsmooth"]) in [int, float]:
                model["frank"]["wsmooth"] = [model["frank"]["wsmooth"]]
            if type(model["frank"]["scale_height"]) in [int, float]:
                model["frank"]["scale_height"] = [model["frank"]["scale_height"]]
            
            # enforce a Normal fit if finding scale height 
            # (LogNormal fit is not supported for vertical inference)
            if model["frank"]["scale_height"] is not None:
                print("Model setup: 'scale_height' is not None in your parameter file -- enforcing frank 'method=Normal' and 'max_iter=2000'")
                model["frank"]["method"] = "Normal"
                model["frank"]["max_iter"] = 2000

    if model["base"]["run_parametric"] is True:
        os.makedirs(model["base"]["parametric_dir"], exist_ok=True)
        # handle non-list input
        if type(model["parametric"]["form"]) is str:
            model["parametric"]["form"] = [model["parametric"]["form"]]

        # implemented functional forms 
        valid_funcs = [x for x in dir(arksia.parametric_forms) if not x.startswith('__')]
        for pp in model["parametric"]["form"]:
            if pp not in valid_funcs:
                raise ValueError(f"{pp} is not one of {valid_funcs}")

    # frank: stellar flux to remove from visibilities as point-source
    if model["frank"]["set_fstar"] == "custom":
        model["frank"]["fstar"] = disk_pars["frank"]["custom_fstar"] / 1e6
    elif model["frank"]["set_fstar"] == "SED":
        model["frank"]["fstar"] = phys_pars["Fstar_SED"] / 1e6
    elif model["frank"]["set_fstar"] == "MCMC":
        try:
            model["frank"]["fstar"] = phys_pars["Fstar_MCMC"] / 1e3
        except TypeError:
            print(f"Model setup: {parsed_args.physical_parameter_filename} has no stellar flux for {model['base']['disk']} --> using SED estimate of stellar flux")
            model["frank"]["fstar"] = phys_pars["Fstar_SED"] / 1e6
    else:
        raise ValueError(f"Parameter ['frank']['set_fstar'] {model['frank']['set_fstar']} must be one of ['MCMC', 'SED', 'custom']") 

    model["clean"]["bestfit"] = disk_pars["clean"]["bestfit"]
    model["frank"]["bestfit"] = disk_pars["frank"]["bestfit"]

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
        model["base"]["input_dir"], 
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
    try:
        pb_image = input_output.load_fits_image(pb_fits, aux_image=True)
    except FileNotFoundError:
        pb_image = None
    try: 
        model_image = input_output.load_fits_image(model_fits, aux_image=True)
    except FileNotFoundError:
        model_image = None

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

    # save radial profile
    ciff = "{}/clean_profile_robust{}.txt".format(
        model["base"]["clean_dir"], model["clean"]["robust"])
    
    print(f"    saving CLEAN image profile to {ciff}")
    np.savetxt(ciff, 
        np.array([r, I, I_err]).T, 
        header='Extracted from {}\nr [arcsec]\tI [Jy/sr]\tI_err [Jy/sr]'.format(
            clean_fits.split('/')[-1])
        )    

    if model_image is not None:
        # profile of CLEAN .model image.
        # average across all azimuths (no need to take separate E and W profiles)
        phis_mod = np.linspace(model["base"]["geom"]["PA"] - 180, 
                                    model["base"]["geom"]["PA"] + 180,
                                    model["clean"]["Nphi"] 
                                    )
        
        r_mod, I_mod = extract_radial_profile.radial_profile_from_image(
            model_image, geom=model["base"]["geom"], phis=phis_mod, bmaj=0, 
            bmin=0, model_image=True, **model["clean"])

        cmff = "{}/clean_model_profile_robust{}.txt".format(
            model["base"]["clean_dir"], model["clean"]["robust"])    
    
        print(f"    saving CLEAN model profile to {cmff}")
        np.savetxt(cmff,
            np.array([r_mod, I_mod]).T,        
            header='Extracted from {}\nr [arcsec]\tI [Jy/sr]'.format(
                model_fits.split('/')[-1])
            )
 
    clean_diag_fig = plot.clean_diag_figure(model, clean_image, [r, I, I_err], model_image, [r_mod, I_mod])
    

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
    print(" Frank fit: running fits")

    uv_data = input_output.get_vis(model)

    print('    shifting visibilities down by {} fstar {} uJy '.format(model["frank"]["set_fstar"], model["frank"]["fstar"] * 1e6))
    uv_data[2] = uv_data[2] - model["frank"]["fstar"]
    
    frank_geom = FixedGeometry(**model["base"]["geom"]) 

    # set scale height
    if model["frank"]["scale_height"] is None:
        hs = [0]

        # pre-process visibilities for multiple frank fits (only valid for a 
        # fixed scale height
        FF_pre = FrankDebrisFitter(Rmax=model["frank"]["rout"], 
                                N=model["frank"]["N"], 
                                geometry=frank_geom,
                                scale_height=None,
                                alpha=0,
                                weights_smooth=0,
                                method=model["frank"]["method"],
                                I_scale=model["frank"]["I_scale"],
                                max_iter=model["frank"]["max_iter"],
                                convergence_failure='warn'
                                )
        ppV = FF_pre.preprocess_visibilities(*uv_data)

    else: 
        if len(model["frank"]["scale_height"]) == 1:
            hs = model["frank"]["scale_height"] * 1
        else:
            hs = np.logspace(*model["frank"]["scale_height"])
        print("    aspect ratios to be sampled: {}".format(hs))

    # perform frank fit(s)
    def frank_fitter(priors):
        alpha, wsmooth, h = priors 
        print(f"        performing {model['frank']['method']} fit for alpha {alpha} wsmooth {wsmooth} h {h}")

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

        if scale_height is None:
            sol = FF.fit_preprocessed(ppV)
        else:
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
        save_prefix = "{}/{}_alpha{}_w{}_rout{}_h{:.3f}_fstar{:.0f}uJy_method{}".format(
                    model["base"]["frank_dir"], model["base"]["disk"], 
                    alpha, wsmooth, sol.Rmax, h,
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
        if model["frank"]["make_quick_fig"] is True:
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
            plot.frank_image_diag_figure(model, sol, frank_resid_vis, 
                                         xy_bounds=model["plot"]["image_xy_bounds"], 
                                         resid_im_robust=model["plot"]["frank_resid_im_robust"],
                                         save_prefix=save_prefix
                                         )

        return sol, FF

    nfits = len((model["frank"]["alpha"])) * len(model["frank"]["wsmooth"]) * len(hs)
    print(f"    {nfits} frank fits will be performed")

    if nfits == 1:
        sol, _ = frank_fitter([model["frank"]["alpha"][0], model["frank"]["wsmooth"][0], hs[0]])
        return sol
    
    else: 
        # commenting in favor of approach that saves compute by pre-processing visibilities
        # use as many threads as there are fits, up to a maximum of 'model["frank"]["nthreads"]'
        # nthreads = min(nfits, model["frank"]["nthreads"])
        # pool = multiprocess.Pool(processes=nthreads)
        # print('    {} fits will be performed. Using {} threads (1 thread per fit; {} threads available on CPU).'.format(nfits, 
        #                                                                                      nthreads, 
        #                                                                                      multiprocess.cpu_count())
        #                                                                                      )

        # grids of prior values
        g0, g1, g2 = np.meshgrid(model["frank"]["alpha"], model["frank"]["wsmooth"], hs)
        g0, g1, g2 = g0.flatten(), g1.flatten(), g2.flatten()
        priors = np.array([g0, g1, g2]).T

        # run fits over grids
        # sols = pool.map(frank_fitter, priors)

        sols = []       
        for nn, prior in enumerate(priors): 
            print(f"      Fit {nn + 1} of {nfits}") 
            sol, FF = frank_fitter(prior)

            if hs[0] == 0:
                logev = None
            else:
                logev = FF.log_evidence_laplace()
            # add evidence to sol
            setattr(sol, 'logevidence', logev)

            sols.append(sol)

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

        multifit_save_prefix = "{}/{}_fstar{:.0f}uJy_method{}".format(
                    model["base"]["frank_dir"], model["base"]["disk"], 
                    model["frank"]["fstar"] * 1e6,
                    model["frank"]["method"]
                    )

        if model["base"]["frank_multifit_fig"] is True:
            # plot all fits (single panel)
            plot.frank_multifit_figure(model, sols, plot_var="I", single_panel=True,
                                            save_prefix=multifit_save_prefix)
            plot.frank_multifit_figure(model, sols, plot_var="V", single_panel=True,
                                            save_prefix=multifit_save_prefix)        
            
            # plot all I(r) fits (panel grid)
            plot.frank_multifit_figure(model, sols, plot_var="I", 
                                            save_prefix=multifit_save_prefix)
            plot.frank_multifit_figure(model, sols, plot_var="V", 
                                            save_prefix=multifit_save_prefix)

        return sols
    

def fit_parametric(fits, model):
    """Perform one or more parametric fits to a nonparametric brightness profile

    Parameters
    ----------
    fits : nested list
        Best-fit real space and Fourier radial profiles for clean, rave, frank. 
        Ouput of `input_output.load_bestfit_profiles`
    model : dict
        Dictionary containing pipeline parameters

    Returns
    -------
    PFits : list of `ParametricFit` objects
        Results of the parameteric fitting optimization
        (see `parametric_fitter.ParametricFit`)
    fit_region : list
        The brightness profile [radial points, brightness, uncertainty] 
        to which a parametric form is being fit
    figs : list of `plt.figure`
        Generated figures, each showing the fit for one parametric form
    """

    print(f"  Parametric fit: fitting {model['parametric']['form']} to frank best-fit profile")

    # get frank best-fit profile
    _, _, results = fits
    frank_profile = results[0]

    fit_region = frank_profile * 1
    
    if model['parametric']['fit_range'] is not None:
        print(f"    restricting fit region to {model['parametric']['fit_range']} arcsec")
        # restrict region of profile to fit parametric form to
        idx = [i for i,x in enumerate(frank_profile[0]) if model['parametric']['fit_range'][0] <= x <= model['parametric']['fit_range'][1]]
        fit_region[0] = fit_region[0][idx]
        fit_region[1] = fit_region[1][idx]
        fit_region[2] = fit_region[2][idx]

    # run parametric fits
    figs = []
    PFits = []
    for pp in model['parametric']['form']:
        print(f"    fitting parametric form {pp}")
        PFit = parametric_fitter.ParametricFit(fit_region, 
                                            model, 
                                            func_form=pp,
                                            learn_rate=model['parametric']['learn_rate'], 
                                            niter=int(model['parametric']['niter']))
        PFit.fit()
        PFits.append(PFit)

        print(f"      initial params {PFit.initial_params}\n      final {PFit.bestfit_params}\n      loss {PFit.loss_history}")

        ff = f"{model['base']['parametric_dir']}/parametric_fit_{pp}.obj"
        print(f"      saving fit results to {ff}")
        with open(ff, 'wb') as f:
            pickle.dump(PFit, f)

        # plot results
        fig = plot.parametric_fit_figure(model, PFit, frank_profile, fit_region)
        figs.append(fig)

    return PFits, fit_region, figs


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

    if model["base"]["compare_models_fig"] is not None:
        if model["base"]["compare_models_fig"] == "all":
            fits = input_output.load_bestfit_profiles(model)
        else:
            fits = input_output.load_bestfit_profiles(model, include_rave=False)

        include_rave = False
        # if there are rave fits, include them in figures
        if fits[1] is not None:
            include_rave = True

        fig1 = plot.profile_comparison_figure(fits, 
                                              model,
                                              resid_im_robust=model["plot"]["frank_resid_im_robust"],
                                              include_rave=include_rave,
                                              )
        
        fig2 = plot.image_comparison_figure(fits, 
                                            model, 
                                            xy_bounds=model["plot"]["image_xy_bounds"],
                                            resid_im_robust=model["plot"]["frank_resid_im_robust"],
                                            include_rave=include_rave,
                                            )

    if model["frank"]["scale_height"] is not None and model["base"]["aspect_ratio_fig"] is True:
        fig3 = plot.aspect_ratio_figure(model)

    if model["base"]["run_parametric"] is True:
        fits = input_output.load_bestfit_profiles(model, include_clean=False, include_rave=False)
        parametric_fits, fit_region, figs = fit_parametric(fits, model)

if __name__ == "__main__":
    main()
