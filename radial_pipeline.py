"""Main file for radial pipeline fits/analysis."""

import os; os.environ.get('OMP_NUM_THREADS', '1')
import json
import argparse
import numpy as np
import multiprocess

from frank.constants import deg_to_rad
from frank.utilities import jy_convert
from frank.debris_fitters import FrankDebrisFitter
from frank.io import save_fit
from frank.make_figs import make_quick_fig
from frank.geometry import FixedGeometry

from image_radial_profile import radial_profile_from_image, find_phic
from io import load_fits_image, get_vis, load_bestfit_profiles
from plot import profile_comparison_figure, image_comparison_figure, aspect_ratio_figure, survey_summary

def parse_parameters(*args):
    """
    Read in a .json parameter files to run the pipeline

    Parameters
    ----------
    parameter_filename : str
        Parameter file (.json)

    Returns
    -------
    model : dict
        Dictionary containing model parameters the pipeline uses
    """

    parser = argparse.ArgumentParser("Run the radial profile pipeline for ARKS data")

    parser.add_argument("-d", "--disk",
                        type=str,
                        help="Disk name")

    parser.add_argument("-b", "--base_parameter_filename",
                        type=str,
                        default="./default_gen_pars.json",
                        help="Parameter file (.json) with generic pars")
    
    parser.add_argument("-s", "--source_parameter_filename",
                        type=str,
                        default="./default_source_pars.json",
                        help="Parameter file (.json) with source-specific pars")

    args = parser.parse_args(*args)
    
    return args


def model_setup(parsed_args):
    # generic parameters
    model = json.load(open(parsed_args.base_parameter_filename, 'r'))
    model["base"]["disk"] = parsed_args.disk

    print('\nRunning radial profile pipeline for {}'.format(model["base"]["disk"]))

    # disk-specific parameters
    source_pars = json.load(open(parsed_args.source_parameter_filename, 'r'))
    disk_pars = source_pars[model["base"]["disk"]]

    if not model["base"]["save_dir"]:
        # If not specified, use the directory with `gen_pars.json` to set the save directory
        model["base"]["save_dir"] = os.path.join(os.path.dirname(parsed_args.base_parameter_filename), "disks/{}".format(model["base"]["disk"]))
        print("Setting load/save paths as {}/<clean, rave, frank>. Visibility tables should be in frank path.".format(model["base"]["save_dir"]))
    else:
        print("Assuming load/save paths are {}/<clean, rave, frank>. Visibility tables should be in frank path.".format(model["base"]["save_dir"]))
    model["base"]["clean_dir"] = os.path.join(model["base"]["save_dir"], "clean")
    model["base"]["rave_dir"] = os.path.join(model["base"]["save_dir"], "rave")
    model["base"]["frank_dir"] = os.path.join(model["base"]["save_dir"], "frank")

    if disk_pars["base"]["SMG_sub"] is True:
        model["base"]["SMG_sub"] = "SMGsub."
    else:
        model["base"]["SMG_sub"] = ""

    model["base"]["dist"] = disk_pars["base"]["dist"]

    model["clean"]["npix"] = disk_pars["clean"]["npix"]
    model["clean"]["pixel_scale"] = disk_pars["clean"]["pixel_scale"]
    model["clean"]["image_rms"] = disk_pars["clean"]["image_rms"]

    model["rave"]["pixel_scale"] = disk_pars["rave"]["pixel_scale"]
    
    model["frank"]["bestfit"] = {}
    model["frank"]["bestfit"]["alpha"] = disk_pars["frank"]["bestfit"]["alpha"]
    model["frank"]["bestfit"]["wsmooth"] = disk_pars["frank"]["bestfit"]["wsmooth"]
    model["frank"]["bestfit"]["method"] = disk_pars["frank"]["bestfit"]["method"]

    # get source geom for clean profile extraction and frank fit
    mcmc = json.load(open(os.path.join(model["base"]["save_dir"], "MCMC_results.json"), 'r'))
    geom = {"inc" : mcmc["i"]["median"],
            "PA" : mcmc["PA"]["median"], 
            "dRA" : mcmc["deltaRA-12m.obs1"]["median"], 
            "dDec" : mcmc["deltaDec-12m.obs1"]["median"]
            }
    model["base"]["geom"] = geom 
    print('  source geometry from MCMC {}'.format(geom))

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
            print('  no stellar flux in MCMC file --> setting fstar = 0')
    else:
        raise ValueError("Parameter ['frank']['set_fstar'] is '{}'. It should be one of 'MCMC', 'SED', 'custom'".format(model["frank"]["set_fstar"])) 

    # enforce a Normal fit if finding scale height (LogNormal fit not compatible with vertical inference)
    if model["frank"]["scale_heights"] is not None:
        print("'scale_heights' is not None in your parameter file -- enforcing 'method=Normal' with 'max_iter=2000'")
        model["frank"]["method"] = "Normal"
        model["frank"]["max_iter"] = 2000

    return model
