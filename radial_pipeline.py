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

