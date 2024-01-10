"""This module runs tests to confirm the code is working correctly."""

import os 
import json
import numpy as np 

from arksia import pipeline

def save_custom_gen_pars(gen_pars):
    """Save an altered generic parameters file"""

    tmp_dir = '/tmp/arksia/tests'
    os.makedirs(tmp_dir, exist_ok=True)
    gen_pars['base']['output_dir'] = tmp_dir

    gen_pars_file = os.path.join(tmp_dir, 'gen_pars.json')
    with open(gen_pars_file, 'w') as f:
        json.dump(gen_pars, f)

    return gen_pars_file


def update_frank_pars(gen_pars):
    """Update default generic parameters for tests using frank"""
    gen_pars['base']['run_frank'] = True
    gen_pars['frank']['rout'] = 1.5
    gen_pars['frank']['N'] = 50
    gen_pars['frank']['alpha'] = [1.5]
    gen_pars['frank']['wsmooth'] = [1e-1]
    gen_pars['frank']['scale_height'] = None    
    gen_pars['frank']['method'] = 'Normal'

    return gen_pars


def _run_pipeline(gen_pars_file):
    """Generic routine to invoke pipeline"""

    arksia_path = pipeline.arksia_path
    # Dummy source-specific parameters file
    source_pars_file = os.path.join(arksia_path, '../test/mock_pars_source.json')

    # Dummy physical parameters file
    phys_pars_file = os.path.join(arksia_path, '../test/mock_pars_phys.cv')

    # Call pipeline
    pipeline.main(['-b', gen_pars_file, '-s', source_pars_file, '-p', phys_pars_file, '-d', 'mockAS209'])


def test_pipeline_frank_fit():
    """Run the pipeline to perform a frank fit (and save fit diagnostics)"""

    # Default generic parameters file
    gen_pars = pipeline.load_default_parameters()

    gen_pars = update_frank_pars(gen_pars)

    gen_pars_file = save_custom_gen_pars(gen_pars)

    _run_pipeline(gen_pars_file)


def test_pipeline_frank_logfit():
    """Run the pipeline to perform a frank fit in log(brightness)"""

    # Default generic parameters file
    gen_pars = pipeline.load_default_parameters()

    gen_pars = update_frank_pars(gen_pars)
    
    gen_pars['frank']['method'] = 'LogNormal'

    gen_pars_file = save_custom_gen_pars(gen_pars)

    _run_pipeline(gen_pars_file)


def test_pipeline_frank_multifit():
    """Run the pipeline to perform multiple frank fits and produce the multi-fit figures"""

    gen_pars = pipeline.load_default_parameters()

    gen_pars = update_frank_pars(gen_pars)

    gen_pars['base']['frank_multifit_fig'] = True
    gen_pars['frank']['alpha'] = [1.5, 1.3]

    gen_pars_file = save_custom_gen_pars(gen_pars)

    _run_pipeline(gen_pars_file)


def test_pipeline_frank_vertical_fit():
    """Run the pipeline to perform a frank fit with vertical inference"""

    gen_pars = pipeline.load_default_parameters()

    gen_pars = update_frank_pars(gen_pars)

    gen_pars['frank']['scale_height'] = 1e-1

    gen_pars_file = save_custom_gen_pars(gen_pars)

    _run_pipeline(gen_pars_file)


def test_pipeline_frank_vertical_multifit():
    """Run the pipeline to perform multiple frank fits with vertical inference and produce the aspect ratio figure"""

    gen_pars = pipeline.load_default_parameters()

    gen_pars = update_frank_pars(gen_pars)

    gen_pars['base']['aspect_ratio_fig'] = True
    gen_pars['frank']['scale_height'] = [1e-1, 1.1e-1]

    gen_pars_file = save_custom_gen_pars(gen_pars)

    _run_pipeline(gen_pars_file)

