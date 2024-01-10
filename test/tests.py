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


