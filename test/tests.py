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


