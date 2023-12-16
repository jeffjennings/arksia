"""This module runs tests to confirm the code is working correctly."""

import os 
import json
import numpy as np 

from arksia import pipeline
    
def load_test_data():
    """Load dataset used for testing"""
    dset = np.load('test/test1.npz')

# def test_model_setup():

def _run_pipeline(make_figs=False, multifit=False):
    
    tmp_dir = '/tmp/arksia/tests'
    os.makedirs(tmp_dir, exist_ok=True)

    # Save the new parameter file
    params = []
    param_file = os.path.join(tmp_dir, 'gen.json')
    with open(param_file, 'w') as f:
        json.dump(params, f)

    # Call the pipeline to perform the fit
    pipeline.main(['-p', param_file])