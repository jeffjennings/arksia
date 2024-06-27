"""This module contains a function to run the arksia pipeline over multiple sources 
(written by Jeff Jennings)."""

import os
import json

def survey_pipeline(gen_par_f='./pars_gen.json', 
         source_par_f='./pars_source.json',
         phys_par_f='./summary_disc_parameters.csv',
        reproduce_best_frank_fits=False,
         ):
    """
    Run the arksia pipeline across multiple survey sources.

    Parameters
    ----------       
    gen_par_f : string, default='pars_gen.json'
        Path to the general parameters file  
    source_par_f : string, default='pars_source.json'
        Path to the parameter file with custom values for each source         
    phys_par_f : string, default='pars_gen.json'
        Path to the physical parameters file
    reproduce_best_frank_fits : bool, default=False
        Whether to reproduce frank fits with the best fit values in `pars_source.json`
    """    
    
    disk_names = []
    source_pars = json.load(open(source_par_f, 'r'))
    for ii in source_pars:
        disk_names.append(ii)
    
    gen_pars = json.load(open(gen_par_f, 'r'))
    # run the radial pipeline over each source in `disk_names`
    for ii, jj in enumerate(disk_names):
        print(f"\nPipeline call {ii + 1} of {len(disk_names)} - disk {jj}")

        gen_pars['base']['input_dir'] = f"./{jj}"
        if reproduce_best_frank_fits is True:
            gen_pars["base"]["reproduce_best_frank"] = True

        gen_pars_current = os.path.join(os.path.dirname(gen_par_f), 'pars_gen_temp.json')
        with open(gen_pars_current, 'w') as f:
            json.dump(gen_pars, f)

        os.system(f"python -m arksia.pipeline -d {jj} -b {gen_pars_current} -s {source_par_f} -p {phys_par_f}")

    os.remove(gen_pars_current)

if __name__ == "__main__":
    survey_pipeline()