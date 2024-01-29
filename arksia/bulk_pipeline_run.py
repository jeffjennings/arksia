"""This module contains a function to run the arksia pipeline over multiple sources 
(written by Jeff Jennings)."""

import os
import json

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
    """    
    
    disk_names = []
    source_pars = json.load(open(source_par_f, 'r'))
    for ii in source_pars:
        disk_names.append(ii)

    # run the radial pipeline over each source in `disk_names`
    for ii, jj in enumerate(disk_names):
        print("\nPipeline call {} of {} - disk {}".format(ii, len(disk_names) - 1, jj))

        os.system('python -m arksia.pipeline -d {} -b {} -s {}'.format(
            jj, gen_par_f, source_par_f))

if __name__ == "__main__":
    main()