"""This module contains a function to run `pipeline` in a loop over multiple sources 
(written by Jeff Jennings)."""

import os
import json

def main(source_par_f='./pars_source.json', gen_par_f='./pars_gen.json'):
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