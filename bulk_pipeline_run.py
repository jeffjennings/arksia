"""File to run `radial_pipeline` in a loop over multiple sources."""

import os
import json

def main(gen_par_f='default_gen_pars.json', source_par_f='default_source_pars.json'):
    gen_pars = json.load(open(gen_par_f, 'r'))
    source_pars = json.load(open(source_par_f, 'r'))

    disk_names = []
    for ii in source_pars:
        disk_names.append(ii)

    # run the radial pipeline over each source in `disk_names`
    for ii, jj in enumerate(disk_names):
        print("\nPipeline call {} of {} - disk {}".format(ii, len(disk_names) - 1, jj))

        os.system('python radial_profile_pipeline.py -d {} -b {} -s {}'.format(
        jj, gen_pars, source_pars))

if __name__ == "__main__":
    main()