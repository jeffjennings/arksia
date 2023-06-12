Pipeline for radial profile (team 2) extraction and analysis. Contact Jeff Jennings with questions.

Dependencies
------------
- frank ('debris_update' branch - `pip install git+https://github.com/discsim/frank.git@debris_update`)
- MPoL (newest codebase - `pip install git+https://github.com/MPoL-dev/MPoL.git`)

Pipeline routines
-----------------
The pipeline is run from the terminal using input parameter files. As of June 2023, it has the following, modular capabilities:
- [x] extracts a radial brightness profile from a clean image
- [x] processes an existing rave fit to obtain a brightness profile and 1d, 2d residuals in consistent units
- [x] runs frank to obtain a brightness profile, obtain 1d residuals, image the 2d residuals (using MPoL)
- [x] runs frank for 1+1D vertical (disk aspect ratio) inference
- [x] produces plots to compare clean, rave, frank brightness profiles, radial visibility profiles, images, residuals
- [x] produces plots to assess frank vertical inference over grids of _h_, _alpha_, _wsmooth_
- [x] adds utilites to prepare visibility files for the above and to save/load/interface with all of the above
- [x] the pipeline runs from general and source-specific parameter files to do any combination of the above
- [x] the pipeline can be run in bulk (across multiple sources) to perform analysis and summarize results

Running the pipeline for a single source
----------------------------------------
The main pipeline file is `radial_pipeline.py`. It can be run from the terminal for fits/analysis of a single source with `python radial_pipeline.py -d '<disk name>'`, e.g., `python radial_pipeline.py -d 'HD76582'`. By default the pipeline runs using the parameter files `./default_gen_pars.json` (contains parameters to choose which of the above pipeline modules run, as well as sensible choices for the pipeline parameters applicable to all sources) and `./default_source_pars.json` (contains sensible choices for source-specific, best-fit parameters). For descriptions of the `.json` parameters, see `gen_pars_description.json` and `source_pars_description.json`.

### Setting up frank fits ###
- For running frank, you will likely want to adjust the `alpha`, `wsmooth` and `scale_heights` parameters in `./default_gen_pars.json`. 

- When performing frank fits to find a radial profile, I recommend setting `method` to `"LogNormal"` to perform fits in logarithmic brightness space. Not all parts of the pipeline support linear brightness space fits with enforced non-negativity; this is because the logarithmic fits are in general a better choice. The exception is that when running a frank 1+1D fit to find _h_, `method` must be `"Normal"` (it will be enforced), because `"LogNormal"` is not supported within frank.

Running the pipeline for multiple/all sources
---------------------------------------------
The pipeline can be looped over multiple sources using `bulk_pipeline_run.py` via `python bulk_pipeline_run.py`. 

Obtaining key results for multiple/all sources
----------------------------------------------
Survey-wide results are a `.txt` file per source with all radial brightness profiles (clean, rave, frank) sampled at the same radii, and a figures with a panel for each source showing the clean, rave, frank brightness profiles (one figure without uncertainties, one figure with). These are generated with `bulk_pipeline_results.py`.