Pipeline for radial profile (team 2) extraction and analysis. Contact Jeff Jennings with questions.

Dependencies
------------
- frank ('debris_update' branch - `pip install git+https://github.com/discsim/frank.git@debris_update`)
- MPoL (newest codebase - `pip install git+https://github.com/MPoL-dev/MPoL.git`)

Pipeline scope
--------------
The pipeline is run from the terminal using input parameter files. As of June 2023, it has the following, modular capabilities:
- extracts a radial brightness profile from a clean image
- processes an existing rave fit to obtain a brightness profile and 1d, 2d residuals in consistent units
- runs frank to obtain a brightness profile, obtain 1d residuals, image the 2d residuals (using MPoL)
- runs frank for 1+1D vertical (disk aspect ratio) inference
- produces plots to compare clean, rave, frank brightness profiles, radial visibility profiles, images, residuals
- produces plots to assess frank vertical inference over grids of _h_, _alpha_, _wsmooth_
- adds utilites to prepare visibility files for the above and to save/load/interface with all of the above
- the pipeline runs from general and source-specific parameter files to do any combination of the above
- the pipeline can be run in bulk (across multiple sources) to perform analysis and summarize results

Prior to running the pipeline for a new source
----------------------------------------------
Before running any pipeline routines:
1) Create the following directory structure:
- Root directory: '<disk name>'
    - Subdirectories: 'clean', 'frank', 'rave'

2) Download and place the following files in these directories:
- root dir: 'MCMC_results.json' (used to read assumed disk geometry and stellar flux) and 'save_figure.py' (used to read clean image RMS noise)
- 'clean' dir: Primary beam-corrected CLEAN image ('*.pbcor.fits') and CLEAN model image ('*.model.fits')
- 'frank' dir: Visibility datasets ('*.corrected.txt')
- 'rave' dir: Rave fit array files ('*.npy')

3) Add the disk to your source parameters (.json) file
-  set 'base: SMG_sub', 'clean: npix' and 'clean: pixel_scale' according to the '.fits' filenames (these will be used to determine the filenames of the appropriate images to load)
- set 'clean: image_rms' as the value of `rms` corresponding to `robust`=0.5 in `save_figure.py`
- set 'rave: pixel_scale' according to the Rave model filename
- set 'base: dist' and 'frank: SED_fstar' according to the github wiki ('ARKS sample')
- 'frank: custom_fstar' and 'frank: bestfit' will have to be determined by running frank fits

Running the pipeline for a single source
----------------------------------------
The main pipeline file is `radial_pipeline.py`. It can be run from the terminal for fits/analysis of a single source with `python radial_pipeline.py -d '<disk name>'`, e.g., `python radial_pipeline.py -d 'HD76582'`. 

By default the pipeline runs using the parameter files `./default_gen_pars.json` (contains parameters to choose which of the above pipeline modules run, as well as sensible choices for the pipeline parameters applicable to all sources) and `./default_source_pars.json` (contains sensible choices for source-specific, best-fit parameters). 

For descriptions of the `.json` parameters, see `gen_pars_description.json` and `source_pars_description.json`.

### Setting up frank fits ###
- For running frank, you will likely want to adjust the `alpha`, `wsmooth` and `scale_heights` parameters in `./default_gen_pars.json`. 

- When performing frank fits to find a radial profile, I recommend setting `method` to `"LogNormal"` to perform fits in logarithmic brightness space. Not all parts of the pipeline support linear brightness space fits with enforced non-negativity; this is because the logarithmic fits are in general a better choice. The exception is that when running a frank 1+1D fit to find _h_, `method` must be `"Normal"` (it will be enforced), because `"LogNormal"` is not supported within frank.

Running the pipeline for multiple/all sources
---------------------------------------------
The pipeline can be looped over multiple sources using `bulk_pipeline_run.py` via `python bulk_pipeline_run.py` (you may want to adjust the referenced `.json` parameter files there). 

Obtaining key results for multiple/all sources
----------------------------------------------
Survey-wide results are a `.txt` file per source with all radial brightness profiles (clean, rave, frank) sampled at the same radii, and a figures with a panel for each source showing the clean, rave, frank brightness profiles (one figure without uncertainties, one figure with). These are generated with `bulk_pipeline_results.py` via `python bulk_pipeline_results.py` (you may want to adjust the referenced `.json` parameter files there).