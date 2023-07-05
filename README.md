_arksia_ ('ARKS Image Analysis') - a pipeline for 1D image analysis of the ALMA large program ARKS ('ALMA survey to Resolve exoKuiper belt Substructures').

Dependencies
------------
Use the newest versions of `frank` and `MPoL`:
- frank - `pip install git+https://github.com/discsim/frank.git`
- MPoL - `pip install git+https://github.com/MPoL-dev/MPoL.git`

Install
-------
```pip install arksia```

Pipeline scope
--------------
The pipeline is run from the terminal using input parameter files. It has the following, modular capabilities:
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
- Root directory: '[disk name]'
    - Subdirectories: 'clean', 'frank', 'rave'

2) Download and place the following files in these directories:
- root dir: 'MCMC_results.json' (used to read assumed disk geometry and stellar flux) and 'pars_image.json' (contains clean image RMS noise per robust value)
- 'clean' dir: Primary beam-corrected CLEAN image ('*.pbcor.fits'), primary beam image ('*.pb.fits'), CLEAN model image ('*.model.fits') for each robust value
- 'frank' dir: Visibility datasets ('*.corrected.txt')
- 'rave' dir: Rave fit array files ('*.npy') for each robust value

3) Add the disk to your source parameters (.json) file
-  set 'base: SMG_sub', 'clean: npix' and 'clean: pixel_scale' according to the '.fits' filenames (these will be used to determine the filenames of the appropriate images to load)
- set 'rave: pixel_scale' according to the Rave model filename
- set 'base: dist' and 'frank: SED_fstar' according to the github wiki (see 'ARKS sample' there)
- 'frank: custom_fstar' and 'frank: bestfit' will be determined by running frank fits

Running the pipeline for a single source
----------------------------------------
The main pipeline file is `pipeline.py`. It can be run from the terminal for fits/analysis of a single source with `python -m arksia.pipeline -d '<disk name>'`, where the disk name is, e.g., `'HD76582'`.

By default the pipeline runs using the parameter files `./pars_gen.json` (which contains parameters to choose which of the above pipeline modules run, as well as sensible choices for the pipeline parameters applicable to all sources) and `./pars_gen.json` (which contains sensible choices for source-specific, best-fit parameters). For a description of the parameters, see `description_pars_gen.json` and `description_pars_source.json`.

### Setting up frank fits ###
- To run frank, you will likely want to adjust the `alpha`, `wsmooth` and `scale_heights` parameters in `./pars_gen.json`. 

- When performing frank fits to find a radial profile, I recommend setting `method` to `"LogNormal"` to perform fits in logarithmic brightness space. Not all parts of the pipeline support linear brightness space fits with enforced non-negativity; this is because the logarithmic fits are in general a better choice. The exception is that when running a frank 1+1D fit to find _h_, `method` must be `"Normal"` (it will be enforced).

Running the pipeline for multiple/all sources
---------------------------------------------
The pipeline can be looped over multiple sources using `bulk_pipeline_run.py` via `python bulk_pipeline_run.py` (you may want to adjust the referenced `.json` parameter files there). 

Obtaining key results for multiple/all sources
----------------------------------------------
Survey-wide results are a `.txt` file per source with all radial brightness profiles (clean, rave, frank) sampled at the same radii, and figures with a panel for each source showing the clean, rave, frank brightness profiles (one figure without uncertainties, one figure with). These are generated with `bulk_pipeline_results.py` via `python bulk_pipeline_results.py` (you may want to adjust the referenced `.json` parameter files there).
