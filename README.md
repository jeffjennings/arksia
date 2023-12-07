_arksia_ ('ARKS Image Analysis') - a pipeline for 1D image analysis of the ALMA large program ARKS ('ALMA survey to Resolve exoKuiper belt Substructures').

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
- runs parametric fits to a nonparametric radial brightness profile using a range of physically motivated parametric forms (uses JAX for hardware acceleration)
- produces plots to compare clean, rave, frank brightness profiles, radial visibility profiles, images, residuals
- produces plots to compare frank fits with different hyperparameter values
- produces plots to compare nonparametric and parametric brightness profiles
- produces plots to assess frank vertical inference over grids of _h_, _alpha_, _wsmooth_

The pipeline has utilites to prepare visibility files and save/load/interface with the above. It runs from general and source-specific parameter files. It can be run in bulk (across multiple sources).

Staging input files for the pipeline
------------------------------------
1) Ensure you have the following parameter files (you may name them differently and pass them in when running the pipeline):
    * 'pars_gen.json' (contains parameters to choose which of the pipeline modules run, as well as sensible choices for the pipeline parameters applicable to all sources)
    * 'pars_source.json' (contains sensible choices for source-specific best-fit parameters, as well as metrics of supplied clean images)
    * 'summary_disc_parameters.csv' (used to read disk geometry and stellar flux)
      
2) In a single directory, depending on which modular components of the pipeline you will run, store the following input files (and set this directory as the `input_dir` in `pars_gen.json`):
    * if using the pipeline to extract a radial profile from a clean image and/or to compare clean results to frank and rave:
        - primary beam-corrected CLEAN image ('<>.pbcor.fits'), primary beam image ('<>.pb.fits'), CLEAN model image ('<>.model.fits') for each robust value 
    * if using the pipeline to run frank fits:
        - visibility datasets from the ARKS survey ('<>corrected.txt')
    * if using the pipeline to compare rave results to clean and frank:
        - rave fit array files ('<>.npy') for each robust value

Running the pipeline for a single source
----------------------------------------
The main pipeline file is `pipeline.py`. By default the pipeline runs using the parameter files `./pars_gen.json` and `./pars_source.json`, and the table `./summary_disc_parameters.csv`. For a description of the parameters, see `description_pars_gen.json` and `description_pars_source.json`, or from a terminal run `python -c 'import arksia.pipeline; arksia.pipeline.helper()'`.

The pipeline can be run from the terminal for fits/analysis of a single source with `python -m arksia.pipeline -d '<disk name>'`, where the disk name, e.g. `'HD76582'`, must match a source name in the source-specific .json parameter file (by default `./pars_source.json`).

### Setting up frank fits ###
- To run frank, you will likely want to adjust the `alpha`, `wsmooth` and `scale_heights` parameters in `pars_gen.json`. 

- When performing frank fits to find a radial profile, set `method` to `"LogNormal"` to perform fits in logarithmic brightness space. This is enforced in some parts of the pipeline because the logarithmic fits are in general a better choice. The exception is that when running a frank 1+1D fit to find _h_, `method` must be `"Normal"` (it will be enforced).

### Setting up parametric fits ###
- To install the needed dependencies for parametric fits, `pip install arksia[analysis]`
    * If when running a parametric fit you receive the warning `WARNING:jax._src.xla_bridge:CUDA backend failed to initialize`, update CUDA with `pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
  
- To run a parametric fit to a nonparametric `frank` profile, set the `form` parameter in `pars_gen.json` as a list of parametric functional forms among those in `description_pars_gen.json`. Functional forms in `form` will be fit sequentially.
    * If during a fit the loss function (shown to the left of the progress bar) is still varying at the end of the optimization, increase `niter` or change `learn_rate` in `pars_gen.json`.

Running the pipeline for multiple/all sources
---------------------------------------------
The pipeline can be looped over multiple sources via `python bulk_pipeline_run.py` (you may want to adjust the referenced `.json` parameter files in `bulk_pipeline_run.py`). 

Obtaining key results for multiple/all sources
----------------------------------------------
Survey-wide summary results are a `.txt` file per source with all radial brightness profiles (clean, rave, frank) sampled at the same radii, and figures with a panel for each source showing the clean, rave, frank brightness profiles (one figure without uncertainties, one with). These are generated via `python bulk_pipeline_results.py` (you may want to adjust the referenced `.json` parameter files in `bulk_pipeline_results.py`).
