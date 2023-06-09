Pipeline for radial profile (team 2) extraction and analysis. The pipeline is run from the terminal using input parameter 
files. It currently has the following, modular capabilities:
- [x] extracts a radial brightness profile from a clean image
- [x] processes an existing rave fit to obtain a brightness profile in consistent units
- [x] runs frank to obtain a brightness profile
- [x] runs frank for 1+1D vertical (disk aspect ratio) inference
- [x] produces plots to compare clean, rave, frank brightness profiles, radial visibility profiles, images
- [x] produces plots to assess frank vertical inference over grids of _h_, _alpha_, _wsmooth_
- [x] adds utilites to prepare visibility files for the above and to save/load/interface with all of the above. This includes the option to produce a single .txt file with the clean, rave and frank brightness profiles (sampled at the same radii) for a given source, to share with other ARKS teams.
- [x] pipeline runs from general and source-specific parameter files to do any combination of the above

The main pipeline file is `radial_pipeline.py`. It can be run from the terminal as `python radial_pipeline.py -d '<disk name>'`, e.g., `python radial_pipeline.py -d 'HD76582'`. By default the pipeline runs using the parameter files `default_gen_pars.json` (which has parameters for which pipeline modules to run, as well as sensible choices for the pipeline parameters applicable to all sources - though for running frank, you will likely want to adjust the `alpha`, `wsmooth` and `scale_heights` parameters) and `default_source_pars.json` (which has sensible choices for source-specific, best-fit parameters). For descriptions of the `.json` parameters, see `gen_pars_description.json` and `source_pars_description.json`.

The pipeline can be run for a single source using the above, or looped over multiple sources using `bulk_pipeline_run.py`.

When performing frank fits to find a radial profile, I strongly recommend setting `method` to `"LogNormal"` to perform fits in logarithmic brightness space. Not all parts of the pipeline support non-negative fits for producing/interfacing with results and performing analysis; this is because the logarithmic fits are in general a better choice. When running a frank 1+1D fit to find _h_, `method` must be `"Normal"` (it will be enforced), because `"LogNormal"` is not supported within frank.