.. :history:

Changelog
+++++++++

v.0.1.1
+++++++
First public implementation of codebase. Codebase stable. Functionality to run 
frank fits, extract radial profiles from clean images, and compare these to 
rave fits.

v.0.2.0
+++++++
- Added functionality to run parametric fits (to an input brightness profile, 
not the visibilities). This includes: 
* Added parametric fitting class in `parametric_fitter.py`
* Added functional forms in `parametric_forms.py`
* Added plotting routine for parametric fits, `plot.parametric_fit_figure`
* Added parametric fit process to pipeline, `pipeline.fit_parametric`
* Added parametric fit parameters to the `.json` parameter files

v.0.2.1
+++++++
- Added tests covering much of code functionality
- Added test and coverage workflows, added badges to README
- Updated some input parameters and where they're used, especially for IO 
- Changed defaults on some input parameters, e.g. to use stellar flux from MCMC by default for frank fits
- Changed the way multiple frank fits are processed, prefering sequential fits with pre-processed visibilities over multiprocessing
- Added figures to compare multiple frank fits