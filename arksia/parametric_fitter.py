"""This module contains a class to fit parametric forms to a radial brightness signal 
(written by Jeff Jennings)."""

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import jax.numpy as jnp
import jax
import optax 

from arksia.parametric_forms import *

class ParametricFit():
    """
    # TODO
    """

    def __init__(self, truth, model, learn_rate=1e-3, niter=10000):

        # get frank brightness profile that we'll fit a parametric form to.
        # 'x' is radial points of frank fit, 'y' is brightness, 'err' is lower
        # bound on brightness uncertainty.
        x, y, err = truth
        self._x, self._y, self._err = jnp.array(x), jnp.array(y), jnp.array(err)

        self._model = model 

        self._learn_rate = learn_rate
        self._niter = niter

