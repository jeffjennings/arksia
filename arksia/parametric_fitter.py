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


        self._model = model 

        self._learn_rate = learn_rate
        self._niter = niter

