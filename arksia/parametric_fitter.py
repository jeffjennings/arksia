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

        self._initial_params = {"form": model["parametric"]["form"] }

        # check if jax is on a gpu or tpu
        self._device = jax.default_backend()
        print(f"    JAX is using the {self._device}.")
        if self._device == 'cpu':
            print("      Using the CPU will slow computation of parametric fits.")

    def parametric_model(self, params: optax.Params, x: jnp.ndarray):
        """
        # TODO
        """
        
        form = params['form']

        if form == 'asym_gauss':
            return asym_gauss(params, x) 
        
        elif form == 'double_powerlaw':
            return double_powerlaw(params, x)
        
        elif form == 'single_erf_powerlaw':
            return single_erf_powerlaw(params, x)
        
        elif form == 'double_erf_powerlaw':
            return double_erf_powerlaw(params, x)
        
        else:
            raise ValueError(f"{form} invalid")
        

    def loss(self, params: optax.Params, x: jnp.ndarray, y: jnp.ndarray, 
             err: jnp.ndarray) -> jnp.ndarray:
        """
        # TODO
        """             
        
        y_hat = self.parametric_model(params, x)

        # return (jnp.mean((y - y_hat) ** 2)) ** 0.5 # RMSE loss
        return ((y - y_hat) ** 2 / err ** 2).sum()

