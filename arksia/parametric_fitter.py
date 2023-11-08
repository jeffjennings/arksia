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



    def _fit(self, params: optax.Params, optimizer: optax.GradientTransformation, 
            niter: int) -> tuple[optax.Params, jnp.ndarray]:
        """
        # TODO
        """

        opt_state = optimizer.init(params)

        # TODO: set bounds for each param 

        @jax.jit
        def step(params, opt_state, x, y, err):
            loss_value, grads = jax.value_and_grad(loss)(params, x, y, err)

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            return params, opt_state, loss_value

        loss_arr = jnp.zeros(niter)
        loss_arr[0] = self.loss(params, self._x, self._y, self._err)

        progress = tqdm(range(1, niter))

        for i in progress:
            params, opt_state, loss_arr[i] = step(params, opt_state, 
                                                  self._x, self._y, self._err)
            if i % 100 == 0:
                progress.set_description(f"{loss_arr[i]:.2f}")

        return params, loss_arr


    def fit(self):
        """
        # TODO
        """

        form = self._initial_params['form']

        
        # fit the parametrized function using the Adam optimizer
        optimizer = optax.adam(self._learn_rate)

        return self._fit(self._initial_params, optimizer, self._niter)


