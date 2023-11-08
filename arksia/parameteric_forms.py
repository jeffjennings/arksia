"""This module contains functions defining parametric forms for radial brightness profiles 
(written by Brianna Zawadzki, adapted for JAX by Jeff Jennings)."""

import jax.numpy as jnp
from jax.scipy.special import erf
import optax

def gauss(r: jnp.ndarray, Rc: jnp.float32, sigma: jnp.float32):
    """
    Gaussian function f(r) of the form: 

        ..math::

            \Sigma(r) = \exp(-(r - Rc)^2 / (2 * \sigma^2)
    """        

    return jnp.exp(-(r - Rc) ** 2 / (2 * sigma ** 2))

