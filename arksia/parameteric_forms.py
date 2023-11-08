"""This module contains functions defining parametric forms for radial brightness profiles 
(written by Brianna Zawadzki, adapted for JAX by Jeff Jennings)."""

import jax.numpy as jnp
from jax.scipy.special import erf
import optax
