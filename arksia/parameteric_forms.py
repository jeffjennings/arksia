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


def asym_gauss(params: optax.Params, r: jnp.ndarray):
    """
    Assymetric (piecewise) Gaussian function f(r) of the form:

        .. math::

            \Sigma(r) = \exp\left[ \dfrac{-(r-R_{c})^{2}}{2\sigma_{\rm{in}}^{2}} \right] \mathrm{for} r < R_{c}, 
            \Sigma(r) = \exp\left[ \dfrac{-(r-R_{c})^{2}}{2\sigma_{\rm{out}}^{2}} \right] \mathrm{for} r \geq R_{c}
    """      

    return jnp.piecewise(r, [r < params['Rc'], r >= params['Rc']], 
                        [lambda r : gauss(r, params['Rc'], params['sigma_in']), 
                         lambda r : gauss(r, params['Rc'], params['sigma_out'])]
                         )

