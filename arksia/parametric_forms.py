"""This module contains functions defining parametric forms for radial brightness profiles 
(written by Jeff Jennings, parametric forms supplied by Brianna Zawadzki)."""

import jax.numpy as jnp
from jax.scipy.special import erf
import optax

def gauss(r: jnp.ndarray, a: jnp.float32, Rc: jnp.float32, sigma: jnp.float32):
    """
    Gaussian function \Sigma(r) of the form: 

        ..math::

            \Sigma(r) = a * \exp(-(r - R_c)^2 / (2 * \sigma^2)
    """        

    return a * jnp.exp(-(r - Rc) ** 2 / (2 * sigma ** 2))


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


def double_powerlaw(params: optax.Params, r: jnp.ndarray):
    """
    Double power law function f(r) of the form:

        .. math::

            \Sigma(r) = \left( \left( \dfrac{r}{R_{c}} \right)^{-\alpha_{\rm{in}}\gamma} + 
            \left( \dfrac{r}{R_{c}} \right)^{-\alpha_{\rm{out}}\gamma} \right)^{-1/\gamma}
    """      

    value = ((r / params['Rc']) ** (-params['alpha_in'] * params['gamma']) +
             (r / params['Rc']) ** (-params['alpha_out'] *
                                    params['gamma'])) ** (-1 / params['gamma'])

    if params['Rin'] is not None:
        value *= (1 + erf((r - params['Rin']) / params['l_in']))
    if params['Rout'] is not None:
        value *= (1 + erf((params['Rout'] - r) / params['l_out']))

    return value


def single_erf_powerlaw(params: optax.Params, r: jnp.ndarray):
    """
    Single error function and power law function f(r) of the form:

        ..math::

            \Sigma(r) = \left( 1 - \rm{erf}\left(\dfrac{R_{c}-r}{\sqrt{2} \:\sigma_{\rm{in}} R_{c}} \right)\right) 
            \left( \dfrac{r}{R_{c}} \right)^{-\alpha_{\rm{out}}}
    """      

    inner_edge = 1 - erf((params['Rc'] - r) / 
                         (2 ** 0.5 * params['sigma_in'] * params['Rc']))

    return inner_edge * (r / params['Rc']) ** -params['alpha_out']


def double_erf_powerlaw(params: optax.Params, r: jnp.ndarray):
    """
    Double error function and power law function f(r) of the form:

        ..math::

            \Sigma(r) = \left( 1 - \rm{erf}\left(\dfrac{R_{\rm{in}}-r}{\sqrt{2} \:\sigma_{\rm{in}} R_{\rm{in}}} \right) \right)
                         \left( 1 - \rm{erf}\left(\dfrac{r-R_{\rm{out}}}{\sqrt{2} \:\sigma_{\rm{out}} R_{\rm{out}}} \right) \right)
                         \left( \dfrac{r}{R_{\rm{in}}} \right)^{-\alpha
    """      

    inner_edge = 1 - erf((params['Rin'] - r) / 
                         (2 ** 0.5 * params['sigma_in'] * params['Rin']))
    outer_edge = 1 - erf((r - params['Rout']) / 
                         (2 ** 0.5 * params['sigma_out'] * params['Rout']))

    return inner_edge * outer_edge * (r / params['Rin']) ** (-params['alpha'])