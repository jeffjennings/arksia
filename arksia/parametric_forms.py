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
    Assymetric (piecewise) Gaussian function \Sigma(r) of the form:

        .. math::

            \Sigma(r) = a_1 * \exp(-(r - Rc)^2 / (2 * \sigma_1^2) \mathrm{for} r < R_c, 
              \Sigma(r) = a_2 * \exp(-(r - Rc)^2 / (2 * \sigma_2^2) \mathrm{for} r \geq R_c
    """      

    return jnp.piecewise(r, [r < params['Rc'], r >= params['Rc']], 
                        [lambda r : gauss(r, params['a1'], params['Rc'], params['sigma1']),
                         lambda r : gauss(r, params['a2'], params['Rc'], params['sigma2'])]
                         )


def triple_gauss(params: optax.Params, r: jnp.ndarray):
    """
    Sum of three Gaussian functions \Sigma(r)
    """      

    gauss1 = gauss(r, params['a1'], params['Rc1'], params['sigma1'])
    gauss2 = gauss(r, params['a2'], params['Rc2'], params['sigma2'])
    gauss3 = gauss(r, params['a3'], params['Rc3'], params['sigma3'])

    return gauss1 + gauss2 + gauss3


def double_powerlaw(r: jnp.ndarray, Rc: jnp.float32, alpha1: jnp.float32, 
                    alpha2: jnp.float32, gamma: jnp.float32):
    """
    Double power law function \Sigma(r) of the form:

        .. math::

            \Sigma(r) = [(r / R_c)^{\alpha_1 * \gamma} + (r / R_c)^{\alpha_2 * \gamma}]^{-1 / \gamma}
    """

    return ((r / Rc) ** (-alpha1 * gamma) + (r / Rc) ** (alpha2 * gamma)) ** (-1 / gamma)

    
def double_powerlaw_limits(params: optax.Params, r: jnp.ndarray):
    """
    Double power law function \Sigma(r), optionally with limits R1 and R2 that
    multiply the double power law by error functions of the form:

        .. math::

            \Sigma(r) = [(r / R_c)^{\alpha_1 * \gamma} + (r / R_c)^{\alpha_2 * \gamma}]^{-1 / \gamma} * 
              [1 + \erf{(r - R_1) / l_1}] * 
              [1 + \erf{(R_2 - r) / l_2}]
    """      

    dpl = double_powerlaw(r, params['Rc'], params['alpha1'], params['alpha2'], params['gamma'])

    if params['R1'] is not None:
        dpl *= (1 + erf((r - params['R1']) / params['l1']))
    if params['R2'] is not None:
        dpl *= (1 + erf((params['R2'] - r) / params['l2']))

    return dpl


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