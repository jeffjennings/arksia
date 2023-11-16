"""This module contains functions defining parametric forms for radial brightness profiles 
(written by Jeff Jennings, parametric forms supplied by Brianna Zawadzki)."""

import jax.numpy as jnp
from jax.scipy.special import erf
import optax

def gauss(r: jnp.ndarray, Rc: jnp.float32, a: jnp.float32, sigma: jnp.float32):
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

            \Sigma(r) = a * \exp(-(r - Rc)^2 / (2 * \sigma_1^2) \mathrm{for} r < R_c, 
              \Sigma(r) = a * \exp(-(r - Rc)^2 / (2 * \sigma_2^2) \mathrm{for} r \geq R_c
    """      

    return jnp.piecewise(r, [r < params['Rc'], r >= params['Rc']], 
                        [lambda r : gauss(r, params['Rc'], params['a'], params['sigma1']),
                         lambda r : gauss(r, params['Rc'], params['a'], params['sigma2'])]
                         )


def triple_gauss(params: optax.Params, r: jnp.ndarray):
    """
    Sum of three Gaussian functions \Sigma(r)
    """      

    gauss1 = gauss(r, params['R1'], params['a1'], params['sigma1'])
    gauss2 = gauss(r, params['R2'], params['a2'], params['sigma2'])
    gauss3 = gauss(r, params['R3'], params['a3'], params['sigma3'])

    return gauss1 + gauss2 + gauss3


def double_powerlaw(r: jnp.ndarray, Rc: jnp.float32, a: jnp.ndarray, alpha1: jnp.float32, 
                    alpha2: jnp.float32, gamma: jnp.float32):
    """
    Double power law function \Sigma(r) of the form:

        .. math::

            \Sigma(r) = a * [(r / R_c)^{-\alpha_1 * \gamma} + (r / R_c)^{-\alpha_2 * \gamma}]^{-1 / \gamma}
    """

    return a * ((r / Rc) ** (-alpha1 * gamma) + (r / Rc) ** (-alpha2 * gamma)) ** (-1 / gamma)

    
def double_powerlaw_erf(params: optax.Params, r: jnp.ndarray):
    """
    Double power law function \Sigma(r), optionally with limits R1 and R2 that
    multiply the double power law by error functions of the form:

        .. math::

            \Sigma(r) = a * [(r / R_c)^{\alpha_1 * \gamma} + (r / R_c)^{\alpha_2 * \gamma}]^{-1 / \gamma} * 
              [1 + \erf{(r - R_1) / l_1}] * 
              [1 + \erf{(R_2 - r) / l_2}]
    """      

    dpl = double_powerlaw(r, params['Rc'], params['a'], params['alpha1'], params['alpha2'], params['gamma'])

    if params['R1'] is not None:
        dpl *= (1 + erf((r - params['R1']) / params['l1']))
    if params['R2'] is not None:
        dpl *= (1 + erf((params['R2'] - r) / params['l2']))

    return dpl


def double_powerlaw_gauss(params: optax.Params, r: jnp.ndarray):
    """
    Double power law function with inner Gaussian, \Sigma(r) of the form:

        .. math::

            \Sigma(r) =  a_1 * \exp(-(r - R_1)^2 / (2 * \sigma^2) + 
              a_2 * [(r / R_2)^{\alpha_1 * \gamma} + (r / R_2)^{\alpha_2 * \gamma}]^{-1 / \gamma}
    """      

    gaussian = gauss(r, params['R1'], params['a1'], params['sigma'])

    dpl = double_powerlaw(r, params['R2'], params['a2'], params['alpha1'], params['alpha2'], params['gamma'])
    
    return gaussian * dpl


def double_powerlaw_double_gauss(params: optax.Params, r: jnp.ndarray):
    """
    Double power law function with two Gaussians, \Sigma(r) of the form:

        .. math::

            \Sigma(r) =  a_1 * \exp(-(r - R_1)^2 / (2 * \sigma_1^2) + 
              a_2 * \exp(-(r - R_2)^2 / (2 * \sigma_2^2) + 
              a_3 * [(r / R_3)^{\alpha_1 * \gamma} + (r / R_3)^{\alpha_2 * \gamma}]^{-1 / \gamma}
    """      

    gaussian1 = gauss(r, params['R1'], params['a1'], params['sigma1'])
    gaussian2 = gauss(r, params['R2'], params['a2'], params['sigma2'])

    dpl = double_powerlaw(r, params['R3'], params['a3'], params['alpha1'], params['alpha2'], params['gamma'])
    
    return dpl + gaussian1 + gaussian2


def single_erf_powerlaw(params: optax.Params, r: jnp.ndarray):
    """
    Single error function and power law function \Sigma(r) of the form:

        ..math::

            \Sigma(r) = a * [1 - \erf{(R_c - r) / (\sigma * R_c * \sqrt{2})}] * 
              (r / R_c)^{-\alpha} 
    """      

    inner_edge = 1 - erf((params['Rc'] - r) / \
                         (2 ** 0.5 * params['sigma'] * params['Rc']))

    return params['a'] * inner_edge * (r / params['Rc']) ** -params['alpha']


def double_erf_powerlaw(params: optax.Params, r: jnp.ndarray):
    """
    Double error function and power law function \Sigma(r) of the form:

        ..math::

            \Sigma(r) = [1 - \erf{(R_1 - r) / (\sigma_1 * R_1 * \sqrt{2})}] *
              [1 - \erf{(r - R_2) / (\sigma_2 * R_2 * \sqrt{2})}] *  
              (r / R_1)^{-\alpha}

    """      

    inner_edge = 1 - erf((params['R1'] - r) / \
                         (2 ** 0.5 * params['sigma1'] * params['R1']))
    outer_edge = 1 - erf((r - params['R2']) / \
                         (2 ** 0.5 * params['sigma2'] * params['R2']))

    return params['a'] * inner_edge * outer_edge * (r / params['R1']) ** (-params['alpha'])