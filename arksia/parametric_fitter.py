"""This module contains a class to fit parametric forms to a radial brightness signal 
(written by Jeff Jennings)."""

from tqdm.auto import tqdm
import jax.numpy as jnp
import jax
import optax

from arksia import parametric_forms

class ParametricFit():
    """
    # TODO
    """

    def __init__(self, truth, model, func_form, learn_rate=1e-3, niter=100000):
        # set jax device. Must come before any jax calls.
        if model["parametric"]["device"] is not None: # TODO
            self._device = model["parametric"]["device"]
            jax.config.update('jax_platform_name', self._device)
        else:
            self._device = jax.default_backend()
        print(f"    JAX is using the {self._device}.")

        # get frank brightness profile that we'll fit a parametric form to.
        # 'x' is radial points of frank fit, 'y' is brightness, 'err' is lower
        # bound on brightness uncertainty.
        x, y, err = truth
        self._x, self._y, self._err = jnp.array(x), jnp.array(y), jnp.array(err)

        self._model = model 

        self._learn_rate = learn_rate
        self._niter = niter

        self._form = func_form

        self._initial_params = {}


    def parametric_model(self, params: optax.Params, x: jnp.ndarray):
        """
        # TODO
        """
        
        form = self._form

        if form == 'asym_gauss':
            return parametric_forms.asym_gauss(params, x) 
        if form == 'triple_gauss':
            return parametric_forms.triple_gauss(params, x) 
        
        elif form == 'double_powerlaw':
            return parametric_forms.double_powerlaw_erf(params, x)
        elif form == 'double_powerlaw_erf':
            return parametric_forms.double_powerlaw_erf(params, x)
        elif form == 'double_powerlaw_gauss':
            return parametric_forms.double_powerlaw_gauss(params, x)
        elif form == 'double_powerlaw_double_gauss':
            return parametric_forms.double_powerlaw_double_gauss(params, x)
                
        elif form == 'single_erf_powerlaw':
            return parametric_forms.single_erf_powerlaw(params, x)
        elif form == 'double_erf_powerlaw':
            return parametric_forms.double_erf_powerlaw(params, x)
        
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

        @jax.jit
        def step(params, opt_state, x, y, err):
            loss_value, grads = jax.value_and_grad(self.loss)(params, x, y, err)

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            return params, opt_state, loss_value

        loss_arr = jnp.zeros(niter)
        loss_initial = self.loss(params, self._x, self._y, self._err)
        loss_arr = loss_arr.at[0].set(loss_initial)

        progress = tqdm(range(1, niter))

        for i in progress:
            params, opt_state, loss_current = step(params, opt_state, 
                                                  self._x, self._y, self._err)
            loss_arr = loss_arr.at[i].set(loss_current)

            if i % 100 == 0:
                progress.set_description(f"    loss {loss_arr[i]:.2f}")

        self._loss_history = loss_arr
        self._params = params

        return


    def fit(self):
        """
        # TODO
        """

        form = self._form
        print(f"    fitting parametric form {form} to input profile")

        # set initial guesses for parameter values by using the signal we're fitting:
        # centroid or critical radius
        peak_idx = jnp.argmax(self._y)
        Rc = self._x[peak_idx]

        # amplitude
        amax = max(self._y)

        # standard deviation (obtain by guessing FWHM)
        idx = next(i for i,j in enumerate(self._y) if
                   i > peak_idx and j < amax / 2)
        fwhm = (self._x[idx] - Rc) * 2
        sigma = fwhm / (8 * jnp.log(2)) ** 0.5

        if form == 'asym_gauss':
            self._initial_params["Rc"] = Rc
            self._initial_params["a"] = amax
            self._initial_params["sigma1"] = sigma
            self._initial_params["sigma2"] = sigma

        elif form == 'triple_gauss':
            self._initial_params["R1"] = 0.7 * Rc
            self._initial_params["R2"] = Rc
            self._initial_params["R3"] = 1.3 * Rc
            self._initial_params["a1"] = 0.3 * amax
            self._initial_params["a2"] = amax
            self._initial_params["a3"] = 0.3 * amax
            self._initial_params["sigma1"] = 0.5 * sigma
            self._initial_params["sigma2"] = sigma
            self._initial_params["sigma3"] = 0.5 * sigma

        elif form in ['double_powerlaw', 'double_powerlaw_erf']:
            self._initial_params["Rc"] = Rc
            self._initial_params["a"] = amax
            self._initial_params["gamma"] = 1.0
            self._initial_params["alpha1"] = 3.0
            self._initial_params["alpha2"] = -3.0

            if form == 'double_powerlaw':
                self._initial_params["R1"] = None
                self._initial_params["R2"] = None
                self._initial_params["l1"] = None
                self._initial_params["l2"] = None
            else:
                self._initial_params["R1"] = 0.5 * Rc
                self._initial_params["R2"] = 1.5 * Rc
                self._initial_params["l1"] = Rc
                self._initial_params["l2"] = Rc

        elif form == 'double_powerlaw_gauss':
            self._initial_params["a1"] = amax
            self._initial_params["a2"] = 0.5 * amax
            self._initial_params["R1"] = Rc
            self._initial_params["R2"] = Rc + 2.0
            self._initial_params["alpha1"] = 3.0
            self._initial_params["alpha2"] = -3.0          
            self._initial_params["sigma"] = sigma
            self._initial_params["gamma"] = 1.0

        elif form == 'double_powerlaw_double_gauss':
            self._initial_params["a1"] = -0.3 * amax
            self._initial_params["a2"] = -0.3 * amax
            self._initial_params["a3"] = amax
            self._initial_params["R1"] = 0.7 * Rc
            self._initial_params["R2"] = 1.3 * Rc
            self._initial_params["R3"] = Rc
            self._initial_params["alpha1"] = 3.0
            self._initial_params["alpha2"] = -3.0          
            self._initial_params["sigma1"] = 0.1 * sigma
            self._initial_params["sigma2"] = 0.1 * sigma
            self._initial_params["gamma"] = 1.0               

        elif form == 'single_erf_powerlaw':
            self._initial_params["a"] = amax
            self._initial_params["Rc"] = Rc
            self._initial_params["alpha"] = 2.0
            self._initial_params["sigma"] = 0.1 * sigma

        elif form == 'double_erf_powerlaw':
            self._initial_params["a"] = amax
            self._initial_params["R1"] = 0.5 * Rc
            self._initial_params["R2"] = 1.5 * Rc
            self._initial_params["sigma1"] = 0.1 * sigma
            self._initial_params["sigma2"] = 0.1 * sigma
            self._initial_params["alpha"] = 1.0

        else:
            raise ValueError(f"{form} invalid")
        
        optimizer = optax.adam(self._learn_rate)

        return self._fit(self._initial_params, optimizer, self._niter)


    @property
    def functional_form(self):
        """String of the name of the fit functional form"""
        return self._form
    
    @property
    def initial_params(self):
        """Dictionary of initial guess values for model parameters"""
        return self._initial_params
    
    @property    
    def bestfit_params(self):
        """Dictionary of best fit values for model parameters"""
        return self._params
    
    @property
    def loss_history(self):
        """Array of loss values over optimization loop"""
        return self._loss_history