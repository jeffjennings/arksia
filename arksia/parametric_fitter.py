"""This module contains a class to fit parametric forms to a radial brightness signal 
(written by Jeff Jennings)."""

import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import jax.numpy as jnp
import jax
import optax 
