"""
Gaussian process dataset.

"""

import random as rnd

import jax.numpy as jnp
from jax import random
from numpyro.infer import Predictive

from priorCVAE.priors import GP, Kernel


class GPDataset:
    """
    Generate GP draws over the regular grid in the interval (x_lim_low, x_lim_high) with n_dataPoints points.

    Note: Currently the data is only generated with dimension as 1.

    """

    def __init__(self, kernel: Kernel, n_data: int = 400, x_lim_low: int = 0,
                 x_lim_high: int = 1, sample_lengthscale: bool = False):
        """
        Initialize the Gaussian Process dataset class.

        :param kernel: Kernel to be used.
        :param n_data: number of data points in the interval.
        :param x_lim_low: lower limit of the interval.
        :param x_lim_high: upper limit if the interval.
        :param sample_lengthscale: whether to sample lengthscale for the kernel or not. Defaults to False.
        """
        self.n_data = n_data
        self.x_lim_low = x_lim_low
        self.x_lim_high = x_lim_high
        self.sample_lengthscale = sample_lengthscale
        self.kernel = kernel
        self.x = jnp.linspace(self.x_lim_low, self.x_lim_high, self.n_data)

    def simulatedata(self, n_samples: int = 10000) -> [jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        
        """
        Simulate data from the GP and Robust Normalize it.
        """
        rng_key, _ = random.split(random.PRNGKey(rnd.randint(0, 9999)))

        gp_predictive = Predictive(GP, num_samples=n_samples)
        all_draws = gp_predictive(rng_key, x=self.x, kernel=self.kernel, jitter=1e-5,
                                  sample_lengthscale=self.sample_lengthscale)

        ls_draws = jnp.array(all_draws['ls'])
        gp_draws = jnp.array(all_draws['y'])

        # =====================================================ee
        # 🟢 THE FIX: Robust Normalization (Median / MAD)
        # =====================================================
        # We must force the simulations to match the scale of the 
        # real test data (which will be approx -3 to +3).
        
        # 1. Calculate Median per simulation
        # (axis=1 means we calculate stats for each curve individually)
        med = jnp.median(gp_draws, axis=1, keepdims=True)
        
        # 2. Calculate MAD (Median Absolute Deviation)
        # 1.4826 is the scaling factor to make MAD consistent with Sigma
        abs_dev = jnp.abs(gp_draws - med)
        mad = 1.4826 * jnp.median(abs_dev, axis=1, keepdims=True) + 1e-6
        
        # 3. Normalize
        gp_draws = (gp_draws - med) / mad
        # =====================================================
        
        # ✅ NORMALIZE X-AXIS
        raw_x = self.x
        x_norm = (raw_x - jnp.min(raw_x)) / (jnp.max(raw_x) - jnp.min(raw_x))
        x_final = x_norm.repeat(n_samples).reshape(self.x.shape[0], n_samples).transpose()

        return x_final, gp_draws, ls_draws