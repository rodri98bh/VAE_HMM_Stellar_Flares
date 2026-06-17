"""
File containing various loss classes that can be directed passed to the trainer object.
"""

from functools import partial
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax.random import KeyArray
from flax.training.train_state import TrainState
from flax.core import FrozenDict
import flax.linen as nn

from priorCVAE.losses import kl_divergence, scaled_sum_squared_loss, square_maximum_mean_discrepancy
from priorCVAE.priors import Kernel


class Loss(ABC):
    """
    Parent class for all the loss classes. This is to enforce the structure of the __call__ function which is
    used by the trainer object.
    """
    def __init__(self, conditional: bool = False):
        self.conditional = conditional

    @abstractmethod
    def __call__(self, state_params: FrozenDict, state: TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 z_rng: KeyArray):
        pass


class SquaredSumAndKL(Loss):
    """
    Loss function with scaled sum squared loss and weighted KL divergence.
    """

    def __init__(self, conditional: bool = False, vae_var: float = 1.0, kl_weight: float = 1e-4):
        super().__init__(conditional)
        self.vae_var = vae_var
        self.kl_weight = kl_weight

    @partial(jax.jit, static_argnames=['self'])
    def __call__(self, state_params: FrozenDict, state: TrainState,
                 batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 z_rng: KeyArray) -> jnp.ndarray:
        _, y, ls = batch
        c = ls if self.conditional else None
        y_hat, z_mu, z_logvar = state.apply_fn({'params': state_params}, y, z_rng, c=c)

        # Safe range checks (no f-strings inside JAX tracing)
        y_max, y_min = jnp.max(y), jnp.min(y)
        yhat_max, yhat_min = jnp.max(y_hat), jnp.min(y_hat)
        mu_max, mu_min = jnp.max(z_mu), jnp.min(z_mu)
        logvar_max, logvar_min = jnp.max(z_logvar), jnp.min(z_logvar)

        # rcl + kl losses
        rcl_loss = scaled_sum_squared_loss(y, y_hat, vae_var=self.vae_var)
        kld_loss = kl_divergence(z_mu, z_logvar)

        total_loss = rcl_loss + self.kl_weight * kld_loss

        # 💡 Use host_callback to print real numbers (not tracers)
        from jax.experimental import host_callback as hcb
        def debug_print(vals, _):
            y_min, y_max, yhat_min, yhat_max, mu_min, mu_max, lv_min, lv_max, rcl, kld, total = vals
            print(f"y range: {y_min:.3e} to {y_max:.3e}")
            print(f"y_hat range: {yhat_min:.3e} to {yhat_max:.3e}")
            print(f"z_mu range: {mu_min:.3e} to {mu_max:.3e}")
            print(f"z_logvar range: {lv_min:.3e} to {lv_max:.3e}")
            print(f"🔍 rcl_loss = {rcl:.3e}, kld_loss = {kld:.3e}, total = {total:.3e}")

        hcb.id_tap(debug_print, (y_min, y_max, yhat_min, yhat_max,
                                 mu_min, mu_max, logvar_min, logvar_max,
                                 rcl_loss, kld_loss, total_loss))

        return total_loss


class MMDAndKL(Loss):
    """
    Loss function with RELU-MMD loss and KL.
    """

    def __init__(self, kernel: Kernel, conditional: bool = False, kl_scaling: float = 1e-6):
        """
        Initialize the SquareMMDAndKL loss.

        :param kernel: Kernel to use for calculaing MMD.
        :param conditional: a variable to specify if conditional version is getting trained or not.
        :param kl_scaling: a float value representing the scaling value for the KL term.
        """
        super().__init__(conditional)
        self.kernel = kernel
        self.kl_scaling = kl_scaling

    @partial(jax.jit, static_argnames=['self'])
    def __call__(self, state_params: FrozenDict, state: TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                 z_rng: KeyArray) -> jnp.ndarray:
        """
        Calculates the loss value.

        :param state_params: Current state parameters of the model.
        :param state: Current state of the model.
        :param batch: Current batch of the data. It is list of [x, y, c] values.
        :param z_rng: a PRNG key used as the random key.
        """
        _, y, ls = batch
        c = ls if self.conditional else None
        y_hat, z_mu, z_logvar = state.apply_fn({'params': state_params}, y, z_rng, c=c)
        sq_mmd_loss = square_maximum_mean_discrepancy(self.kernel, y, y_hat, efficient_grads=True)
        relu_sq_mmd_loss = nn.relu(sq_mmd_loss)
        kld_loss = kl_divergence(z_mu, z_logvar)
        loss = jnp.sqrt(relu_sq_mmd_loss) + self.kl_scaling * kld_loss
        return loss
