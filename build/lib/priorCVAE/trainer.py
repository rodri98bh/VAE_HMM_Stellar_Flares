"""
Trainer class for training Prior{C}VAE models.
"""
import time
from typing import List
from functools import partial
import random

from optax import GradientTransformation
import jax
import jax.numpy as jnp
from jax.random import KeyArray
from flax.training import train_state
from jax.tree_util import tree_map, tree_flatten

from priorCVAE.models import VAE
from priorCVAE.losses import SquaredSumAndKL, Loss


class VAETrainer:
    """
    VAE trainer class.
    """

    def __init__(self, model: VAE, optimizer: GradientTransformation, loss: Loss = SquaredSumAndKL()):
        """
        Initialize the VAETrainer object.

        :param model: model object of the class `priorCVAE.models.VAE`.
        :param optimizer: optimizer to be used to train the model.
        :param loss: loss function object of the `priorCVAE.losses.Loss`
        """
        self.model = model
        self.optimizer = optimizer
        self.state = None
        self.loss_fn = loss

    def init_params(self, y: jnp.ndarray, c: jnp.ndarray = None, key: KeyArray = None):
        """
        Initialize the parameters of the model.

        :param y: sample input of the model.
        :param c: conditional variable, while using vanilla VAE model this should be None.
        :param key: Jax PRNGKey to ensure reproducibility. If none, it is set randomly.
        """
        if key is None:
            key = jax.random.PRNGKey(random.randint(0, 9999))
        key, rng = jax.random.split(key, 2)

        params = self.model.init(rng, y, key, c)['params']
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.optimizer)
        
    @staticmethod
    def clip_grads(grads, max_norm=1.0):
        norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_flatten(grads)[0]]))
        return tree_map(lambda g: g * (max_norm / (norm + 1e-6)) if norm > max_norm else g, grads)

    #@partial(jax.jit, static_argnames=['self'])
    def train_step(self, state: train_state.TrainState, batch, z_rng):
        val, grads = jax.value_and_grad(self.loss_fn)(state.params, state, batch, z_rng)
    
        # Debug info
        grad_max = jnp.max(jnp.array([jnp.max(jnp.abs(g)) for g in jax.tree_util.tree_flatten(grads)[0]]))
        if jnp.isnan(val) or jnp.isinf(val) or grad_max > 1e6:
            raise ValueError("🚨 Loss or gradients exploded")
    
        # Clip gradients
        grads = self.clip_grads(grads, max_norm=1.0)
    
        return state.apply_gradients(grads=grads), val


    #@partial(jax.jit, static_argnames=['self'])
    def eval_step(self, state: train_state.TrainState, batch: [jnp.ndarray, jnp.ndarray, jnp.ndarray],
                  z_rng: KeyArray) -> jnp.ndarray:
        """
        Evaluates the model on the batch.

        :param state: Current state of the model.
        :param batch: Current batch of the data. It is list of [x, y, c] values.
        :param z_rng: a PRNG key used as the random key.

        :returns: The loss value.
        """
        return self.loss_fn(state.params, state, batch, z_rng)

    def train(self, data_generator, test_set: [jnp.ndarray, jnp.ndarray, jnp.ndarray], num_iterations: int = 10,
              batch_size: int = 100, debug: bool = True, key: KeyArray = None) -> [List, List, float]:
    
        if self.state is None:
            raise Exception("Initialize the model parameters before training!!!")
    
        loss_train = []
        loss_test = []
        t_start = time.time()
    
        if key is None:
            key = jax.random.PRNGKey(random.randint(0, 9999))
        z_key, test_key = jax.random.split(key, 2)
    
        for iterations in range(num_iterations):
            try:
                # --- Data generation ---
                batch_train = data_generator.simulatedata(batch_size)

                if debug:
                    f = batch_train[1]  # ✅ GP draws
                    f_mean = jnp.mean(f)
                    f_std = jnp.std(f)
                    f_min = jnp.min(f)
                    f_max = jnp.max(f)
                
                    print(f"[{iterations}] f batch stats: mean={f_mean:.3e}, std={f_std:.3e}, min={f_min:.3e}, max={f_max:.3e}")
                
                    if not jnp.all(jnp.isfinite(f)):
                        raise ValueError(f"Non-finite values in `f`: {f}")
    
                # --- Train step ---
                z_key, key = jax.random.split(z_key)
                self.state, loss_train_value = self.train_step(self.state, batch_train, key)
    
                if not jnp.isfinite(loss_train_value):
                    raise ValueError(f"[{iterations}] Training loss is not finite: {loss_train_value}")
    
                loss_train.append(float(loss_train_value))
    
                # --- Eval step ---
                test_key, key = jax.random.split(test_key)
                loss_test_value = self.eval_step(self.state, test_set, test_key)
    
                if not jnp.isfinite(loss_test_value):
                    raise ValueError(f"[{iterations}] Test loss is not finite: {loss_test_value}")
    
                loss_test.append(float(loss_test_value))
    
                # --- Logging ---
                if debug and iterations % 10 == 0:
                    print(f'[{iterations + 1:5d}] training loss: {loss_train[-1]:.3f}, test loss: {loss_test[-1]:.3f}')
    
            except Exception as e:
                print(f"\n🚨 Error at iteration {iterations}: {e}")
                break  # or `continue` if you want to skip this batch and proceed
    
        t_elapsed = time.time() - t_start
        return loss_train, loss_test, t_elapsed
