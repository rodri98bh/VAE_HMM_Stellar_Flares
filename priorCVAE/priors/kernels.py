"""
File contains the code for Gaussian processes kernels.
"""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from priorCVAE.utility import sq_euclidean_dist


class Kernel(ABC):
    """
    Abstract class for the kernels.
    """
    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    @abstractmethod
    def __call__(self, x1, x2):
        pass

    def _handle_input_shape(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        The function checks if the input is in the shape (N, D). If (N, ) then a dimension is added in the end.
        Otherwise, Exception is raised.
        """
        if len(x.shape) == 1:
            x = x[..., None]
        if len(x.shape) > 2:
            raise Exception("Kernel only supports calculations with the input of shape (N, D).")
        return x

    def _scale_by_lengthscale(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Scale the input tensor by 1/lengthscale.
        """
        return x / self.lengthscale


class SquaredExponential(Kernel):
    """
    Squared exponential kernel.
    K(x1, x2) = var * exp(-0.5 * ||x1 - x2||^2/l**2)
    """
    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__(lengthscale, variance)

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the kernel value for x1 and x2.

        :param x1: Jax ndarray of the shape `(N1, D)`.
        :param x2: Jax ndarray of the shape `(N2, D)`.

        :return: kernel matrix of the shape `(N1, N2)`.

        """
        x1 = self._handle_input_shape(x1)
        x2 = self._handle_input_shape(x2)
        assert x1.shape[-1] == x2.shape[-1]
        x1 = self._scale_by_lengthscale(x1)
        x2 = self._scale_by_lengthscale(x2)
        dist = sq_euclidean_dist(x1, x2)
        k = self.variance * jnp.exp(-0.5 * dist)
        assert k.shape == (x1.shape[0], x2.shape[0])
        return k


class Matern32(Kernel):
    """
    Matern 3/2 Kernel.

    K(x1, x2) = variance * (1 + √3 * ||x1 - x2|| / l**2) exp{-√3 * ||x1 - x2|| / l**2}

    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__(lengthscale, variance)

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the kernel value for x1 and x2.

        :param x1: Jax ndarray of the shape `(N1, D)`.
        :param x2: Jax ndarray of the shape `(N2, D)`.

        :return: kernel matrix of the shape `(N1, N2)`.

        """
        x1 = self._handle_input_shape(x1)
        x2 = self._handle_input_shape(x2)
        assert x1.shape[-1] == x2.shape[-1]
        x1 = self._scale_by_lengthscale(x1)
        x2 = self._scale_by_lengthscale(x2)
        dist = jnp.sqrt(sq_euclidean_dist(x1, x2))
        sqrt3 = jnp.sqrt(3.0)
        k = self.variance * (1.0 + sqrt3 * dist) * jnp.exp(-sqrt3 * dist)
        assert k.shape == (x1.shape[0], x2.shape[0])
        return k


class Matern52(Kernel):
    """
    Matern 5/2 Kernel.

    k(x1, x2) = σ² (1 + √5 * (||x1 - x2||) + 5/3 * ||x1 - x2||^2) exp{-√5 * ||x1 - x2||}
    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        super().__init__(lengthscale, variance)

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        """
        Calculates the kernel value for x1 and x2.

        :param x1: Jax ndarray of the shape `(N1, D)`.
        :param x2: Jax ndarray of the shape `(N2, D)`.

        :return: kernel matrix of the shape `(N1, N2)`.

        """
        x1 = self._handle_input_shape(x1)
        x2 = self._handle_input_shape(x2)
        assert x1.shape[-1] == x2.shape[-1]
        x1 = self._scale_by_lengthscale(x1)
        x2 = self._scale_by_lengthscale(x2)
        dist = jnp.sqrt(sq_euclidean_dist(x1, x2))
        sqrt5 = jnp.sqrt(5.0)
        k = self.variance * (1.0 + sqrt5 * dist + 5.0 / 3.0 * jnp.square(dist)) * jnp.exp(-sqrt5 * dist)
        assert k.shape == (x1.shape[0], x2.shape[0])
        return k


class TwoSHO(Kernel):
    """
    Time-domain kernel: Sum of two underdamped simple harmonic oscillators (SHOs).
    Parameterized with log-domain inputs for numerical stability and domain constraints.
    """

    def __init__(self,
                 log_S0_1: float = 7.216781, log_w0_1: float = 1.02038, log_Q_1: float = 5.337618,
                 log_S0_2: float = 5.836664, log_w0_2: float = 1.718205, log_Q_2: float = 3.186107):
        super().__init__()
        self.log_S0_1 = log_S0_1
        self.log_w0_1 = log_w0_1
        self.log_Q_1  = log_Q_1
        self.log_S0_2 = log_S0_2
        self.log_w0_2 = log_w0_2
        self.log_Q_2  = log_Q_2

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        # Step 1: Parameter transformation
        S0_1 = jnp.exp(self.log_S0_1)
        w0_1 = jnp.exp(self.log_w0_1)
        Q_1  = jnp.exp(self.log_Q_1) + 0.5  # ensures Q > 0.5

        S0_2 = jnp.exp(self.log_S0_2)
        w0_2 = jnp.exp(self.log_w0_2)
        Q_2  = jnp.exp(self.log_Q_2) + 0.5

        # Step 2: Handle shapes and extract time vectors
        x1 = self._handle_input_shape(x1)
        x2 = self._handle_input_shape(x2)
        assert x1.shape[-1] == x2.shape[-1] == 1, "Time input must be 1D."

        t1 = x1[:, 0:1]  # shape (N1, 1)
        t2 = x2[:, 0:1].T  # shape (1, N2)
        tau = jnp.abs(t1 - t2)  # pairwise absolute time differences

        # Step 3: SHO kernel function (time domain)
        def sho_term(S0, w0, Q, tau):
            wd = w0 * jnp.sqrt(1.0 - 1.0 / (4.0 * Q**2))
            exp_decay = jnp.exp(-0.5 * w0 * tau / Q)
            cosine = jnp.cos(wd * tau)
            sine = jnp.sin(wd * tau)
            return S0 * w0 * Q * exp_decay * (cosine + sine / (2 * Q * wd))

        # Step 4: Combine two SHO terms
        k1 = sho_term(S0_1, w0_1, Q_1, tau)
        k2 = sho_term(S0_2, w0_2, Q_2, tau)
        k = k1 + k2

        assert k.shape == (x1.shape[0], x2.shape[0])
        return k
