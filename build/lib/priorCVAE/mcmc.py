"""
File contains the code for Monte Carlo Markov Chain (MCMC) used for inference.
"""
from typing import Dict
import time
import os
import jax
import jax.numpy as jnp
import multiprocessing

import numpy as np
from jax.random import KeyArray
import jax.numpy as jnp
import numpyro
import numpyro.distributions as npdist
from numpyro.infer import init_to_median, MCMC, NUTS
from jax.scipy.special import log_ndtr, logsumexp
import jax.lax as lax

from priorCVAE.models import Decoder

os.environ["OMP_NUM_THREADS"] = "1"
multiprocessing.set_start_method("spawn", force=True)

def vae_mcmc_inference_model(args: Dict, decoder: Decoder, decoder_params: Dict, c: jnp.array = None):
    """
    VAE numpyro model used for running MCMC inference.

    :param args: a dictionary with the arguments required for MCMC.
    :param decoder: a decoder model.
    :param decoder_params: a dictionary with decoder network parameters.
    :param c: a Jax ndarray used for cVAE of the shape, (N, C).
    """

    z_dim = args["latent_dim"]
    y = args["y_obs"]
    obs_idx = args["obs_idx"]

    z = numpyro.sample("z", npdist.Normal(jnp.zeros(z_dim), jnp.ones(z_dim)))  # (Z_dim,)
    if c is not None:
        z = jnp.concatenate([z, c], axis=0)  # (Z_dim + C, )

    f = numpyro.deterministic("f", decoder.apply({'params': decoder_params}, z))
    sigma = numpyro.sample("sigma", npdist.HalfNormal(0.1))

    if y is None:  # during prediction
        y_pred = numpyro.sample("y_pred", npdist.Normal(f, sigma))
    else:  # during inference
        y = numpyro.sample("y", npdist.Normal(f[obs_idx], sigma), obs=y)


# ------------------------------------------------------------------
# Helper: Exponentially modified normal log-pdf (Stan equivalent)
# ------------------------------------------------------------------
def expmod_normal_logpdf(x, mu, sigma, lam):
    """
    Numerically stable log-pdf using log_ndtr instead of erfcx.
    """
    ls = lam * sigma
    arg = (mu + ls * sigma - x) / (jnp.sqrt(2.0) * sigma)
    
    # 1. Base terms
    # log(lambda/2) + lambda*(mu - x) + lambda^2*sigma^2/2
    base_term = (
        jnp.log(lam) 
        - jnp.log(2.0) 
        + lam * (mu - x) 
        + 0.5 * (ls ** 2)
    )
    
    # 2. Stable log(erfc(arg)) using log_ndtr
    # log(erfc(z)) = log(2) + log_ndtr(-z * sqrt(2))
    # We use -arg * sqrt(2) because log_ndtr is strictly for the standard normal CDF
    log_erfc_term = jnp.log(2.0) + log_ndtr(-arg * jnp.sqrt(2.0))
    
    return base_term + log_erfc_term
    
# ------------------------------------------------------------------
# Helper: HMM forward algorithm (Stan style)
# ------------------------------------------------------------------
def hmm_forward_loglik(yd, observed_mask,
                       sigma_noise,
                       rate_firing, rate_decay,
                       theta_quiet, theta_firing, theta_decay):
    N = yd.shape[0]

    gq = observed_mask[1] * npdist.Normal(0.0, sigma_noise).log_prob(yd[1])
    gf = observed_mask[1] * expmod_normal_logpdf(yd[1], yd[0], sigma_noise, rate_firing)
    gd = observed_mask[1] * npdist.Normal(rate_decay * yd[0], sigma_noise).log_prob(yd[1])
    gamma0 = jnp.array([gq, gf, gd])

    def step(prev_gamma, t):
        prev_y = yd[t - 1]
        ycur = yd[t]
        obs = observed_mask[t]

        emi_q = obs * npdist.Normal(0.0, sigma_noise).log_prob(ycur)
        emi_f = obs * expmod_normal_logpdf(ycur, prev_y, sigma_noise, rate_firing)
        emi_d = obs * npdist.Normal(rate_decay * prev_y, sigma_noise).log_prob(ycur)

        q1 = prev_gamma[0] + jnp.log(theta_quiet[0]) + emi_q
        q2 = prev_gamma[2] + jnp.log(theta_decay[0]) + emi_q
        gamma_q = logsumexp(jnp.array([q1, q2]))

        f1 = prev_gamma[0] + jnp.log(theta_quiet[1]) + emi_f
        f2 = prev_gamma[1] + jnp.log(theta_firing[0]) + emi_f
        f3 = prev_gamma[2] + jnp.log(theta_decay[1]) + emi_f
        gamma_f = logsumexp(jnp.array([f1, f2, f3]))

        d1 = prev_gamma[1] + jnp.log(theta_firing[1]) + emi_d
        d2 = prev_gamma[2] + jnp.log(theta_decay[2]) + emi_d
        gamma_d = logsumexp(jnp.array([d1, d2]))

        new_gamma = jnp.array([gamma_q, gamma_f, gamma_d])
        return new_gamma, new_gamma

    _, gamma_hist = jax.lax.scan(step, gamma0, jnp.arange(2, N))
    return logsumexp(gamma_hist[-1])

# ------------------------------------------------------------------
# Main model
# ------------------------------------------------------------------
def vae_hmm_mcmc_inference_model(args, decoder, decoder_params, c=None):
    """
    Joint VAE (trend prior) + HMM (likelihood) model.

    Assumes y_full is already normalized, like in Stan.
    """

    # 1. Data
    y = args["y_full"]
    N = args["input_dim"]
    observed_mask = args.get("observed_mask", jnp.ones(N))

    # 2. VAE prior: latent z and decoded trend
    z_dim = args["latent_dim"]
    z = numpyro.sample("z", npdist.Normal(0, 1).expand([z_dim]).to_event(1))
    if c is not None:
        z = jnp.concatenate([z, c], axis=0)
    trend = decoder.apply({"params": decoder_params}, z)
    numpyro.deterministic("trend", trend)

    # 3. HMM priors
    sigma2_noise = numpyro.sample("sigma2_noise",
                                  npdist.InverseGamma(args["gamma_noise"][0], args["gamma_noise"][1]))
    sigma_noise = jnp.sqrt(sigma2_noise)

    #mu_quiet = numpyro.sample("mu_quiet",
                              #npdist.Normal(args["mu0_quiet"], jnp.sqrt(sigma2_noise / args["lambda_quiet"])))

    lograte_firing = numpyro.sample("lograte_firing",
                                    npdist.Normal(args["mu0_rate_firing"], args["sigma_rate_firing"]))
    rate_firing = jnp.exp(lograte_firing)

    logitrate_decay = numpyro.sample("logitrate_decay",
                                     npdist.Normal(args["mu0_rate_decay"], args["sigma_rate_decay"]))
    rate_decay = jax.nn.sigmoid(logitrate_decay)

    theta_quiet = numpyro.sample("theta_quiet", npdist.Dirichlet(args["alpha_quiet"]))
    theta_firing = numpyro.sample("theta_firing", npdist.Dirichlet(args["alpha_firing"]))
    theta_decay = numpyro.sample("theta_decay", npdist.Dirichlet(args["alpha_decay"]))

    # 4. HMM likelihood
    yd = y - trend - 0
    log_lik_hmm = hmm_forward_loglik(
        yd, observed_mask, sigma_noise, rate_firing, rate_decay,
        theta_quiet, theta_firing, theta_decay
    )
    numpyro.factor("hmm_log_likelihood", log_lik_hmm)

    # 5. Monitoring
    numpyro.deterministic("rate_firing", rate_firing)
    numpyro.deterministic("rate_decay", rate_decay)
    numpyro.deterministic("sigma_noise_det", sigma_noise)


'''
LOG2PI = jnp.log(2.0 * jnp.pi)

def erfcx(x):
    """Stable scaled complementary error function."""
    return jnp.exp(x**2) * erfc(x)

def normal_logpdf(x, mean, sigma):
    z = (x - mean) / sigma
    return -0.5 * (LOG2PI + 2.0 * jnp.log(sigma) + z * z)

def hmm_only_model(args):
    """
    HMM-only NumPyro model (Quiet / Firing / Decay) with same structure as your Stan code.
    The trend is FIXED and passed in, so we can diagnose the HMM separately.

    Required args:
      - y_full:          (N,) unnormalized flux
      - trend_fixed:     (N,) fixed trend on original scale (e.g., from decoder or zeros)
      - input_dim:       int N
      - alpha_quiet:     (2,) Dirichlet prior
      - alpha_firing:    (2,) Dirichlet prior
      - alpha_decay:     (3,) Dirichlet prior
      - mu0_quiet:       float
      - lambda_quiet:    float
      - mu0_rate_firing: float
      - sigma_rate_firing: float
      - mu0_rate_decay:  float
      - sigma_rate_decay: float

    Optional:
      - observed_mask:   (N,) {0,1}  (defaults to ones)
    """
    # --- data ---
    y_raw  = args["y_full"]
    trend  = args["trend_fixed"]
    N      = args["input_dim"]
    observed_mask = args.get("observed_mask", jnp.ones(N))

    # --- priors (non-centered mu, sigma on sigma-not-sigma^2) ---
    sigma_noise = numpyro.sample("sigma_noise", npdist.HalfNormal(200.0))

    mu_tilde = numpyro.sample("mu_tilde", npdist.Normal(0.0, 1.0))
    mu_quiet = args["mu0_quiet"] + (sigma_noise / jnp.sqrt(args["lambda_quiet"])) * mu_tilde
    numpyro.deterministic("mu_quiet", mu_quiet)

    lograte_firing = numpyro.sample("lograte_firing",
                                    npdist.Normal(args["mu0_rate_firing"], args["sigma_rate_firing"]))
    rate_firing = jnp.exp(lograte_firing)

    logitrate_decay = numpyro.sample("logitrate_decay",
                                     npdist.Normal(args["mu0_rate_decay"], args["sigma_rate_decay"]))
    rate_decay = jax.nn.sigmoid(logitrate_decay)

    theta_quiet  = numpyro.sample("theta_quiet",  npdist.Dirichlet(args["alpha_quiet"]))
    theta_firing = numpyro.sample("theta_firing", npdist.Dirichlet(args["alpha_firing"]))
    theta_decay  = numpyro.sample("theta_decay",  npdist.Dirichlet(args["alpha_decay"]))

    # --- detrended series for HMM ---
    yd = y_raw - trend - mu_quiet

    # --- forward algorithm (Stan 1:1; start at t=2 using y[1] with prev y[0]) ---
    gamma0 = jnp.array([
        observed_mask[1] * normal_logpdf(yd[1], 0.0,              sigma_noise),                 # quiet
        observed_mask[1] * expmod_normal_logpdf(yd[1], yd[0],     sigma_noise, rate_firing),    # firing
        observed_mask[1] * normal_logpdf(yd[1], rate_decay*yd[0], sigma_noise)                  # decay
    ])

    def step(prev_gamma, t):
        prev = yd[t - 1]
        ycur = yd[t]
        obs  = observed_mask[t]

        emi_q = obs * normal_logpdf(ycur, 0.0,             sigma_noise)
        emi_f = obs * expmod_normal_logpdf(ycur, prev,     sigma_noise, rate_firing)
        emi_d = obs * normal_logpdf(ycur, rate_decay*prev, sigma_noise)

        # quiet <- {quiet, decay}
        q1 = prev_gamma[0] + jnp.log(theta_quiet[0]) + emi_q
        q2 = prev_gamma[2] + jnp.log(theta_decay[0]) + emi_q
        gamma_q = logsumexp(jnp.array([q1, q2]))

        # firing <- {quiet, firing, decay}
        f1 = prev_gamma[0] + jnp.log(theta_quiet[1])  + emi_f
        f2 = prev_gamma[1] + jnp.log(theta_firing[0]) + emi_f
        f3 = prev_gamma[2] + jnp.log(theta_decay[1])  + emi_f
        gamma_f = logsumexp(jnp.array([f1, f2, f3]))

        # decay <- {firing, decay}
        d1 = prev_gamma[1] + jnp.log(theta_firing[1]) + emi_d
        d2 = prev_gamma[2] + jnp.log(theta_decay[2])  + emi_d
        gamma_d = logsumexp(jnp.array([d1, d2]))

        new_gamma = jnp.array([gamma_q, gamma_f, gamma_d])
        return new_gamma, new_gamma

    _, gamma_hist = jax.lax.scan(step, gamma0, jnp.arange(2, N))
    log_lik_hmm = logsumexp(gamma_hist[-1])
    numpyro.factor("hmm_log_likelihood", log_lik_hmm)

    # diagnostics
    numpyro.deterministic("rate_firing", rate_firing)
    numpyro.deterministic("rate_decay", rate_decay)
'''

def run_mcmc_vae(rng_key: KeyArray, model: numpyro.primitives, args: Dict, decoder: Decoder, decoder_params: Dict,
                 c: jnp.array = None, verbose: bool = True) -> [MCMC, jnp.ndarray, float]:
    """
    Run MCMC inference using VAE decoder.

    :param rng_key: a PRNG key used as the random key.
    :param model: a numpyro model of the type numpypro primitives.
    :param args: a dictionary with the arguments required for MCMC.
    :param decoder: a decoder model.
    :param decoder_params: a dictionary with decoder network parameters.
    :param c: a Jax ndarray used for cVAE of the shape, (N, C).
    :param verbose: if True, prints the MCMC summary.

    Returns:
        - MCMC object
        - MCMC samples
        - time taken

    """
    init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(
        kernel,
        num_warmup=args["num_warmup"],
        num_samples=args["num_mcmc_samples"],
        num_chains=args["num_chains"],
        thinning=args["thinning"],
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    start = time.time()
    mcmc.run(rng_key, args, decoder, decoder_params, c)
    t_elapsed = time.time() - start
    if verbose:
        mcmc.print_summary(exclude_deterministic=False)

    print("\nMCMC elapsed time:", round(t_elapsed), "s")
    ss = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))
    r = np.mean(ss['f']['n_eff'])
    print("Average ESS for all VAE-GP effects : " + str(round(r)))

    return mcmc, mcmc.get_samples(), t_elapsed


def run_mcmc_vae_hmm(rng_key: KeyArray, model: numpyro.primitives, args: Dict, decoder: Decoder, decoder_params: Dict,
                 c: jnp.array = None, verbose: bool = True) -> [MCMC, jnp.ndarray, float]:
    """
    Run MCMC inference using VAE decoder (or VAE+HMM model).

    :param rng_key: PRNG key.
    :param model: numpyro model.
    :param args: dictionary with model arguments.
    :param decoder: decoder model.
    :param decoder_params: decoder network parameters.
    :param c: optional conditioning vector.
    :param verbose: if True, prints the MCMC summary.
    """

 #   init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(
        model,
        init_strategy=init_to_median(num_samples=10),
        target_accept_prob=0.99,
        max_tree_depth=12,
        dense_mass=True,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args["num_warmup"],
        num_samples=args["num_mcmc_samples"],
        num_chains=args["num_chains"],
        thinning=args["thinning"],
        chain_method = "sequential",
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )

    start = time.time()
    mcmc.run(rng_key, args, decoder, decoder_params, c)
    t_elapsed = time.time() - start

    if verbose:
        mcmc.print_summary(exclude_deterministic=False)

    print("\nMCMC elapsed time:", round(t_elapsed), "s")

    # ---- FIXED SECTION BELOW ----
    ss = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))

    # Pick a reasonable site to report ESS for
    if "trend" in ss:
        site = "trend"
    elif "z" in ss:
        site = "z"
    elif "mu_quiet_post" in ss:
        site = "mu_quiet_post"
    else:
        site = list(ss.keys())[0]  # fallback

    r = np.mean(ss[site]["n_eff"])
    print(f"Average ESS for {site}: {round(r)}")

    return mcmc, mcmc.get_samples(), t_elapsed
