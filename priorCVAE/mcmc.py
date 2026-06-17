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
from numpyro.infer import init_to_median, MCMC, NUTS, init_to_value
from jax.scipy.special import log_ndtr, logsumexp
import jax.lax as lax

from priorCVAE.models import Decoder

os.environ["OMP_NUM_THREADS"] = "1"
multiprocessing.set_start_method("spawn", force=True)


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
    

# --- Helper for robust masking ---
def safe_mask(mask, value):
    """
    Safely applies mask. Returns 0.0 if mask is 0, avoiding NaN propagation.
    Formula: jnp.where(mask > 0.5, value, 0.0)
    """
    return jnp.where(mask > 0.5, value, 0.0)
    


def hmm_forward_loglik(yd, observed_mask,
                       sigma_noise,
                       rate_firing, rate_decay,
                       theta_quiet, theta_firing, theta_decay):
    
    # --- 1. PRE-COMPUTE EMISSIONS (VECTORIZED) ---
    y_curr = yd[1:]   # t=2 to N
    y_prev = yd[:-1]  # t=1 to N-1
    obs_curr = observed_mask[1:] 
    
    # State 1: Quiet
    lp_quiet = npdist.Normal(0.0, sigma_noise).log_prob(y_curr)
    log_emit_quiet = safe_mask(obs_curr, lp_quiet)
    
    # State 2: Firing
    # Note: Ensure expmod_normal_logpdf is available in your scope
    lp_firing = expmod_normal_logpdf(y_curr, y_prev, sigma_noise, rate_firing)
    log_emit_firing = safe_mask(obs_curr, lp_firing)
    
    # State 3: Decay
    lp_decay = npdist.Normal(rate_decay * y_prev, sigma_noise).log_prob(y_curr)
    log_emit_decay = safe_mask(obs_curr, lp_decay)
    
    # Stack: Shape (N-1, 3)
    log_emissions = jnp.stack([log_emit_quiet, log_emit_firing, log_emit_decay], axis=1)

    # --- 2. INITIALIZATION (t=1 / Index 1) ---
    # We use the mask at index 1 to safely compute initial gamma
    mask0 = observed_mask[1]
    
    gq_0 = safe_mask(mask0, npdist.Normal(0.0, sigma_noise).log_prob(yd[1]))
    gf_0 = safe_mask(mask0, expmod_normal_logpdf(yd[1], yd[0], sigma_noise, rate_firing))
    gd_0 = safe_mask(mask0, npdist.Normal(rate_decay * yd[0], sigma_noise).log_prob(yd[1]))
    
    gamma0 = jnp.array([gq_0, gf_0, gd_0])

    # --- 3. TRANSITION MATRIX ---
    ninf = -1e30 # Representing log(0)
    
    l_theta_q = jnp.log(theta_quiet)   
    l_theta_f = jnp.log(theta_firing)  
    l_theta_d = jnp.log(theta_decay)   

    log_trans_mat = jnp.array([
        [l_theta_q[0],  l_theta_q[1],   ninf],           
        [ninf,          l_theta_f[0],   l_theta_f[1]],   
        [l_theta_d[0],  l_theta_d[1],   l_theta_d[2]]    
    ])

    # --- 4. SCAN ---
    def step(prev_gamma, current_log_emission):
        next_prob_matrix = prev_gamma[:, None] + log_trans_mat
        gamma_new = logsumexp(next_prob_matrix, axis=0) + current_log_emission
        return gamma_new, None

    # Run scan on remaining emissions (indices 1 to end)
    gamma_final, _ = jax.lax.scan(step, gamma0, log_emissions[1:])

    return logsumexp(gamma_final)

# ------------------------------------------------------------------
# Main model
# ------------------------------------------------------------------
def vae_hmm_mcmc_inference_model(args, decoder, decoder_params, c=None):
    """
    Joint VAE + HMM model with masking support.
    """
    # 1. Data
    y = args["y_full"]
    N = args["input_dim"]
    
    # Robustly get mask, default to all ones if missing
    observed_mask = args.get("observed_mask", jnp.ones(N))

    # 2. VAE prior
    z_dim = args["latent_dim"]
    z = numpyro.sample("z", npdist.Normal(0, 1).expand([z_dim]).to_event(1))
    if c is not None:
        z = jnp.concatenate([z, c], axis=0)
    trend = decoder.apply({"params": decoder_params}, z)
    numpyro.deterministic("trend", trend)

    # 3. HMM priors
    sigma_noise = numpyro.sample("sigma_noise", npdist.HalfNormal(1.0))
    
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
    yd = y - trend
    
    # PASS THE MASK HERE
    log_lik_hmm = hmm_forward_loglik(
        yd, observed_mask, 
        sigma_noise, rate_firing, rate_decay,
        theta_quiet, theta_firing, theta_decay
    )
    
    numpyro.factor("hmm_log_likelihood", log_lik_hmm)

    # 5. Monitoring
    numpyro.deterministic("rate_firing", rate_firing)
    numpyro.deterministic("rate_decay", rate_decay)


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
    init_values = {
        # VAE: Start with the average trend
        "z": jnp.zeros(args["latent_dim"]), 
        
        # NOISE: Start small (assuming normalized data)
        # If your model samples 'sigma_noise' (HalfNormal), use this:
        "sigma_noise": 0.1,
        # If your model still samples 'sigma2_noise' (InverseGamma), use 0.01
        
        # FIRING: Start with moderate drift
        # rate = 1.0 -> log(1.0) = 0.0
        "lograte_firing": 0.0,
        
        # DECAY: Start with slow decay (easy to see)
        # rate = 0.9 -> logit(0.9) approx 2.2
        "logitrate_decay": 2.2,
        
        # TRANSITIONS (Dirichlet)
        # High probability on the diagonal (staying in state)
        
        # Quiet: 95% stay Quiet, 5% go Firing
        "theta_quiet": jnp.array([0.95, 0.05]),
        
        # Firing: 90% stay Firing, 10% go Decay
        "theta_firing": jnp.array([0.90, 0.10]),
        
        # Decay: 10% go Quiet, 5% go Firing, 85% stay Decay
        # (Must sum to 1.0)
        "theta_decay": jnp.array([0.10, 0.05, 0.85])
    }

 #   init_strategy = init_to_median(num_samples=10)
    kernel = NUTS(
        model,
        init_strategy=init_to_value(values=init_values),
        target_accept_prob=0.90,
        max_tree_depth=12,
        dense_mass=True,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=args["num_warmup"],
        num_samples=args["num_mcmc_samples"],
        num_chains=args["num_chains"],
        thinning=args["thinning"],
        chain_method = "parallel",
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
