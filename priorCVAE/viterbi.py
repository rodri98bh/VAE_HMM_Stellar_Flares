"""
File contains the code for the Viterbi Algorithm
"""

import numpy as np
import jax.numpy as jnp
from scipy.special import log_ndtr
from scipy.stats import norm

# ------------------------------------------------------------------
# Helper: Raw MAD (Robust Noise Estimation)
# ------------------------------------------------------------------


def robust_normalize(flux):
    """
    Normalizes light curves using Median and MAD (Median Absolute Deviation).
    Robust against outliers (flares) and handles missing data (NaNs).
    """
    # 1. Center on the Median (Robust Baseline)
    # np.nanmedian ignores NaNs so gaps don't break the calculation
    median_val = np.nanmedian(flux)
    
    # 2. Scale by MAD (Robust Noise Estimation)
    # We calculate deviation only on valid points
    diff = np.abs(flux - median_val)
    mad_val = 1.4826 * np.nanmedian(diff)
    
    # Safety: Avoid division by zero if the light curve is flat or empty
    if mad_val == 0 or np.isnan(mad_val):
        # Fallback to standard deviation (ignoring NaNs), or 1.0 if that fails
        std_val = np.nanstd(flux)
        mad_val = std_val if std_val > 0 else 1.0

    # 3. Normalize
    # NaNs in the input 'flux' will remain NaNs in the output (which is correct)
    return (flux - median_val) / mad_val

# ------------------------------------------------------------------
# Helper: Exponentially modified normal log-pdf (Stan equivalent)
# ------------------------------------------------------------------

def log_pdf_expmod_numpy(x, mu, sigma, lam):
    """
    Numpy version of the ExpModNormal log-pdf for Viterbi decoding.
    Matches the logic used in the likelihood function.
    """
    ls = lam * sigma
    # Avoid division by zero
    s_safe = np.maximum(sigma, 1e-10) 
    arg = (mu + ls * sigma - x) / (np.sqrt(2.0) * s_safe)
    
    # Base terms
    base_term = (
        np.log(lam) 
        - np.log(2.0) 
        + lam * (mu - x) 
        + 0.5 * (ls ** 2)
    )
    
    # Stable log(erfc(arg)) using log_ndtr
    log_erfc_term = np.log(2.0) + log_ndtr(-arg * np.sqrt(2.0))
    
    return base_term + log_erfc_term


def create_observed_mask(flux):
    """
    Creates a binary mask for the HMM.
    1 = Observed (Valid Data)
    0 = Missing (NaN or Inf)
    """
    # 1. Check for NaNs (Not a Number)
    is_nan = np.isnan(flux)
    
    # 2. Check for Infinite values (optional but recommended safety)
    is_inf = np.isinf(flux)
    
    # 3. Combine: Data is "Bad" if it is NaN OR Inf
    is_bad = is_nan | is_inf
    
    # 4. Invert: Data is "Observed" (1) if it is NOT Bad
    # Convert boolean (True/False) to float (1.0/0.0)
    mask = (~is_bad).astype(float)
    
    # Return as JAX array for compatibility with your model
    return jnp.array(mask)


# ------------------------------------------------------------------
# The Viterbi Algorithm
# ------------------------------------------------------------------

def run_viterbi_decoding(y, trend, observed_mask, params):
    """
    Viterbi with 'Physics Gate': Enforces that Flares must be positive.
    """
    # Unpack Params
    sigma = float(params['sigma_noise'])
    rate_f = float(params['rate_firing']) 
    rate_d = float(params['rate_decay'])  
    
    log_theta_q = np.log(params['theta_quiet'])  
    log_theta_f = np.log(params['theta_firing']) 
    log_theta_d = np.log(params['theta_decay'])  
    
    yd = np.array(y - trend)
    N = len(yd)
    obs = np.array(observed_mask)

    best_logp = np.full((N, 3), -np.inf)
    back_ptr = np.zeros((N, 3), dtype=int)
    
    # --- Helper to calculate log_prob safely ---
    def get_log_emit(val_curr, val_prev, state, mask_val):
        if mask_val == 0:
            return 0.0 
        
        # ==========================================================
        # The Physics Gate
        # ==========================================================
        # If the data point is below the trend (negative), it is impossible
        # for it to be a flare. We return -infinity probability.
        if state == 1 and val_curr < (-0.1 * sigma): 
            return -np.inf 
        # ==========================================================

        if state == 0: # Quiet
            return norm.logpdf(val_curr, 0, sigma)
        elif state == 1: # Firing
            return log_pdf_expmod_numpy(val_curr, val_prev, sigma, rate_f)
        elif state == 2: # Decay
            return norm.logpdf(val_curr, rate_d * val_prev, sigma)
        return -np.inf

    # --- Initialization ---
    best_logp[1, 0] = get_log_emit(yd[1], yd[0], 0, obs[1])
    best_logp[1, 1] = get_log_emit(yd[1], yd[0], 1, obs[1])
    best_logp[1, 2] = get_log_emit(yd[1], yd[0], 2, obs[1])
    
    # --- Forward Pass ---
    for t in range(2, N):
        y_curr = yd[t]
        y_prev = yd[t-1]
        mask_val = obs[t]
        
        # Calculate Emissions safely
        emi_q = get_log_emit(y_curr, y_prev, 0, mask_val)
        emi_f = get_log_emit(y_curr, y_prev, 1, mask_val)
        emi_d = get_log_emit(y_curr, y_prev, 2, mask_val)
        
        # 1. To Quiet
        s_q = [best_logp[t-1, 0] + log_theta_q[0], 
               -np.inf, 
               best_logp[t-1, 2] + log_theta_d[0]]
        best_logp[t, 0] = np.max(s_q) + emi_q
        back_ptr[t, 0] = np.argmax(s_q)
            
        # 2. To Firing
        s_f = [best_logp[t-1, 0] + log_theta_q[1],
               best_logp[t-1, 1] + log_theta_f[0],
               best_logp[t-1, 2] + log_theta_d[1]]
        best_logp[t, 1] = np.max(s_f) + emi_f
        back_ptr[t, 1] = np.argmax(s_f)

        # 3. To Decay
        s_d = [-np.inf,
               best_logp[t-1, 1] + log_theta_f[1],
               best_logp[t-1, 2] + log_theta_d[2]]
        best_logp[t, 2] = np.max(s_d) + emi_d
        back_ptr[t, 2] = np.argmax(s_d)

    # --- Backtracking ---
    viterbi_path = np.zeros(N, dtype=int)
    viterbi_path[N-1] = np.argmax(best_logp[N-1])
    
    for t in range(N-1, 1, -1):
        prev_state = back_ptr[t, viterbi_path[t]]
        viterbi_path[t-1] = prev_state
        
    return viterbi_path + 1