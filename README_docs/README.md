# Variational Autoencoder + hidden Markov model for Stellar Flare Detection

This repository provides the codebase required to run the full VAE+HMM framework presented in the following paper:

Herrera, R., Leos-Barajas, V., Eadie, G., & Semenova, E. (Year). Scalable Bayesian Additive Models for Stellar Flare Detection via Amortized Gaussian Process Inference and Hidden Markov Models.

# PriorCVAE in JAX

The main VAE structure is based on the following two papers:

1. Elizaveta Semenova, Yidan Xu, Adam Howes, Theo Rashid, Samir Bhatt, Swapnil Mishra, Seth Flaxman["PriorVAE: encoding spatial priors with variational autoencoders for small-area estimation."](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2022.0094) Journal of the Royal Society Interface 19.191 (2022): 20220094. Original code is avilable [here](https://github.com/elizavetasemenova/PriorVAE). 
2. Elizaveta Semenova, Prakhar Verma, Max Cairney-Leeming, Arno Solin, Samir Bhatt, Seth Flaxman ["PriorCVAE: scalable MCMC parameter inference with Bayesian deep generative modelling."](https://arxiv.org/abs/2304.04307) arXiv preprint arXiv:2304.04307 (2023). Original code is avilable [here](https://github.com/elizavetasemenova/PriorcVAE).


## Environment

Before running or modifying any scripts, please ensure the required environment is properly set up. Detailed setup instructions can be found in `INSTALL.md`.

## Main Script

The main Jupyter Notebook demonstrating the full algorithm described in the paper is located at: 

`VAE_HMM/PriorVAE_HMM_TwoSHO.ipynb`

**Note:** The notebook includes intermediary cells—specifically for loading and saving the parameters/weights of the decoders. If you are testing or training a new model from scratch, you may not need to run these specific cells. Please follow the section titles and descriptions provided within the notebook for guidance.

**Note:** For experiments it is recommended to use float64 precision to avoid numerical instability:
```python
import jax.config as config
config.update("jax_enable_x64", True)
```

## Data

The datasets containing the brightness-over-time observations for the three M-dwarf stars analyzed in the paper (TIC 031381302, TIC 089257479, and TIC 234526939) are located in the following directory: 

`VAE_HMM/data/Stellar_Flares/`

## Configurations & Checkpoints

Finally, the pre-trained VAE configurations used to model the three stellar objects of interest are stored in:

`VAE_HMM/checkpoints/`

**Note:** If you want to bypass the training phase and reproduce the exact results shown in the paper, you can load these configurations directly into the main notebook without needing to retrain the models from scratch.