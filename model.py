"""
Denoising Diffusion (DDPM) from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - linear_beta_schedule
import torch
import torch.nn.functional as F

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    # TODO: return a linear beta schedule of length T
    beta = torch.zeros(T)
    beta[0] = beta_start
    for i in range(1,T):
        beta[i] = beta[i-1] + (beta_end - beta_start) / (T-1)
    return beta

# Step 2 - alphas_from_betas
import torch
import torch.nn.functional as F

def alphas_from_betas(betas):
    n = betas.shape
    return torch.ones(n) - betas

# Step 3 - cumprod_alphas
import torch
import torch.nn.functional as F

def cumprod_alphas(alphas):
    return torch.cumprod(alphas, dim=0)

# Step 4 - extract_into_batch
import torch
import torch.nn.functional as F

def extract_into_batch(a, t, x):
    # TODO: gather a[t] and reshape to (B, 1, 1, 1) for broadcasting with x
    value = a[t]
    return value.reshape(x.shape[0], 1, 1, 1)

# Step 5 - q_sample
import torch
import torch.nn.functional as F

def q_sample(x0, t, noise, alphas_cumprod):
    # TODO: x_t = sqrt(bar_alpha_t) * x0 + sqrt(1 - bar_alpha_t) * noise
    sqrt_alphas_cumprod = extract_into_batch(alphas_cumprod, t, x0)
    xt = sqrt_alphas_cumprod.sqrt() * x0 + (1-sqrt_alphas_cumprod).sqrt() * noise
    return xt

# Step 6 - build_diffusion_schedule (not yet solved)
# TODO: implement

# Step 7 - noise_prediction_loss (not yet solved)
# TODO: implement

# Step 8 - diffusion_training_loss (not yet solved)
# TODO: implement

# Step 9 - timestep_embedding (not yet solved)
# TODO: implement

# Step 10 - init_tiny_unet (not yet solved)
# TODO: implement

# Step 11 - tiny_unet_forward (not yet solved)
# TODO: implement

# Step 12 - make_blob_dataset (not yet solved)
# TODO: implement

# Step 13 - ddpm_train_step (not yet solved)
# TODO: implement

# Step 14 - train_ddpm (not yet solved)
# TODO: implement

# Step 15 - predict_x0_from_eps (not yet solved)
# TODO: implement

# Step 16 - ddpm_p_mean_variance (not yet solved)
# TODO: implement

# Step 17 - ddpm_p_sample (not yet solved)
# TODO: implement

# Step 18 - ddpm_sample_loop (not yet solved)
# TODO: implement

# Step 19 - sample_quality_mse (not yet solved)
# TODO: implement

# Step 20 - ddpm_experiment (not yet solved)
# TODO: implement

