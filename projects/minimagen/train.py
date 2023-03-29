import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import IPython
from tqdm import tqdm



class GaussianDiffusion(nn.Module):
    """Add Gaussian noise to the input image."""
    def __init__(self, *, timesteps):
        super().__init__()

        # Implement variance schedule
        self.num_timesteps = timesteps

        scale = 1000 / timesteps # TODO, I don't know why we use this scale
        beta_start = 1e-4 * scale
        beta_end = 2e-2 * scale
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float64)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.) # Adds 1 to the beginning of the tensor

        # register all of this as buffer
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        Building forward diffusion process but understanding first how it works.

        IPython.embed()
GaussianDiffusion(timesteps=10)