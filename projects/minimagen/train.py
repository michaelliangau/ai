# Our imports
import utils

# Third party imports
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
        # Beta is the variance of the gaussian noise added to the image at each timestep
        beta_start = 1e-4 * scale
        beta_end = 2e-2 * scale # larger beta = greater variance towards the identity matrix (?) and mean getting closer to 0
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float64) # variance schedule at each timestep.

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.) # Adds 1 to the beginning of the tensor
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """Apply gaussian diffusion to timestep t
        
        Args:
            x_start (torch.tensor): input image
            t (torch.tensor): timestep
            noise (torch.tensor, optional): noise to add to the image. Defaults to None.

        Returns:
            noised (torch.tensor): noised image
        """
        noise = utils.default(noise, lambda: torch.randn_like(x_start)) # torch.randn_like returns a tensor with the same size as x_start, filled with random numbers from a normal distribution with mean 0 and variance 1.

        # This is a shortcut way of sampling the accumulated noised image at specific timestep t - https://roysubhradip.hashnode.dev/a-beginners-guide-to-diffusion-models-understanding-the-basics-and-beyond
        noised = (
            utils.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + # multiple every element of x_start by the t-th index of sqrt of the alphas_cumprod
            utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return noised

m = GaussianDiffusion(timesteps=1000)

x_start = torch.randn(1, 3, 32, 32)
t = torch.tensor([5])
_ = m.q_sample(x_start, t)
IPython.embed()