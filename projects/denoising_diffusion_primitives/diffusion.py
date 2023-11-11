import torch
from typing import Tuple
import torch.nn.functional as F


class GaussianDiffusion:
    """Adds noise to an image in a forward process."""

    def __init__(
        self,
        num_timesteps: int = 100,
        torch_device: torch.device = torch.device("cuda"),
    ) -> None:
        """Initialize the forward process.

        Args:
            num_timesteps: Number of timesteps in the diffusion process.
        """
        self.num_timesteps = num_timesteps
        self.betas = self._generate_betas(num_timesteps).to(torch_device)

        # Used for forward process
        self.alphas = 1 - self.betas
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas.cumprod(dim=0)).to(
            torch_device
        )
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1 - self.alphas.cumprod(dim=0)
        ).to(torch_device)

        # Calculate x_0 (start_image) from current image
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.alphas.cumprod(dim=0)).to(
            torch_device
        )
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            (1 / self.alphas.cumprod(dim=0)) - 1
        ).to(torch_device)

        # Calculate mean
        self.posterior_mean_coef1 = (
            torch.sqrt(self.alphas.cumprod(dim=0)) * self.betas
        ) / (1 - self.alphas.cumprod(dim=0))
        self.alphas_cumprod_prev = F.pad(
            self.alphas.cumprod(dim=0)[:-1], (1, 0), value=1.0
        )
        self.posterior_mean_coef2 = (
            torch.sqrt(self.alphas)
            * (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas.cumprod(dim=0))
        )
        self.posterior_variance = (
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas.cumprod(dim=0))
            * self.betas
        )
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )

    def _generate_betas(self, num_timesteps: int) -> torch.Tensor:
        """Generate an array of betas for diffusion.

        DDPM paper specifies beta start and end values to be 1e-4 and 2e-2 respectively.

        Args:
            num_timesteps: Number of timesteps in the diffusion process.

        Returns:
            A torch.Tensor containing generated betas.
        """
        # Scaling the betas helps to ensure that even with different timesteps the magnitude
        # of each timestep scales with it.
        scale = 1000 / num_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 2e-2
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        return betas

    def sample_random_times(
        self, batch_size: int, device: torch.device
    ) -> torch.tensor:
        """Randomly sample `batch_size` timestep values uniformly from [0, 1, ..., `self.num_timesteps`]

        Args:
            batch_size (int): Number of images in the batch.
            device (torch.device): Device on which to place the return tensor.

        Returns:
            torch.tensor: Tensor of integers (`dtype=torch.long`) of shape `(batch_size,)`
        """
        return torch.randint(
            0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long
        )

    def sample(
        self, image: torch.Tensor, timestep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the forward process at a specific timestep.

        According to DDPM paper, we're sampling from a Gaussian distribution with mean of
        (sqrt_alphas_cumprod * image) and variance of (1 - alphas_cum_prod) * I.

        Args:
            image (torch.Tensor): The image to noise.
            timestep (torch.Tensor): The timestep to sample at.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The noised image and the noise.
        """
        # epsilon ~ N(0, I)
        noise = torch.randn_like(image)

        # Create noised image
        noised = (
            self.sqrt_alphas_cumprod[timestep].reshape(image.shape[0], 1, 1, 1) * image
            + self.sqrt_one_minus_alphas_cumprod[timestep].reshape(
                image.shape[0], 1, 1, 1
            )
            * noise
        )

        return noised, noise

    def predict_start_from_noise(
        self, image: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """Predict the start image from a noised image.

        Args:
            image: The noised image.
            timestep: The timestep to predict from.

        Returns:
            The predicted start image (x_0).
        """
        noise = torch.randn_like(image)
        start_image = (
            self.sqrt_recip_alphas_cumprod[timestep].reshape(image.shape[0], 1, 1, 1)
            * image
            - self.sqrt_recipm1_alphas_cumprod[timestep].reshape(
                image.shape[0], 1, 1, 1
            )
            * noise
        )

        return start_image

    def q_posterior(
        self,
        start_image: torch.Tensor,
        current_image: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the posterior distribution of q(x_t|x_0).

        These values are used in the reverse process to calculate each iterative
        diffusion step.

        Args:
            start_image: The image.
            current_image: The current image.
            timestep: The timestep.

        Returns:
            The posterior distribution of q(x_t|x_0).
        """
        # Calculate posterior mean
        # TODO: Validate shapes match, you probably need to pull at a specific timestep
        posterior_mean = (
            self.posterior_mean_coef1 * start_image
            + self.posterior_mean_coef2 * current_image
        )
        posterior_variance = self.posterior_variance
        posterior_log_variance_clipped = self.posterior_log_variance_clipped
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
