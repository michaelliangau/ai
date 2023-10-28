import torch
from torch import nn
import torch.nn.functional as F

class ForwardProcess():
    """Adds noise to an image in a forward process."""
    def __init__(self, num_timesteps: int = 100, torch_device: torch.device = torch.device("cuda")) -> None:
        """Initialize the forward process.

        Args:
            num_timesteps: Number of timesteps in the diffusion process.
        """
        self.betas = self.generate_betas(num_timesteps).to(torch_device)
        
        # Used for forward process
        self.alphas = 1 - self.betas
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas.cumprod(dim=0)).to(torch_device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas.cumprod(dim=0)).to(torch_device)

        # Reverse process

        # Calculate x_0 (start_image) from current image
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.alphas.cumprod(dim=0)).to(torch_device)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt((1 / self.alphas.cumprod(dim=0)) - 1).to(torch_device)

        # Calculate mean
        self.posterior_mean_coef1 = (torch.sqrt(self.alphas.cumprod(dim=0)) * self.betas) / (1 - self.alphas.cumprod(dim=0))
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_mean_coef2 = torch.sqrt(self.alphas) * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_variance = (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * self.betas
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))



    def generate_betas(self, num_timesteps: int) -> torch.Tensor:
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
    
    def sample(self, image: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Sample from the forward process at a specific timestep.
        
        According to DDPM paper, we're sampling from a Gaussian distribution with mean of
        (sqrt_alphas_cumprod * image) and variance of (1 - alphas_cum_prod) * I.

        Args:
            image: The image to noise.
            timestep: The timestep to sample at.
        """
        # epsilon ~ N(0, I)
        noise = torch.randn_like(image)

        # Create noised image
        noised = self.sqrt_alphas_cumprod[timestep].reshape(image.shape[0], 1, 1, 1) * image + self.sqrt_one_minus_alphas_cumprod[timestep].reshape(image.shape[0], 1, 1, 1) * noise
        
        return noised
    
    def predict_start_from_noise(self, image: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Predict the start image from a noised image.
        
        Args:
            image: The noised image.
            timestep: The timestep to predict from.
        
        Returns:
            The predicted start image (x_0).
        """
        noise = torch.randn_like(image)
        start_image = self.sqrt_recip_alphas_cumprod[timestep].reshape(image.shape[0], 1, 1, 1) * image - self.sqrt_recipm1_alphas_cumprod[timestep].reshape(image.shape[0], 1, 1, 1) * noise

        return start_image

    def q_posterior(self, start_image: torch.Tensor, current_image: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Calculate the posterior distribution of q(x_t|x_0).
        
        Args:
            start_image: The image.
            current_image: The current image.
            timestep: The timestep.
        
        Returns:
            The posterior distribution of q(x_t|x_0).
        """
        # Calculate posterior mean
        # TODO: Validate shapes match, you probably need to pull at a specific timestep
        posterior_mean = self.posterior_mean_coef1 * start_image + self.posterior_mean_coef2 * current_image
        posterior_variance = self.posterior_variance
        posterior_log_variance_clipped = self.posterior_log_variance_clipped
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    

class BackwardProcess():
    """Generates an image from a noised image in a backward process."""
    def __init__(self, model, torch_device=torch.device("cuda")) -> None:
        """
        Initialize the backward process.

        Args:
            model: The model to be used in the backward process.
        """
        self.unet = model
        self.torch_device = torch_device
    
    def predict(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Predict the amount of noise
        
        TODO: You can also embed timestep into the upsampling.

        Args:
            image (torch.Tensor): The image to denoise. Shape is (batch_size, channels, height, width).
            text (torch.Tensor): The text embedding. Shape is (batch_size, embedding_dim).
        
        Returns:
            torch.Tensor: Predict the amount of noise. Shape is (batch_size, channels, height, width).
        """
        output = self.unet(image, text)
        return output

class UNet(nn.Module):
    """This UNet is the main workhorse of the backward denoising process."""

    def __init__(self):
        """Initialize the UNet model."""
        super(UNet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1)
        )

        self.embedding_projector = nn.Linear(512, 256)


    def forward(self, x: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UNet model.

        Args:
            x (torch.Tensor): The input tensor, typically an image.
            text_embedding (torch.Tensor): The text embedding tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        # Encode
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        # Project the text embedding to 256 dimensions
        text_embedding = self.embedding_projector(text_embedding)

        # Expand text embedding into same dim as enc3
        text_embedding = text_embedding.unsqueeze(-1).unsqueeze(-1).expand(enc3.shape)

        # Concatenate enc3 and text_embedding
        enc3 = enc3 + text_embedding

        # Decode
        dec2 = self.dec2(torch.cat([self.up2(enc3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        return dec1