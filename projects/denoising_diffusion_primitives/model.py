import torch
from torch import nn
import numpy as np

class ForwardProcess():
    """Adds noise to an image in a forward process."""
    def __init__(self, num_timesteps=100, initial_beta=0.2, decay_rate=0.98):
        """Initialize the forward process.

        Args:
            num_timesteps: Number of timesteps in the diffusion process.
            initial_beta: Initial beta value. This is a hyperparameter that we tune.
                It represents what is the standard deviation of the noise that we add to
                the images at the first timestep (which has maximum noise).
            decay_rate: Decay rate for each subsequent beta.
        """
        self.betas = self.generate_betas(num_timesteps, initial_beta, decay_rate)
    
    def generate_betas(self, num_timesteps, initial_beta, decay_rate):
        """Generate an array of betas for diffusion.
        
        Q: Why is betas going from high values to low?
        A: It follows the timesteps of the backward process which starts from lots of
            noise and gradually removes noise.
        
        Args:
            num_timesteps: Number of timesteps in the diffusion process.
            initial_beta: Initial beta value.
            decay_rate: Decay rate for each subsequent beta.
            
        Returns:
            A torch.Tensor containing generated betas.
        """
        betas = [initial_beta]
        for _ in range(1, num_timesteps):
            next_beta = betas[-1] * decay_rate
            betas.append(next_beta)
        
        # Normalize betas so their sum is 1 (optional)
        betas = np.array(betas)
        betas /= np.sum(betas)

        return torch.tensor(betas)
    
    def sample(self, image, timestep):
        """Sample from the forward process at a specific timestep.
        
        Args:
            image: The image to noise.
            timestep: The timestep to sample at.
        """
        noise_std = torch.sqrt(self.betas[timestep])
        noise = torch.randn_like(image) * noise_std
        return image + noise

class BackwardProcess():
    """Generates an image from a noised image in a backward process."""
    def __init__(self):
        """Init the backward process."""
        self.unet = UNet()
        self.downsample_text_embedding_layer = nn.Linear(512, 128)
    
    
    def denoise(self, image: torch.Tensor, text: torch.Tensor, timestep: int) -> torch.Tensor:
        """Denoise an image at a specific timestep.
        
        Args:
            image (torch.Tensor): The image to denoise. Shape is (batch_size, channels, height, width).
            text (torch.Tensor): The text embedding. Shape is (batch_size, embedding_dim).
            timestep (int): The timestep to denoise at.
        
        Returns:
            torch.Tensor: The denoised image. Shape is (batch_size, channels, height, width).
        """
        # Push image through UNet encoder
        image_embedding = self.unet.forward_encoder(image)

        # Expand text embedding into same dim as encoded_image
        text_embedding = self.downsample_text_embedding_layer(text)
        text_embedding = text_embedding.unsqueeze(-1).unsqueeze(-1)
        text_embedding = text_embedding.expand(image_embedding.shape)

        # Concatenate encoded_image and text_embedding
        concatenated_embedding = torch.cat((image_embedding, text_embedding), dim=1)

        # Run concatenated tensor through UNet decoder
        denoised_image = self.unet.forward_decoder(concatenated_embedding)

        return denoised_image


class UNet(nn.Module):
    """This UNet is the main workhorse of the backward denoising process."""

    def __init__(self):
        """Initialize the UNet model."""
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output shape: [1, 128, 360, 640]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=1)  # Output shape: [1, 3, 360, 640]
        )

    def forward_encoder(self, x):
        """Forward pass through the UNet encoder.
        
        Args:
            x: The input tensor.
        
        Returns:
            The output tensor after passing through the encoder.
        """
        x = self.encoder(x)
        return x

    def forward_decoder(self, x):
        """Forward pass through the UNet decoder.
        
        Args:
            x: The input tensor.
        
        Returns:
            The output tensor after passing through the decoder.
        """
        x = self.decoder(x)
        return x