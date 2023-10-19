import torch
from torch import nn
import numpy as np

class ForwardProcess():
    """Adds noise to an image in a forward process."""
    def __init__(self, num_timesteps: int = 100, initial_beta: float = 0.2, decay_rate: float = 0.98, torch_device: torch.device = torch.device("cuda")) -> None:
        """Initialize the forward process.

        Args:
            num_timesteps: Number of timesteps in the diffusion process.
            initial_beta: Initial beta value. This is a hyperparameter that we tune.
                It represents what is the standard deviation of the noise that we add to
                the images at the first timestep (which has maximum noise).
            decay_rate: Decay rate for each subsequent beta.
        """
        self.betas = self.generate_betas(num_timesteps, initial_beta, decay_rate).to(torch_device)
    
    def generate_betas(self, num_timesteps: int, initial_beta: float, decay_rate: float) -> torch.Tensor:
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
        # Create an array of indices
        indices = np.arange(num_timesteps)
        # Compute the betas in a vectorized manner
        betas = initial_beta * (decay_rate ** indices)
        # Convert to a torch tensor and return
        return torch.tensor(betas, dtype=torch.float32)
    
    def sample(self, image: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Sample from the forward process at a specific timestep.
        
        Args:
            image: The image to noise.
            timestep: The timestep to sample at.
        """
        noise_std = torch.sqrt(self.betas[timestep])
        noise = torch.randn_like(image) * noise_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        noised_image = image + noise
        return noised_image

class BackwardProcess():
    """Generates an image from a noised image in a backward process."""
    def __init__(self, model, torch_device=torch.device("cuda")) -> None:
        """
        Initialize the backward process.

        Args:
            model: The model to be used in the backward process.
        """
        self.unet = model
        self.downsample_text_embedding_layer = nn.Linear(512, 128).to(torch_device)
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
        # Push image through UNet encoder
        image_embedding = self.unet.forward_encoder(image)

        # Expand text embedding into same dim as encoded_image
        text_embedding = self.downsample_text_embedding_layer(text)
        text_embedding = text_embedding.unsqueeze(-1).unsqueeze(-1)
        text_embedding = text_embedding.expand(image_embedding.shape)

        # Concatenate encoded_image and text_embedding
        concatenated_embedding = torch.cat((image_embedding, text_embedding), dim=1).to(self.torch_device)

        # Run concatenated tensor through UNet decoder
        predicted_noise = self.unet.forward_decoder(concatenated_embedding)

        return predicted_noise


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