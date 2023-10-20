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