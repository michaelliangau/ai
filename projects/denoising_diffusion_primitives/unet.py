import torch
from torch import nn
import math
from einops import rearrange

class SinusoidalPosEmb(nn.Module):
    """Generates sinusoidal positional embedding tensor."""
    def __init__(self, dim: int):
        """Initialize the SinusoidalPosEmb class.

        Args:
            dim (int): The dimension of the positional embedding tensor.
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SinusoidalPosEmb class.

        Equations are introduced in the Attention Is All You Need paper.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The positional embedding tensor.
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class UNet(nn.Module):
    """This UNet is the main workhorse of the backward denoising process."""

    def __init__(self, *args, **kwargs):
        """Initialize the UNet model."""
        super(UNet, self).__init__()


        # Time conditioning
        self.to_time_hiddens = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_cond_dim),
            nn.SiLU()
        )
        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * NUM_TIME_TOKENS),
            Rearrange('b (r d) -> b r d', r=NUM_TIME_TOKENS)
        )
        self.to_time_cond = nn.Linear(time_cond_dim, time_cond_dim)



    def forward(self, x: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UNet model.

        Args:
            x (torch.Tensor): The input tensor, typically an image.
            text (torch.Tensor): The text.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        # Generate time_hiddens

        # Generate time_tokens, concatenated to text conditioning tokens used in cross attention layers.

        # Generate time conditioning tensor for each layer of UNet

        return 