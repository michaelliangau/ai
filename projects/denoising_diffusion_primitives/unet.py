import torch
from torch import nn
import math
from einops import rearrange
from einops.layers.torch import Rearrange
import constants
import torch.nn.functional as F
from typing import Tuple

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

    def __init__(self,
        dim: int = 128,
        cond_dim: int = 128,
        encoded_text_dim: int = 512,
        device: torch.device = torch.device("cpu"),
        max_text_len: int = 256,
        num_resnet_blocks: int = 1,
        layer_attns: bool = True,
        layer_cross_attns: bool = True):
        """Initialize the UNet model.
        
        Args:
            dim (int, optional): # of channels at the greatest spatial resolution of Unet.
                Defaults to 128.
            cond_dim (int, optional): Dimensionality of conditioning tensor. Defaults
                same as dim.
            encoded_text_dim (int, optional): Dimensionality of encoded text from T5.
                Defaults to 512.
            device (torch.device, optional): The device to use. Defaults to torch.device("cpu").
            max_text_len (int, optional): The maximum length of the text. Defaults to 256.
        """
        super(UNet, self).__init__()
        self.device = device
        self.max_text_len = max_text_len

        # Calculate variables
        time_cond_dim = dim * 4
        cond_dim = dim
        if dim is None:
            dim = cond_dim
        NUM_TIME_TOKENS = constants.NUM_TIME_TOKENS

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

        # Text conditioning
        self.text_to_cond = nn.Linear(encoded_text_dim, cond_dim)
        self.to_text_non_attn_cond = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, time_cond_dim),
            nn.SiLU(),
            nn.Linear(time_cond_dim, time_cond_dim)
        )
        self.norm_cond = nn.LayerNorm(cond_dim)

        # UNet layers
        # TODO: Writing this.
        # Downsampling and Upsampling modules of the Unet
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # Parameter lists for downsampling and upsampling trajectories
        resnet_groups = (8)
        layer_params = [num_resnet_blocks, resnet_groups, layer_attns, layer_cross_attns]
        reversed_layer_params = list(map(reversed, layer_params))

        # DOWNSAMPLING LAYERS

        # Keep track of skip connection channel depths for concatenation later
        skip_connect_dims = []

        # For each layer in the Unet
        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_cross_attn) in enumerate(
                zip(in_out, *layer_params)):

            is_last = ind == (num_resolutions - 1)

            layer_cond_dim = cond_dim if layer_cross_attn else None

            # Potentially use Transformer encoder at end of layer
            transformer_block_klass = TransformerBlock if layer_attn else Identity

            current_dim = dim_in

            # Whether to downsample at the beginning of the layer - cuts image spatial size-length
            pre_downsample = None
            if memory_efficient:
                pre_downsample = Downsample(dim_in, dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # Downsample at the end of the layer if not `pre_downsample`
            post_downsample = None
            if not memory_efficient:
                post_downsample = Downsample(current_dim, dim_out) if not is_last else Parallel(
                    nn.Conv2d(dim_in, dim_out, 3, padding=1), nn.Conv2d(dim_in, dim_out, 1))

            # Create the layer
            self.downs.append(nn.ModuleList([
                pre_downsample,
                # ResnetBlock that conditions, in addition to time, on the main tokens via cross attention.
                ResnetBlock(current_dim,
                            current_dim,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=groups),
                # Sequence of ResnetBlocks that condition only on time
                nn.ModuleList(
                    [
                        ResnetBlock(current_dim,
                                    current_dim,
                                    time_cond_dim=time_cond_dim,
                                    groups=groups
                                    )
                        for _ in range(layer_num_resnet_blocks)
                    ]
                ),
                # Transformer encoder for multi-headed self attention
                transformer_block_klass(dim=current_dim,
                                        heads=attn_heads,
                                        dim_head=ATTN_DIM_HEAD),
                post_downsample,
            ]))        

    def _generate_t_tokens(self, time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate time conditioning tensor and time tokens to be used throughout UNet.

        Time conditioning tensor will be embedded throughout the unet.
        Time tokens will be concatenated with text tokens to be used in cross attention layers.
        
        Args:
            time (torch.Tensor): The time tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The time conditioning tensor and time tokens.
        """
        time_hiddens = self.to_time_hiddens(time)
        time_tokens = self.to_time_tokens(time_hiddens) # TODO how is this used in cross attn?
        t = self.to_time_cond(time_hiddens)
        return t, time_tokens

    def _prob_mask_like(self, shape: Tuple[int], prob: float, device: torch.device) -> torch.Tensor:
        """
        For classifier free guidance. Creates a boolean mask for given input shape and probability of `True`.

        Args:
            shape (Tuple[int]): Shape of mask.
            prob (float): Probability of True. In interval [0., 1.].
            device (torch.device): Device to put the mask on. Should be the same as that of the tensor which it will be used on.

        Returns:
            torch.Tensor: The mask.
        """
        if prob == 1:
            return torch.ones(shape, device=device, dtype=torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device=device, dtype=torch.bool)
        else:
            return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

    def _text_condition(self, encoded_text: torch.Tensor, t: torch.Tensor, time_tokens: torch.Tensor, cond_drop_prob: float = 0, text_mask: torch.Tensor = None):
        """Generate the text conditioning tensor and text tokens to be used throughout UNet.

        Args:
            encoded_text (torch.Tensor): The encoded text from T5.
            t (torch.Tensor): The time conditioning tensor.
            time_tokens (torch.Tensor): The time tokens.
            cond_drop_prob (float, optional): The probability of dropping out text tokens.
                This is used to support classifier free guidance. We don't really need
                this. Defaults to 0.
            text_mask (torch.Tensor, optional): The text mask. Defaults to None.
        """
        text_tokens = None
        if encoded_text is not None:
            # Create text tokens
            text_tokens = self.text_to_cond(encoded_text)

            # Truncate max text len
            text_tokens = text_tokens[:, :self.max_text_len]

            # Pad text tokens to max text len
            text_tokens = F.pad(text_tokens, (0, 0, 0, self.max_text_len - text_tokens.shape[1]))
            mean_pooled_text_tokens = text_tokens.mean(dim=-2)

            # NOTE: I removed the classifier free guidance code from here.

            # Project tokens to to time condition shape.
            text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)

            # Add text to time condition
            t = t + text_hiddens
        
        # Concatenate time tokens and text tokens - main conditioning token.
        c = time_tokens if not text_tokens is not None else torch.cat((time_tokens, text_tokens), dim=-2)
        c = self.norm_cond(c)
        return t, c

    def forward(self, image: torch.Tensor, encoded_text: torch.Tensor, timestep: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UNet model.

        Args:
            image (torch.Tensor): The image tensor
            encoded_text (torch.Tensor): The text.
            timestep (torch.Tensor): The timestep.
            text_mask (torch.Tensor): The text mask.

        Returns:
            torch.Tensor: The output tensor after passing through the model.
        """
        # Time condition
        t, time_tokens = self._generate_t_tokens(timestep)

        # Text condition
        t, c = self._text_condition(encoded_text=encoded_text, t=t, time_tokens=time_tokens, text_mask=text_mask)


        # UNet
        import IPython; IPython.embed()
        pass