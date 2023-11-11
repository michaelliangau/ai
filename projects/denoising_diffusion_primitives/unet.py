import torch
from torch import nn
import math
from einops import rearrange
from einops_exts.torch import EinopsToAndFrom
from einops.layers.torch import Rearrange
import constants
import torch.nn.functional as F
from typing import Tuple
import layers

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
        dim_mults: tuple = (1, 2, 4),
        channels: int = 3,
        cond_dim: int = 128,
        encoded_text_dim: int = 512,
        device: torch.device = torch.device("cpu"),
        max_text_len: int = 256,
        num_resnet_blocks: int = 1,
        attn_heads: int = 8,
        layer_attns: bool = True,
        layer_cross_attns: bool = True,
        attend_at_middle: bool = False):
        """Initializes the UNet model.
        
        Args:
            dim (int, optional): Number of channels at the highest spatial resolution of
                Unet. Defaults to 128.
            dim_mults (tuple, optional): Number of channels multiplier for each layer of
                the Unet. E.g. a 128 channel, 64x64 image put into a U-Net with
                :code:`dim_mults=(1, 2, 4)` will be shape

                - (128, 64, 64) in the first layer of the U-Net

                - (256, 32, 32) in the second layer of the U-net, and

                - (512, 16, 16) in the third layer of the U-Net  
            channels (int, optional): Number of channels in the input image. Defaults to 3.
            cond_dim (int, optional): Dimensionality of the conditioning tensor. Defaults
                to the same as dim.
            encoded_text_dim (int, optional): Dimensionality of the text encoded by T5.
                Defaults to 512.
            device (torch.device, optional): The device to run the model on. Defaults to
                torch.device("cpu").
            max_text_len (int, optional): Maximum length of the text input. Defaults to 256.
            num_resnet_blocks (int, optional): Number of ResNet blocks. Defaults to 1.
            attn_heads (int, optional): Number of attention heads. Defaults to 8.
            layer_attns (bool, optional): If True, applies attention to each layer.
                Defaults to True.
            layer_cross_attns (bool, optional): If True, applies cross-attention to each
                layer. Defaults to True.
            attend_at_middle (bool, optional): If True, applies attention to the bottleneck
                of the UNet. Defaults to False.
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
        # Initial convolution that brings input images to proper number of channels for the Unet
        self.init_conv = layers.CrossEmbedLayer(channels, dim_out=dim, kernel_sizes=(3, 7, 15), stride=1)

        # Downsampling layers
        self.downs = nn.ModuleList([])
        resnet_groups = (constants.RESNET_GROUPS,)
        num_resnet_blocks = (num_resnet_blocks,)
        layer_attns = (layer_attns,)
        layer_cross_attns = (layer_cross_attns,)
        layer_params = [num_resnet_blocks, resnet_groups, layer_attns, layer_cross_attns]
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Keep track of skip connection channel depths for concatenation later
        skip_connect_dims = []

        num_resolutions = len(in_out)
        # For each layer in the Unet
        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_cross_attn) in enumerate(
                zip(in_out, *layer_params)):

            is_last = ind == (num_resolutions - 1)

            layer_cond_dim = cond_dim if layer_cross_attn else None

            # Potentially use Transformer encoder at end of layer
            transformer_block_klass = layers.TransformerBlock if layer_attn else layers.Identity

            current_dim = dim_in

            # Whether to downsample at the beginning of the layer - cuts image spatial size-length
            pre_downsample = None
            skip_connect_dims.append(current_dim)

            # Downsample at the end of the layer if not `pre_downsample`
            post_downsample = None

            # Create the layer
            self.downs.append(nn.ModuleList([
                pre_downsample,
                # ResnetBlock that conditions, in addition to time, on the main tokens via cross attention.
                layers.ResnetBlock(current_dim,
                            current_dim,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=groups),
                # Sequence of ResnetBlocks that condition only on time
                nn.ModuleList(
                    [
                        layers.ResnetBlock(current_dim,
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
                                        dim_head=constants.ATTN_DIM_HEAD),
                post_downsample,
            ]))        

        # Middle layers
        mid_dim = dims[-1]

        # ResnetBlock that incorporates cross-attention conditioning on main tokens
        self.mid_block1 = layers.ResnetBlock(mid_dim, mid_dim, cond_dim=cond_dim, time_cond_dim=time_cond_dim,
                                      groups=resnet_groups[-1])

        # Optional residual self-attention
        self.mid_attn = EinopsToAndFrom('b c h w', 'b (h w) c',
                                        layers.Residual(layers.Attention(mid_dim, heads=attn_heads,
                                                           dim_head=constants.ATTN_DIM_HEAD))) if attend_at_middle else None

        # ResnetBlock that incorporates cross-attention conditioning on main tokens
        self.mid_block2 = layers.ResnetBlock(mid_dim, mid_dim, cond_dim=cond_dim, time_cond_dim=time_cond_dim,
                                      groups=resnet_groups[-1])

        

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
            image (torch.Tensor): The image tensor. Shape batch, channel, height, width.
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
        x = self.init_conv(image)

        # Downsampling trajectory

        # To store images for skip connections
        hiddens = []

        # For every layer in the downwards trajectory
        for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs:

            # Downsample before processing at this resolution if using efficient UNet
            if pre_downsample is not None:
                x = pre_downsample(x)

            # Initial block. Conditions on `c` via cross attention and conditions on `t` via scale-shift.
            x = init_block(x, t, c)

            # Series of residual blocks that are like `init_block` except they don't condition on `c`.
            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
                hiddens.append(x)

            # Transformer encoder
            x = attn_block(x)
            hiddens.append(x)

            # If not using efficient UNet, downsample after processing at this resolution
            if post_downsample is not None:
                x = post_downsample(x)
   
        # MIDDLE PASS

        # Pass through two ResnetBlocks that condition on `c` and `t`, with a possible residual Attention layer between.
        x = self.mid_block1(x, t, c)
        if self.mid_attn is not None:
            x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)        