from typing import List, Tuple, Union, Callable, Literal

import PIL
from tqdm import tqdm
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from einops import rearrange, repeat
from einops_exts import check_shape

from utils import cast_tuple, default, resize_image_to, normalize_neg_one_to_one, \
    unnormalize_zero_to_one, identity, exists, module_device, right_pad_dims_to, maybe, eval_decorator, null_context, extract
from t5 import t5_encode_text, get_encoded_dim
from diffusion_model import GaussianDiffusion

from einops import rearrange
from einops_exts import rearrange_many, repeat_many
from einops_exts.torch import EinopsToAndFrom
import math
from torch import nn, einsum


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
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod_prev)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1.)
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_mean_coef2 = sqrt_alphas_cumprod * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod_prev)
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_log_variance_clipped = utils.log(posterior_variance, eps=1e-20)

        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped)


    def q_sample(self, x_start, t, noise=None):
        """Apply gaussian diffusion to timestep t
        
        Args:
            x_start (torch.tensor): input image
            t (torch.tensor): timestep
            noise (torch.tensor, optional): noise to add to the image. Defaults to None.

        Returns:
            noised (torch.tensor): noised image
        """
        noise = default(noise, lambda: torch.randn_like(x_start)) # torch.randn_like returns a tensor with the same size as x_start, filled with random numbers from a normal distribution with mean 0 and variance 1.

        # This is a shortcut way of sampling the accumulated noised image at specific timestep t - https://roysubhradip.hashnode.dev/a-beginners-guide-to-diffusion-models-understanding-the-basics-and-beyond
        noised = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + # multiple every element of x_start by the t-th index of sqrt of the alphas_cumprod
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return noised

    def predict_start_from_noise(self, x_t, t, noise):
        """Tries to predict the starting image from the noised image, knowing the variance schedule""" # TODO is it the starting image? What about the U-Net?

        # Rearrange the q_sample equation to solve for x_0
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - 
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t, **kwargs):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + 
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape) # done for numerical stability

        return posterior_mean, posterior_variance, posterior_log_variance_clipped




class Attention(nn.Module):
    """
    Multi-headed attention
    """

    def __init__(
            self,
            dim: int,
            *,
            dim_head: int = 64,
            heads: int = 8,
            context_dim: int = None
    ):
        """
        :param dim: Input dimensionality.
        :param dim_head: Dimensionality for each attention head.
        :param heads: Number of attention heads.
        :param context_dim: Context dimensionality.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(
            context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x: torch.tensor, context: torch.tensor = None, mask: torch.tensor = None,
                attn_bias: torch.tensor = None) -> torch.tensor:

        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b 1 d', b=b)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # add text conditioning, if present

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim=-1)
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1, dtype=torch.float32)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Block(nn.Module):
    """
    Sub-block for :class:`.ResnetBlock`. GroupNorm/Identity, SiLU, and Conv2D in sequence, with the potential for
        scale-shift incorporation of timestep information.
    """

    def __init__(
            self,
            dim: int,
            dim_out: int,
            groups: int = 8,
            norm: bool = True
    ):
        """
        :param dim: Input number of channels.
        :param dim_out: Output number of channels.
        :param groups: Number of GroupNorm groups.
        :param norm: Whether to use GroupNorm or Identity.
        """
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x: torch.tensor, scale_shift: tuple[torch.tensor, torch.tensor] = None) -> torch.tensor:
        """
        Forward pass

        :param x: Input images.
        :param scale_shift: Tensors to use for scale-shift.
        """
        x = self.groupnorm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)


def ChanFeedForward(dim: int,
                    mult: int = 2) -> torch.nn.Sequential:  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    """
    MLP for :class:`.TransformerBlock`. Maps images to a multiple of the number of channels and then back with
        convolutions, with layernorms before each convolution a GELU between them.
    """
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv2d(dim, hidden_dim, 1, bias=False),
        nn.GELU(),
        ChanLayerNorm(hidden_dim),
        nn.Conv2d(hidden_dim, dim, 1, bias=False)
    )


class ChanLayerNorm(nn.Module):
    """
    LayerNorm for :class:`.ChanFeedForward`.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.tensor) -> torch.tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g


class CrossAttention(nn.Module):
    """
    Multi-headed cross attention.
    """

    def __init__(
            self,
            dim: int,
            *,
            context_dim: int = None,
            dim_head: int = 64,
            heads: int = 8,
            norm_context: bool = False
    ):
        """
        :param dim: Input dimensionality.
        :param context_dim: Context dimensionality.
        :param dim_head: Dimensionality for each attention head.
        :param heads: Number of attention heads.
        :param norm_context: Whether to LayerNorm the context.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x: torch.tensor, context: torch.tensor, mask: torch.tensor = None) -> torch.tensor:
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b h 1 d', h=self.heads, b=b)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossEmbedLayer(nn.Module):
    '''
    Module that performs cross embedding on an input image (essentially an Inception module) which maintains channel
        depth.

    E.g. If input a 64x64 image with 128 channels and use kernel_sizes = (3, 7, 15) and stride=1, then 3 convolutions
        will be performed:

        1: 64 filters, (3x3) kernel, stride=(1x1), padding=(1x1) -> 64x64 output
        2: 32 filters, (7x7) kernel, stride=(1x1), padding=(3x3) -> 64x64 output
        3: 32 filters, (15x15) kernel, stride=(1x1), padding=(7x7) -> 64x64 output

        Concatenate them for a resulting 64x64 image with 128 output channels
    '''

    def __init__(
            self,
            dim_in: int,
            kernel_sizes: tuple[int, ...],
            dim_out: int = None,
            stride: int = 2
    ):
        """
        :param dim_in: Number of channels in the input image.
        :param kernel_sizes: Tuple of kernel sizes to use for convolutions.
        :param dim_out: Number of channels in output image. Defaults to `dim_in`.
        :param stride: Stride of convolutions.
        """
        super().__init__()
        # Ensures stride and all kernels are either all odd or all even
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])

        # Set output dimensionality to be same as input if not provided
        dim_out = default(dim_out, dim_in)

        # Sort the kernels by size and determine number of kernels
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # Determine number of filters for each kernel. They will sum to dim_out and be descending with kernel size
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        # Create the convolution objects
        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2))

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Perform each convolution and then concatenate the results along the channel dim.
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)


def Downsample(dim: int, dim_out: int = None) -> torch.nn.Conv2d:
    """
    Return a convolution layer that cuts the spatial dimensions of an image in half and potentially modifies the
        number of channels

    :param dim: Input dimensionality of the image
    :param dim_out: Output dimensionality of the image. Defaults to `dim`.
    :return: Convolution layer.
    """

    dim_out = default(dim_out, dim)
    return nn.Conv2d(dim, dim_out, kernel_size=4, stride=2, padding=1)


class Identity(nn.Module):
    """
    Identity module - forward pass returns input.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.tensor, *args, **kwargs) -> torch.tensor:
        return x


class LayerNorm(nn.Module):
    """
    LayerNorm
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x: torch.tensor) -> torch.tensor:
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class Parallel(nn.Module):
    """
    Passes input through parallel functions and then sums the result.
    """
    def __init__(self, *fns: tuple[Callable, ...]):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x: torch.tensor) -> torch.tensor:
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)


class Residual(nn.Module):
    """
    Residual module. Passes input through a function and then adds the result to the input.
    """
    def __init__(self, fn: callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.tensor, **kwargs) -> torch.tensor:
        return self.fn(x, **kwargs) + x


class ResnetBlock(nn.Module):
    """
    ResNet Block.
    """
    def __init__(
            self,
            dim: int,
            dim_out: int,
            *,
            cond_dim: int = None,
            time_cond_dim: int = None,
            groups: int = 8,
    ):
        """
        :param dim: Number of channels in the input.
        :param dim_out: Number of channels in the output.
        :param cond_dim: Dimension of the conditioning tokens on which to perform cross attention with the input.
        :param time_cond_dim: Dimension of the time conditioning tensor.
        :param groups: Number of groups to use in the GroupNorms. See :class:`.Block`.
        """
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None
        if exists(cond_dim):
            self.cross_attn = EinopsToAndFrom(
                'b c h w',
                'b (h w) c',
                CrossAttention(
                    dim=dim_out,
                    context_dim=cond_dim
                )
            )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()

    def forward(self, x: torch.tensor, time_emb: torch.tensor = None, cond: torch.tensor = None) -> torch.tensor:
        """
        :param x: Input image. Shape (b, c, s, s).
        :param time_emb: Time conditioning tensor. Shape (b, c2).
        :param cond: Main conditioning tensor. Shape (b, c3).
        :return: Output image. Shape (b, c, s, s)
        """

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)

        if exists(self.cross_attn):
            assert exists(cond)
            h = self.cross_attn(h, context=cond) + h

        h = self.block2(h, scale_shift=scale_shift)

        return h + self.res_conv(x)


class SinusoidalPosEmb(nn.Module):
    '''
    Generates sinusoidal positional embedding tensor. In this case, position corresponds to time. For more information
        on sinusoidal embeddings, see ["Positional Encoding - Additional Details"](https://www.assemblyai.com/blog/how-imagen-actually-works/#timestep-conditioning).
    '''

    def __init__(self, dim: int):
        """
        :param dim: Dimensionality of the embedding space
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        :param x: Tensor of positions (i.e. times) to generate embeddings for.
        :return: T x D tensor where T is the number of positions/times and D is the dimensionality of the embedding
            space
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TransformerBlock(nn.Module):
    """
    Transformer encoder block. Responsible for applying attention at the end of a chain of :class:`.ResnetBlock`s at
        each layer in the U-Met.
    """
    def __init__(
            self,
            dim: int,
            *,
            heads: int = 8,
            dim_head: int = 32,
            ff_mult: int = 2,
            context_dim: int = None
    ):
        """

        :param dim: Number of channels in the input.
        :param heads: Number of attention heads for multi-headed :class:`.Attention`.
        :param dim_head: Dimensionality for each attention head in multi-headed :class:`.Attention`.
        :param ff_mult: Channel depth multiplier for the :class:`.ChanFeedForward` MLP applied after multi-headed
            attention.
        :param context_dim: Dimensionality of the context.
        """
        super().__init__()
        self.attn = EinopsToAndFrom('b c h w', 'b (h w) c',
                                    Attention(dim=dim, heads=heads, dim_head=dim_head, context_dim=context_dim))
        self.ff = ChanFeedForward(dim=dim, mult=ff_mult)

    def forward(self, x: torch.tensor, context: torch.tensor = None) -> torch.tensor:
        x = self.attn(x, context=context) + x
        x = self.ff(x) + x
        return x


def Upsample(dim: int, dim_out: int = None) -> torch.nn.Sequential:
    """
    Returns Sequential module that upsamples to twice the spatial width with an [Upsample](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)
        followed by a [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).

    :param dim: Number of channels in the input.
    :param dim_out: Number of channels in the output. Defaults to `dim`.
    """
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, dim_out, 3, padding=1)
    )




class Imagen(nn.Module):
    """
    Minimal `Imagen <https://imagen.research.google/>`_ implementation.
    """

    def __init__(
            self,
            unets: Union[Unet, List[Unet], Tuple[Unet, ...]],
            *,
            image_sizes: Union[int, List[int], Tuple[int, ...]],
            text_encoder_name: str = 't5_small',
            text_embed_dim: int = None,
            channels: int = 3,
            timesteps: Union[int, List[int], Tuple[int, ...]] = 1000,
            cond_drop_prob: float = 0.1,
            loss_type: Literal["l1", "l2", "huber"] = 'l2',
            lowres_sample_noise_level: float = 0.2,
            auto_normalize_img: bool = True,
            dynamic_thresholding_percentile: float = 0.9,
            only_train_unet_number: int = None
    ):
        """
        :param unets: :class:`Unet(s) <.minimagen.Unet.Unet>`, where the first element in the argument is the base
            model (image generator), and the following Unets are super-resolution models (if provided).
        :param image_sizes: The side length of the images input to each unet. Same length as :code:`unets`.
        :param text_encoder_name: The name of the T5 text encoder to use. See :func:`.minimagen.t5.t5_encode_text`
        :param text_embed_dim: Embedding dimension of text encoder. Do not set if using a built-in T5 from the list
            in :func:`.minimagen.t5.t5_encode_text` (will be set automatically).
        :param channels: Number of channels in images.
        :param timesteps: Number of timesteps in the `Diffusion Process <https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/>`_.
            Either one value used for every Unet in Imagen, or a list/tuple of values, one for each Unet in Imagen.
        :param cond_drop_prob: Probability of dropping for `classifier-free guidance <https://www.assemblyai.com/blog/how-imagen-actually-works/#classifier-free-guidance>`_
        :param loss_type: Type of loss function to use. L1 (:code:`l1`), L2 (:code:`l2`), or Huber (:code:`huber`).
        :param lowres_sample_noise_level: Noise scale for `low-res conditioning augmentation <https://www.assemblyai.com/blog/how-imagen-actually-works/#robust-cascaded-diffusion-models>`_.
            fixed to a level in the range [0.1, 0.3] in the original Imagen implementation.
        :param auto_normalize_img: Whether to auto normalize images to the range [-1., 1.]. Leave :code:`True` if
            feeding in images in the standard range [0., 1.], or turn :code:`False` if you will preprocess to [-1., 1.]
            before feeding in.
        :param dynamic_thresholding_percentile: Percentile value at which to activate `dynamic thresholding <https://www.assemblyai.com/blog/how-imagen-actually-works/#large-guidance-weight-samplers>`_
        :param only_train_unet_number: Specify number of unet in :code:`Unets` to train if only one.
        """
        super().__init__()

        # Set loss
        self.loss_type = loss_type
        self.loss_fn = self._set_loss_fn(loss_type)

        self.channels = channels

        unets = cast_tuple(unets)
        num_unets = len(unets)

        # Create noise schedulers for each UNet
        self.noise_schedulers = self._make_noise_schedulers(num_unets, timesteps)

        # Lowres augmentation noise schedule
        self.lowres_noise_schedule = GaussianDiffusion(timesteps=timesteps)

        # Text encoder params
        self.text_encoder_name = text_encoder_name
        self.text_embed_dim = default(text_embed_dim, lambda: get_encoded_dim(text_encoder_name))

        # Keep track of which unet is being trained at the moment
        self.unet_being_trained_index = -1

        self.only_train_unet_number = only_train_unet_number

        # Cast the relevant hyperparameters to the input Unets, ensuring that the first Unet does not condition on
        #   lowres images (base unet) while the remaining ones do (super-res unets)
        self.unets = nn.ModuleList([])
        for ind, one_unet in enumerate(unets):
            assert isinstance(one_unet, Unet)
            is_first = ind == 0

            one_unet = one_unet._cast_model_parameters(
                lowres_cond=not is_first,
                text_embed_dim=self.text_embed_dim,
                channels=self.channels,
                channels_out=self.channels,
            )

            self.unets.append(one_unet)

        # Uet image sizes
        self.image_sizes = cast_tuple(image_sizes)
        assert num_unets == len(
            image_sizes), f'you did not supply the correct number of u-nets ({len(self.unets)}) for resolutions' \
                          f' {image_sizes}'

        self.sample_channels = cast_tuple(self.channels, num_unets)

        self.lowres_sample_noise_level = lowres_sample_noise_level

        # Classifier free guidance
        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.

        # Normalize and un-normalize image functions
        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity
        self.input_image_range = (0. if auto_normalize_img else -1., 1.)

        # Dynamic thresholding
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        # one temp parameter for keeping track of device
        self.register_buffer('_temp', torch.tensor([0.]), persistent=False)

        # default to device of unets passed in
        self.to(next(self.unets.parameters()).device)

    @property
    def device(self) -> torch.device:
        # Returns device of Imagen instance (not writeable)
        return self._temp.device

    @staticmethod
    def _set_loss_fn(loss_type: str) -> Callable:
        """
        Helper function to set the loss of an Imagen instance

        :param loss_type: Type of loss to use. Either 'l1', 'l2', or 'huber'
        :return: loss function.
        """
        # loss
        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()
        return loss_fn

    @staticmethod
    def _make_noise_schedulers(
            num_unets: int,
            timesteps: Union[int, List[int], Tuple[int, ...]]
    ) -> Tuple[GaussianDiffusion, ...]:
        """
        Makes :class:`noise schedulers minimal_imagen.diffusion_model.GaussianDiffusion`.

        :param num_unets: Number of Unets to make schedulers for.
        :param timesteps: Timesteps in the diffusion process for the schedulers.
        :return: Noise schedulers
        """
        # determine noise schedules per unet
        timesteps = cast_tuple(timesteps, num_unets)

        # construct noise schedulers
        noise_schedulers = nn.ModuleList([])
        for timestep in timesteps:
            noise_scheduler = GaussianDiffusion(timesteps=timestep)
            noise_schedulers.append(noise_scheduler)

        return noise_schedulers

    def _get_unet(self, unet_number: int) -> Unet:
        """
         Gets the unet that is to be trained and places it on the same device as the Imagen instance, while placing all
            other Unets on the CPU.

        :param unet_number: The number of the Unet in `self.unets` to get.
        :return: The selected unet.
        """
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.unets]
            delattr(self, 'unets')
            self.unets = unets_list

        # If gotten unet different than one listed as being trained, pl
        if index != self.unet_being_trained_index:
            for unet_index, unet in enumerate(self.unets):
                unet.to(self.device if unet_index == index else 'cpu')

        # Update relevant attribute
        self.unet_being_trained_index = index
        return self.unets[index]

    def _reset_unets_all_one_device(self, device: torch.device = None):
        """
        Creates a ModuleList out of all Unets in Imagen instance and places it on one device. Device either specified
            or defaults to Imagen instance device.

        :param device: Device on which to place the Unets
        :return: None
        """
        # Creates ModuleList out of the Unets and places on the relevant device.
        device = default(device, self.device)
        self.unets = nn.ModuleList([*self.unets])
        self.unets.to(device)

        # Resets relevant attribute to specify that no Unet is being trained at the moment
        self.unet_being_trained_index = -1

    def state_dict(self, *args, **kwargs):
        """
        Overrides `state_dict <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict>`_ to place all Unets in Imagen instance on one device when called.
        """
        self._reset_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """
        Overrides `load_state_dict <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict>`_ to place all Unets in Imagen instance on one device when called.
        """
        self._reset_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    @contextmanager
    def _one_unet_in_gpu(self, unet_number: int = None, unet: Unet = None):
        """
        Context manager for placing one unet on the GPU. Ensures that all Unets are placed back onto their original
            devices upon closing.

        :param unet_number: Number of the unet to place on the GPU
        :param unet: Unet object to place on the GPU.
        :return:
        """
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        # Store which device each UNet is on, place them all on CPU except the specified one
        devices = [module_device(unet) for unet in self.unets]
        self.unets.cpu()
        unet.to(self.device)

        yield

        # Restore all UNets back to their original devices
        for unet, device in zip(self.unets, devices):
            unet.to(device)

    def _p_mean_variance(self,
                         unet: Unet,
                         x: torch.tensor,
                         t: torch.tensor,
                         *,
                         noise_scheduler: GaussianDiffusion,
                         text_embeds: torch.tensor = None,
                         text_mask: torch.tensor = None,
                         lowres_cond_img: torch.tensor = None,
                         lowres_noise_times: torch.tensor = None,
                         cond_scale: float = 1.,
                         model_output: torch.tensor = None) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Predicts noise component of `x` with `unet`, and then returns the corresponding forward process posterior
            parameters given the predictions.

        .. image:: minimal_imagen/minimagen/images/q_posterior.png
        .. image:: minimal_imagen/minimagen/images/q_posterior_mean.png
        .. image:: minimal_imagen/minimagen/images/posterior_variance.png


        :param unet: Unet that predicts either the noise component of noised images
        :param x: Images to operate on. Shape (b, c, s, s)
        :param t: Timesteps of images. Shape (b,)
        :return: tuple (
            posterior mean (shape (b, c, h, w)),
            posterior variance (shape (b, 1, 1, 1)),
            posterior log variance clipped (shape (b, 1, 1, 1))
            )
        """
        assert not (
                cond_scale != 1. and not self.can_classifier_guidance), 'imagen was not trained with conditional' \
                                                                        ' dropout, and thus one cannot use classifier' \
                                                                        ' free guidance (cond_scale anything other' \
                                                                        ' than 1)'

        # Get the prediction from the base unet
        pred = default(model_output, lambda: unet.forward_with_cond_scale(x,
                                                                          t,
                                                                          text_embeds=text_embeds,
                                                                          text_mask=text_mask,
                                                                          cond_scale=cond_scale,
                                                                          lowres_cond_img=lowres_cond_img,
                                                                          lowres_noise_times=lowres_noise_times))

        # Calculate the starting images from the noise
        x_start = noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        # DYNAMIC THRESHOLDING
        #   https://www.assemblyai.com/blog/how-imagen-actually-works/#large-guidance-weight-samplers

        # Calculate threshold for each image
        s = torch.quantile(
            rearrange(x_start, 'b ... -> b (...)').abs(),
            self.dynamic_thresholding_percentile,
            dim=-1
        )

        # If threshold is less than 1, simply clamp values to [-1., 1.]
        s.clamp_(min=1.)
        s = right_pad_dims_to(x_start, s)
        # Clamp to +/- s and divide by s to bring values back to range [-1., 1.]
        x_start = x_start.clamp(-s, s) / s

        # Return the forward process posterior parameters given the predicted x_start
        return noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)

    @torch.no_grad()
    def _p_sample(self,
                  unet: Unet,
                  x: torch.tensor,
                  t: torch.tensor,
                  *,
                  noise_scheduler: GaussianDiffusion,
                  text_embeds: torch.tensor = None,
                  text_mask: torch.tensor = None,
                  lowres_cond_img: torch.tensor = None,
                  lowres_noise_times: torch.tensor = None,
                  cond_scale: float = 1.
                  ) -> torch.tensor:
        """
        Given a denoising Unet and noisy images, takes one step back in time in the diffusion model. I.e. given
        a noisy image x_t, `_p_sample` samples from q(x_{t-1}|x_t) to get a slightly denoised image x_{t-1}.

        .. image:: minimal_imagen/minimagen/images/x_tm1.png

        :param unet: Unet for denoising.
        :param x: Noisy images. Shape (b, c, s, s)
        :param t: Noisy image timesteps. Shape (b,)
        :return: Slightly denoised images. Shape (b, c, s, s)
        """
        b, *_, device = *x.shape, x.device
        # Calculate sampling distribution parameters
        model_mean, _, model_log_variance = self._p_mean_variance(unet, x=x, t=t,
                                                                  noise_scheduler=noise_scheduler,
                                                                  text_embeds=text_embeds, text_mask=text_mask,
                                                                  cond_scale=cond_scale,
                                                                  lowres_cond_img=lowres_cond_img,
                                                                  lowres_noise_times=lowres_noise_times)
        # Noise for sampling
        noise = torch.randn_like(x)

        # Don't denoise when t == 0
        is_last_sampling_timestep = (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        # Calculate sample from posterior distribution. See
        #   https://github.com/oconnoob/minimal_imagen/blob/minimal/images/x_tm1.png
        #  equivalent to mean + sqrt(variance) * epsilon but calculate this way to be more numerically stable
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def _p_sample_loop(self,
                       unet: Unet,
                       shape: tuple,
                       *,
                       noise_scheduler: GaussianDiffusion,
                       text_embeds: torch.tensor = None,
                       text_mask: torch.tensor = None,
                       lowres_cond_img: torch.tensor = None,
                       lowres_noise_times: torch.tensor = None,
                       cond_scale: float = 1.
                       ):
        """
        Given a Unet, iteratively generates a sample via [reverse-diffusion](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/#diffusion-modelsintroduction).

        :param unet: The Unet to use for reverse-diffusion.
        :param shape: The shape of the image(s) to generate. (b, c, s, s).
        """
        device = self.device

        # Normalize conditioning images if needed
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # Get reverse-diffusion timesteps (i.e. (T, T-1, T-2, ..., 2, 1, 0) )
        batch = shape[0]
        timesteps = noise_scheduler._get_sampling_timesteps(batch, device=device)

        # Generate starting "noised images"
        img = torch.randn(shape, device=device)

        # For each timestep, take one step back in time (i.e. one step forward in the reverse-diffusion process),
        # slightly denoising the images at each step. Supply conditioning information to direct the process.
        for times in tqdm(timesteps, desc='sampling loop time step', total=len(timesteps)):
            img = self._p_sample(
                unet,
                img,
                times,
                text_embeds=text_embeds,
                text_mask=text_mask,
                cond_scale=cond_scale,
                lowres_cond_img=lowres_cond_img,
                lowres_noise_times=lowres_noise_times,
                noise_scheduler=noise_scheduler,
            )

        # Clamp images to the allowed range and un \-normalize to [0., 1.] if needed.
        img.clamp_(-1., 1.)
        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img

    @torch.no_grad()
    @eval_decorator
    def sample(
            self,
            texts: List[str] = None,
            text_masks: torch.tensor = None,
            text_embeds: torch.tensor = None,
            cond_scale: float = 1.,
            lowres_sample_noise_level: float = None,
            return_pil_images: bool = False,
            device: torch.device = None,
    ) -> Union[torch.tensor, PIL.Image.Image]:
        """
        Generate images with Imagen.

        :param texts: Text prompts to generate images for.
        :param text_masks: Text encoder mask. Used if :code:`texts` is not supplied.
        :param text_embeds: Text encoder embeddings. Used if :code:`texts` is not supplied.
        :param cond_scale: Conditioning scale for `classifier-free guidance <https://www.assemblyai.com/blog/how-imagen-actually-works/#classifier-free-guidance>`_.
        :param lowres_sample_noise_level: Noise scale for `low-res noise conditioning augmentation <https://www.assemblyai.com/blog/how-imagen-actually-works/#robust-cascaded-diffusion-models>`_.
        :param return_pil_images: Whether to return output as PIL image (rather than a :code:`torch.tensor`).
        :param device: Device on which to operate. Defaults to Imagen instance's device.
        :return: Tensor of images, shape (b, c, s, s).
        """
        device = default(device, self.device)
        self._reset_unets_all_one_device(device=device)

        # Calculate text embeddings/mask if not passed in
        if exists(texts) and not exists(text_embeds):
            text_embeds, text_masks = t5_encode_text(texts, name=self.text_encoder_name)
            text_embeds, text_masks = map(lambda t: t.to(device), (text_embeds, text_masks))

        assert exists(text_embeds), 'text or text encodings must be passed into Imagen'
        assert not (exists(text_embeds) and text_embeds.shape[
            -1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        batch_size = text_embeds.shape[0]

        outputs = None

        is_cuda = next(self.parameters()).is_cuda
        device = next(self.parameters()).device

        lowres_sample_noise_level = default(lowres_sample_noise_level, self.lowres_sample_noise_level)

        # For each unet, sample with the appropriate conditioning
        for unet_number, unet, channel, image_size, noise_scheduler in tqdm(
                zip(range(1, len(self.unets) + 1), self.unets, self.sample_channels, self.image_sizes,
                    self.noise_schedulers)):

            # If GPU is available, place the Unet currently being sampled from on the GPU
            context = self._one_unet_in_gpu(unet=unet) if is_cuda else null_context()

            with context:
                lowres_cond_img = lowres_noise_times = None

                # If on a super-resolution model, noise the previously generated images for conditioning
                if unet.lowres_cond:
                    lowres_noise_times = self.lowres_noise_schedule._get_times(batch_size, lowres_sample_noise_level,
                                                                              device=device)
                    lowres_cond_img = resize_image_to(img, image_size, pad_mode='reflect')
                    lowres_cond_img = self.lowres_noise_schedule.q_sample(x_start=lowres_cond_img,
                                                                          t=lowres_noise_times,
                                                                          noise=torch.randn_like(lowres_cond_img))

                shape = (batch_size, self.channels, image_size, image_size)

                # Generate images with the current unet
                img = self._p_sample_loop(
                    unet,
                    shape,
                    text_embeds=text_embeds,
                    text_mask=text_masks,
                    cond_scale=cond_scale,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_times=lowres_noise_times,
                    noise_scheduler=noise_scheduler,
                )

                # Output the image if at the end of the super-resolution chain
                outputs = img if unet_number == len(self.unets) else None

        # Return torch tensors or PIL Images
        if not return_pil_images:
            return outputs

        pil_images = list(map(T.ToPILImage(), img.unbind(dim=0)))

        return pil_images

    def _p_losses(self,
                  unet: Unet,
                  x_start: torch.tensor,
                  times: torch.tensor,
                  *,
                  noise_scheduler: GaussianDiffusion,
                  lowres_cond_img: torch.tensor = None,
                  lowres_aug_times: torch.tensor = None,
                  text_embeds: torch.tensor = None,
                  text_mask: torch.tensor = None,
                  noise: torch.tensor = None,
                  ) -> torch.tensor:
        """
        Performs the forward diffusion process to corrupt training images (`x_start`), performs the reverse diffusion
            process using `unet` to get predictions, and then calculates the loss from these predictions.

        Loss is calculated on a per-pixel basis according to `self.loss_fn`, and then averaged across channels/spatial
            dimensions for each batch. The average loss over the batch is returned.

        :param unet: Unet to be trained.
        :param x_start: Training images. Shape (b, c, l, l).
        :param times: Timestep for each image in the batch. Shape (b,).
        :param noise_scheduler: Noise scheduler used for forward diffusion noising process.
        :param lowres_cond_img: Low-resolution version of images to condition on for super-resolution models.
            Shape (b, c, s, s)
        :param lowres_aug_times: Timesteps for [low-resolution noise augmentation](https://www.assemblyai.com/blog/how-imagen-actually-works/#robust-cascaded-diffusion-models)
        :param text_embeds: Text embeddings of conditioning text.
        :param text_mask: Text mask for text embeddings.
        :param noise: Noise to use for the forward process. If not provided, defaults to Gaussian.
        :return: Loss.
        """

        # If no noise is provided, randomly sample
        noise = default(noise, lambda: torch.randn_like(x_start))

        # normalize x_start to [-1, 1] and so too lowres_cond_img if it exists
        x_start = self.normalize_img(x_start)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # get x_t (i.e. noise the inputs)
        x_noisy = noise_scheduler.q_sample(x_start=x_start, t=times, noise=noise)

        # Also noise the lowres conditioning image
        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_aug_times = default(lowres_aug_times, times)
            lowres_cond_img_noisy = self.lowres_noise_schedule.q_sample(x_start=lowres_cond_img, t=lowres_aug_times,
                                                                        noise=torch.randn_like(lowres_cond_img))

        # Predict the noise component of the noised images
        pred = unet.forward(
            x_noisy,
            times,
            text_embeds=text_embeds,
            text_mask=text_mask,
            lowres_noise_times=lowres_aug_times,
            lowres_cond_img=lowres_cond_img_noisy,
            cond_drop_prob=self.cond_drop_prob,
        )

        # Return loss between prediction and ground truth
        return self.loss_fn(pred, noise)

    def forward(
            self,
            images,
            texts: List[str] = None,
            text_embeds: torch.tensor = None,
            text_masks: torch.tensor = None,
            unet_number: int = None,
    ):
        """
        Imagen forward pass. Noises images and then calculates loss from U-Net noise prediction.

        :param images: Images to operate on. Shape (b, c, s, s).
        :param texts: Text captions to condition on. List of length b.
        :param text_embeds: Text embeddings to condition on. Used if :code:`texts` is not passed in.
        :param text_masks: Text embedding mask. Used if :code:`texts` is not passed in.
        :param unet_number: Which number unet to train if there are multiple.
        :return: Loss.
        """
        assert not (len(self.unets) > 1 and not exists(unet_number)), \
            f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, ' \
            f'if you are training cascading DDPM (multiple unets)'

        unet_number = default(unet_number, 1)
        assert not exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, \
            f'you can only train on unet #{self.only_train_unet_number}'

        # Get the proper models, objective, etc. for the unet to be trained.
        unet_index = unet_number - 1
        unet = self._get_unet(unet_number)

        noise_scheduler = self.noise_schedulers[unet_index]
        target_image_size = self.image_sizes[unet_index]
        prev_image_size = self.image_sizes[unet_index - 1] if unet_index > 0 else None
        b, c, h, w, device, = *images.shape, images.device

        # Make sure images have proper number of dimensions and channels.
        check_shape(images, 'b c h w', c=self.channels)
        assert h >= target_image_size and w >= target_image_size

        # Randomly sample a timestep value for each image in the batch.
        times = noise_scheduler._sample_random_times(b, device=device)

        # If text conditioning info supplied as text rather than embeddings, calculate the embeddings/mask
        if exists(texts) and not exists(text_embeds):
            assert len(texts) == len(images), \
                'number of text captions does not match up with the number of images given'

            text_embeds, text_masks = t5_encode_text(texts, name=self.text_encoder_name)
            text_embeds, text_masks = map(lambda t: t.to(images.device), (text_embeds, text_masks))

        # Make sure embeddings are not supplied if not conditioning on text and vice versa
        assert exists(text_embeds), \
            'text or text encodings must be passed into decoder'

        # Ensure text embeddings are right dimensionality
        assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), \
            f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        # Create low-res conditioning information if a super-res model
        lowres_cond_img = lowres_aug_times = None
        if exists(prev_image_size):
            lowres_cond_img = resize_image_to(images, prev_image_size, clamp_range=self.input_image_range,
                                              pad_mode='reflect')
            lowres_cond_img = resize_image_to(lowres_cond_img, target_image_size, clamp_range=self.input_image_range,
                                              pad_mode='reflect')

            lowres_aug_time = self.lowres_noise_schedule._sample_random_times(1, device=device)
            lowres_aug_times = repeat(lowres_aug_time, '1 -> b', b=b)

        # Resize images to current unet size
        images = resize_image_to(images, target_image_size)

        # Calculate and return the loss
        return self._p_losses(unet, images, times, text_embeds=text_embeds, text_mask=text_masks,
                              noise_scheduler=noise_scheduler, lowres_cond_img=lowres_cond_img,
                              lowres_aug_times=lowres_aug_times)