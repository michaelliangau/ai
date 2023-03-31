# Our imports
import utils

# Native imports
import math

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import IPython
from tqdm import tqdm
import einops


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
        noise = utils.default(noise, lambda: torch.randn_like(x_start)) # torch.randn_like returns a tensor with the same size as x_start, filled with random numbers from a normal distribution with mean 0 and variance 1.

        # This is a shortcut way of sampling the accumulated noised image at specific timestep t - https://roysubhradip.hashnode.dev/a-beginners-guide-to-diffusion-models-understanding-the-basics-and-beyond
        noised = (
            utils.extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + # multiple every element of x_start by the t-th index of sqrt of the alphas_cumprod
            utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return noised

    def predict_start_from_noise(self, x_t, t, noise):
        """Tries to predict the starting image from the noised image, knowing the variance schedule""" # TODO is it the starting image? What about the U-Net?

        # Rearrange the q_sample equation to solve for x_0
        return (
            utils.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - 
            utils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t, **kwargs):
        posterior_mean = (
            utils.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + 
            utils.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = utils.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = utils.extract(self.posterior_log_variance_clipped, t, x_t.shape) # done for numerical stability

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

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
        emb = einops.rearrange(x, 'i -> i 1') * einops.rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
    
class UNet(nn.Module):
    """U-Net architecture for image denoising"""
    def __init__(self, *args, **kwargs):
        self.to_time_hiddens = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_cond_dim),
            nn.SiLU()
        )
        self.to_time_cond = nn.Linear(time_cond_dim, time_cond_dim)
        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * NUM_TIME_TOKENS),
            einops.layers.torch.Rearrange('b (r d) -> b r d', r=NUM_TIME_TOKENS)
        )
        self.text_to_cond = nn.Linear(self.text_embed_dim, cond_dim)

    def _text_condition(
                self,
                text_embeds: torch.tensor,
                batch_size: int,
                cond_drop_prob: float,
                device: torch.device,
                text_mask: torch.tensor,
                t: torch.tensor,
                time_tokens: torch.tensor
        ):
            '''
            Condition on text.
            :param text_embeds: Text embedding from T5 encoder. Shape (b, mw, ed), where
                :code:`b` is the batch size,
                :code:`mw` is the maximum number of words in a caption in the batch, and
                :code:`ed` is the T5 encoding dimension.
            :param batch_size: Size of the batch/number of captions
            :param cond_drop_prob: Probability of conditional dropout for `classifier-free guidance <https://www.assemblyai.com/blog/how-imagen-actually-works/#classifier-free-guidance>`_
            :param device: Device to use.
            :param text_mask: Text mask for text embeddings. Shape (b, minimagen.t5.MAX_LENGTH)
            :param t: Time conditioning tensor.
            :param time_tokens: Time conditioning tokens.
            :return: tuple(t, c)
                :code:`t`: Time conditioning tensor
                :code:`c`: Main conditioning tokens
            '''

            text_tokens = None
            if utils.exists(text_embeds):

                # Project the text embeddings to the conditioning dimension `cond_dim`.
                text_tokens = self.text_to_cond(text_embeds)

                # Truncate the tokens to have the maximum number of allotted words.
                text_tokens = text_tokens[:, :self.max_text_len]

                # Pad the text tokens up to self.max_text_len if needed
                text_tokens_len = text_tokens.shape[1]
                remainder = self.max_text_len - text_tokens_len
                if remainder > 0:
                    text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

                # Prob. mask for clf-free guidance conditional dropout. Tells which elts in the batch to keep. Size (b,).
                text_keep_mask = utils.prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)

                # Combines T5 and clf-free guidance masks
                text_keep_mask_embed = einops.rearrange(text_keep_mask, 'b -> b 1 1')
                if utils.exists(text_mask):
                    if remainder > 0:
                        text_mask = F.pad(text_mask, (0, remainder), value=False)

                    text_mask = einops.rearrange(text_mask, 'b n -> b n 1')  # (b, self.max_text_len, 1)
                    text_keep_mask_embed = text_mask & text_keep_mask_embed  # (b, self.max_text_len, 1)

                # Creates NULL tensor of size (1, self.max_text_len, cond_dim)
                null_text_embed = self.null_text_embed.to(text_tokens.dtype)  # for some reason pytorch AMP not working

                # Replaces masked elements with NULL
                text_tokens = torch.where(
                    text_keep_mask_embed,
                    text_tokens,
                    null_text_embed
                )

                # Extra non-attention conditioning by projecting and then summing text embeddings to time (text hiddens)
                # Pool the text tokens along the word dimension.
                mean_pooled_text_tokens = text_tokens.mean(dim=-2)

                # Project to `time_cond_dim`
                text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)  # (b, cond_dim) -> (b, time_cond_dim)

                null_text_hidden = self.null_text_hidden.to(t.dtype)

                # Drop relevant conditioning info as demanded by clf-free guidance mask
                text_keep_mask_hidden = einops.rearrange(text_keep_mask, 'b -> b 1')
                text_hiddens = torch.where(
                    text_keep_mask_hidden,
                    text_hiddens,
                    null_text_hidden
                )

                # Add this conditioning to our `t` tensor
                t = t + text_hiddens

            # main conditioning tokens `c` - concatenate time/text tokens
            c = time_tokens if not utils.exists(text_tokens) else torch.cat((time_tokens, text_tokens), dim=-2)

            # normalize conditioning tokens
            c = self.norm_cond(c)

            return t, c

    def _generate_t_tokens(
            self,
            time: torch.tensor,
            lowres_noise_times: torch.tensor
    ):
        '''
        Generate t and time_tokens
        :param time: Tensor of shape (b,). The timestep for each image in the batch.
        :param lowres_noise_times:  Tensor of shape (b,). The timestep for each low-res conditioning image.
        :return: tuple(t, time_tokens)
            t: Tensor of shape (b, time_cond_dim) where `time_cond_dim` is 4x the UNet `dim`, or 8 if conditioning
            on lowres image.
            time_tokens: Tensor of shape (b, NUM_TIME_TOKENS, dim), where `NUM_TIME_TOKENS` defaults to 2.
        '''
        time_hiddens = self.to_time_hiddens(time)
        t = self.to_time_cond(time_hiddens)
        time_tokens = self.to_time_tokens(time_hiddens)

        return t, time_tokens    

    def forward(self, *args, **kwargs):
        # Time conditioned vectors
        t, time_tokens = self._generate_t_tokens(time, lowres_noise_times=False)

        # Text conditioned vectors
        t, c = self._text_condition(text_embeds, batch_size, cond_drop_prob, device, text_mask, t, time_tokens)





m = GaussianDiffusion(timesteps=1000)

x_start = torch.randn(1, 3, 32, 32)
t = torch.tensor([5])
_ = m.q_sample(x_start, t)
IPython.embed()