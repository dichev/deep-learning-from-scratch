import torch
from typing import Callable
from lib.layers import Module


class DenoiseDiffusion(Module): # aka DDPM
    """
    Paper: Denoising Diffusion Probabilistic Models
    https://arxiv.org/pdf/2006.11239
    """

    def __init__(self, img_sizes=(1, 32, 32), T=100, betas=(0.0001, 0.02), noise_predictor: Callable=None):
        assert noise_predictor is not None, f'Provide noise predictor function or model'

        # Cache variances schedule
        beta = torch.cat((
            torch.zeros(1),            # no variance at t=0
            torch.linspace(*betas, T)  # linear schedule
        ), dim=0)
        alpha = 1 - beta
        alpha_cum = torch.cumprod(alpha, dim=0)

        self.beta = self.register_buffer('beta', beta)
        self.alpha = self.register_buffer('alpha', alpha)
        self.alpha_cum = self.register_buffer('alpha_cum', alpha_cum)
        self.predictor = noise_predictor
        self.img_sizes = img_sizes
        self.T = T

    def diffuse(self, x0, t):  # forward process
        B, C, H, W = x0.shape

        # q posterior: q(xₜ|x₀) = N(μ=√Παₜ x₀), σ²=(1−Παₜ)I)      # ref: (4)
        mean_scale =self.alpha_cum[t].sqrt().view(B, 1, 1, 1)
        std = torch.sqrt(1-self.alpha_cum[t]).view(B, 1, 1, 1)

        # sampling: xₜ ~ q(xₜ|x₀), using reparameterization trick: x = μ + σz₀,  with z₀∼N(0, I)
        z = torch.randn_like(x0)
        x_t = mean_scale * x0 + std * z     # xₜ = μ + σz₀ = √Παₜ x₀ + (1−Παₜ)z₀
        return x_t, z

    def ELBO_noise(self, noise, noise_0, t, ignore_weighting=True):
        assert torch.all(t > 0) and ignore_weighting # paper: "We do not weight the denoising loss equally for all steps. However, for simplicity, we can drop these. It has minor impacts."
        loss = ((noise - noise_0)**2).mean()
        return loss

    @torch.no_grad()
    def sample_denoise(self, n, context, device=None):  # reverse process
        C, H, W = self.img_sizes
        alpha, alpha_cum = self.alpha, self.alpha_cum

        # ref: (Algorithm 2 Sampling)
        x_t = torch.randn((n, C, H, W)).to(device)
        history = [x_t]
        for t in range(self.T, 0, -1): # [T -> 1]
            # estimate the image from noise image
            t_batch = torch.tensor([t]).expand(n).to(device)
            # extract noise z:
            z = torch.randn((n, C, H, W)).to(device) if t > 1 else 0.  # ignoring the variance of the final step (t=1)
            noise_pred = self.predictor(x_t, t_batch, context.to(device))  # to be subtracted from x_t

            # denoise: remove predicted noise + add some scheduled noise z
            mean_t = (x_t - noise_pred * (1-alpha[t]) / torch.sqrt(1-alpha_cum[t])) / alpha[t].sqrt()
            std_t = self.beta[t].sqrt()  # paper: Experimentally, both σₜ² = βₜ and σₜ²= βₜ(1−Πα[t-1])/(1−α[t]) had similar results. The first choice is optimal for x₀∼N(0, I)
            x_t = mean_t + std_t * z     # + extra noise
            history.append(x_t)

        return x_t, torch.stack(history)
