import numpy as np
import contextlib
import torch
import random
import os
from sgm.util import append_dims


def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        print("Note: not using cudnn.deterministic")


def unclip_recon(
    x,
    diffusion_engine,
    vector_suffix,
    num_samples=1,
    offset_noise_level=0.04,
    device="cpu",
    autocast_dtype=torch.float16,
):
    assert x.ndim == 3
    if x.shape[0] == 1:
        x = x[[0]]
    device_type = torch.device(device).type
    amp_context = (
        torch.autocast(device_type=device_type, dtype=autocast_dtype)
        if device_type == "cuda" and autocast_dtype is not None
        else contextlib.nullcontext()
    )
    with torch.no_grad(), amp_context, diffusion_engine.ema_scope():
        z = torch.randn(num_samples, 4, 96, 96).to(device)
        tokens = x
        c = {
            "crossattn": tokens.repeat(num_samples, 1, 1).to(z.device),
            "vector": vector_suffix.repeat(num_samples, 1).to(z.device),
        }
        unconditional_tokens = torch.zeros_like(x)
        uc = {
            "crossattn": unconditional_tokens.repeat(num_samples, 1, 1).to(z.device),
            "vector": vector_suffix.repeat(num_samples, 1).to(z.device),
        }
        for k in c:
            (c[k], uc[k]) = map(lambda y: y[k][:num_samples].to(device), (c, uc))
        noise = torch.randn_like(z)
        sigmas = diffusion_engine.sampler.discretization(
            diffusion_engine.sampler.num_steps
        )
        sigma = sigmas[0].to(z.device)
        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(1.0 + sigmas[0] ** 2.0)

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
        samples = (samples_x + 1.0) / 2.0
        samples = torch.clamp(samples, min=0.0, max=1.0)
        return samples
