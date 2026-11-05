"""Feature prior with the corrected DDIM sampling schedule."""

import torch
import random
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import (
    l2norm,
    default,
    exists,
    RotaryEmbedding,
    SinusoidalPosEmb,
    MLP,
    Rearrange,
    repeat,
    rearrange,
    prob_mask_like,
    LayerNorm,
    RelPosBias,
    Attention,
    FeedForward,
)


class BrainDiffusionPrior(DiffusionPrior):

    @torch.no_grad()
    def p_sample_loop_ddim(
        self, shape, text_cond, *, timesteps, eta=1.0, cond_scale=1.0, generator=None
    ):
        (batch, device, alphas, total_timesteps) = (
            shape[0],
            self.device,
            self.noise_scheduler.alphas_cumprod,
            self.noise_scheduler.num_timesteps,
        )
        times = torch.linspace(
            -1,
            total_timesteps - 1,
            steps=timesteps + 1,
            device="cpu",
            dtype=torch.float64,
        )
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        image_embed = torch.randn(shape, device=device, generator=generator)
        x_start = None
        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale
        for time, time_next in time_pairs:
            alpha = alphas[time]
            alpha_next = (
                alphas[time_next]
                if time_next >= 0
                else torch.ones((), device=device, dtype=alphas.dtype)
            )
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.net.self_cond else None
            pred = self.net.forward_with_cond_scale(
                image_embed,
                time_cond,
                self_cond=self_cond,
                cond_scale=cond_scale,
                **text_cond,
            )
            if self.predict_v:
                x_start = self.noise_scheduler.predict_start_from_v(
                    image_embed, t=time_cond, v=pred
                )
            elif self.predict_x_start:
                x_start = pred
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(
                    image_embed, t=time_cond, noise=pred
                )
            if not self.predict_x_start:
                x_start.clamp_(-1.0, 1.0)
            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)
            pred_noise = self.noise_scheduler.predict_noise_from_start(
                image_embed, t=time_cond, x0=x_start
            )
            if time_next < 0:
                image_embed = x_start
                continue
            c1_squared = (
                (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
            ).clamp_min(0)
            c1 = eta * c1_squared.sqrt()
            c2 = (1 - alpha_next - torch.square(c1)).clamp_min(0).sqrt()
            noise = (
                torch.randn(
                    image_embed.shape,
                    device=image_embed.device,
                    dtype=image_embed.dtype,
                    generator=generator,
                )
                if time_next > 0
                else 0.0
            )
            image_embed = x_start * alpha_next.sqrt() + c1 * noise + c2 * pred_noise
        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            image_embed = self.l2norm_clamp_embed(image_embed)
        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise=None, sample_weight=None):
        noise = default(noise, lambda: torch.randn_like(image_embed))
        image_embed_noisy = self.noise_scheduler.q_sample(
            x_start=image_embed, t=times, noise=noise
        )
        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with torch.no_grad():
                self_cond = self.net(image_embed_noisy, times, **text_cond).detach()
        pred = self.net(
            image_embed_noisy,
            times,
            self_cond=self_cond,
            text_cond_drop_prob=self.text_cond_drop_prob,
            image_cond_drop_prob=self.image_cond_drop_prob,
            **text_cond,
        )
        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)
        if self.predict_v:
            target = self.noise_scheduler.calculate_v(image_embed, times, noise)
        elif self.predict_x_start:
            target = image_embed
        else:
            target = noise
        if sample_weight is None:
            loss = nn.functional.mse_loss(pred, target)
        else:
            weights = torch.as_tensor(
                sample_weight, device=pred.device, dtype=pred.dtype
            )
            if weights.ndim != 1 or weights.shape[0] != pred.shape[0]:
                raise ValueError(
                    f"sample_weight must have shape [batch], got {tuple(weights.shape)} for batch={pred.shape[0]}."
                )
            if torch.any(weights < 0) or not torch.isfinite(weights).all():
                raise ValueError("sample_weight must be finite and non-negative.")
            per_sample = (
                nn.functional.mse_loss(pred, target, reduction="none")
                .flatten(1)
                .mean(dim=1)
            )
            loss = (per_sample * weights).mean()
        return (loss, pred)

    def forward(self, *, text_embed, image_embed):
        times = self.noise_scheduler.sample_random_times(image_embed.shape[0])
        return self.p_losses(image_embed, times, text_cond={"text_embed": text_embed})


class PriorNetwork(nn.Module):

    def __init__(
        self,
        dim,
        num_timesteps=None,
        num_time_embeds=1,
        num_tokens=257,
        causal=True,
        learned_query_mode="none",
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode
        self.to_time_embeds = nn.Sequential(
            (
                nn.Embedding(num_timesteps, dim * num_time_embeds)
                if exists(num_timesteps)
                else nn.Sequential(
                    SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)
                )
            ),
            Rearrange("b (n d) -> b n d", n=num_time_embeds),
        )
        if self.learned_query_mode == "token":
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim))
        if self.learned_query_mode == "pos_emb":
            scale = dim ** (-0.5)
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        if self.learned_query_mode == "all_pos_emb":
            scale = dim ** (-0.5)
            self.learned_query = nn.Parameter(
                torch.randn(num_tokens * 2 + 1, dim) * scale
            )
        self.causal_transformer = FlaggedCausalTransformer(
            dim=dim, causal=causal, **kwargs
        )
        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))
        self.num_tokens = num_tokens
        self.self_cond = False

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(
            *args, brain_cond_drop_prob=1.0, image_cond_drop_prob=1, **kwargs
        )
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        self_cond=None,
        brain_embed=None,
        text_embed=None,
        brain_cond_drop_prob=0.0,
        text_cond_drop_prob=None,
        image_cond_drop_prob=0.0,
    ):
        if text_embed is not None:
            brain_embed = text_embed
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob
        (batch, _, dim, device, dtype) = (
            *image_embed.shape,
            image_embed.device,
            image_embed.dtype,
        )
        brain_keep_mask = prob_mask_like(
            (batch,), 1 - brain_cond_drop_prob, device=device
        )
        brain_keep_mask = rearrange(brain_keep_mask, "b -> b 1 1")
        image_keep_mask = prob_mask_like(
            (batch,), 1 - image_cond_drop_prob, device=device
        )
        image_keep_mask = rearrange(image_keep_mask, "b -> b 1 1")
        null_brain_embeds = self.null_brain_embeds.to(brain_embed.dtype)
        brain_embed = torch.where(brain_keep_mask, brain_embed, null_brain_embeds[None])
        null_image_embed = self.null_image_embed.to(image_embed.dtype)
        image_embed = torch.where(image_keep_mask, image_embed, null_image_embed[None])
        if self.continuous_embedded_time:
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)
        if self.learned_query_mode == "token":
            learned_queries = repeat(self.learned_query, "n d -> b n d", b=batch)
        elif self.learned_query_mode == "pos_emb":
            pos_embs = repeat(self.learned_query, "n d -> b n d", b=batch)
            image_embed = image_embed + pos_embs
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        elif self.learned_query_mode == "all_pos_emb":
            pos_embs = repeat(self.learned_query, "n d -> b n d", b=batch)
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        else:
            learned_queries = torch.empty((batch, 0, dim), device=brain_embed.device)
        tokens = torch.cat(
            (brain_embed, time_embed, image_embed, learned_queries), dim=-2
        )
        if self.learned_query_mode == "all_pos_emb":
            tokens = tokens + pos_embs
        tokens = self.causal_transformer(tokens)
        pred_image_embed = tokens[..., -self.num_tokens :, :]
        return pred_image_embed


class FlaggedCausalTransformer(nn.Module):

    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        norm_in=False,
        norm_out=True,
        attn_dropout=0.0,
        ff_dropout=0.0,
        final_proj=True,
        normformer=False,
        rotary_emb=True,
        causal=True,
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity()
        self.rel_pos_bias = RelPosBias(heads=heads)
        rotary_emb = RotaryEmbedding(dim=min(32, dim_head)) if rotary_emb else None
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            causal=causal,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_emb=rotary_emb,
                        ),
                        FeedForward(
                            dim=dim,
                            mult=ff_mult,
                            dropout=ff_dropout,
                            post_activation_norm=normformer,
                        ),
                    ]
                )
            )
        self.norm = LayerNorm(dim, stable=True) if norm_out else nn.Identity()
        self.project_out = (
            nn.Linear(dim, dim, bias=False) if final_proj else nn.Identity()
        )

    def forward(self, x):
        (n, device) = (x.shape[1], x.device)
        x = self.init_norm(x)
        attn_bias = self.rel_pos_bias(n, n + 1, device=device)
        for attn, ff in self.layers:
            x = attn(x, attn_bias=attn_bias) + x
            x = ff(x) + x
        out = self.norm(x)
        return self.project_out(out)
