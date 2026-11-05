"""Visual supervision heads inherited from ZEBRA."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from diffusers.models.autoencoders.vae import Decoder


class BlurryReconDecoder(nn.Module):

    def __init__(self, vision_dim=1664):
        super(BlurryReconDecoder, self).__init__()
        "Blurry Recon"
        self.maps_projector = nn.Sequential(
            nn.Conv2d(vision_dim, 512, 1, bias=False),
            nn.GroupNorm(1, 512),
            nn.ReLU(True),
            nn.Conv2d(512, 128, 1, bias=False),
            nn.GroupNorm(1, 128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 1, bias=True),
        )
        self.bdropout = nn.Dropout(0.3)
        self.bnorm = nn.GroupNorm(1, 64)
        self.bupsampler = Decoder(
            in_channels=64,
            out_channels=4,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            block_out_channels=[32, 64, 128],
            layers_per_block=1,
        )

    def forward(self, clip_vision_embed):
        (B, N, C) = clip_vision_embed.shape
        (H, W) = (int(math.sqrt(N)), int(math.sqrt(N)))
        clip_vision_embed = rearrange(
            clip_vision_embed, "b (h w) c -> b c h w", h=H, w=W
        )
        clip_vision_embed = F.interpolate(clip_vision_embed, (7, 7))
        b = self.maps_projector(clip_vision_embed)
        b = self.bdropout(b)
        b = self.bnorm(b)
        b_up = self.bupsampler(b)
        return b_up


class GradReverse(torch.autograd.Function):
    lambd = 1.0

    @staticmethod
    def forward(ctx, x):
        ctx.lambd = float(GradReverse.lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output


class ImageDiscriminator(nn.Module):

    def __init__(self, embed_dim=1664, num_classes=80):
        super().__init__()
        self.attn_proj = nn.Linear(embed_dim, 1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        attn_weights = self.attn_proj(GradReverse.apply(x))
        attn_weights = torch.softmax(attn_weights, dim=1)
        x_weighted = (attn_weights * x).sum(dim=1)
        logits = self.classifier(x_weighted)
        return logits


class ImageClassifier(nn.Module):

    def __init__(self, embed_dim=1664, num_classes=80):
        super().__init__()
        self.attn_proj = nn.Linear(embed_dim, 1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        attn_weights = self.attn_proj(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        x_weighted = (attn_weights * x).sum(dim=1)
        logits = self.classifier(x_weighted)
        return logits


class CLIPProj(nn.Module):

    def __init__(self):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(1664, 1280))

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = x @ self.proj
        return x
