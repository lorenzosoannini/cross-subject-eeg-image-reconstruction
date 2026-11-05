"""Learned shared/private branches and joint reconstruction."""

from __future__ import annotations
import torch
from torch import nn
from .timm_compat import make_transformer_block


class DSNLatentBranch(nn.Module):
    """A non-residual bottleneck branch over aligned REVE channel tokens."""

    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        depth: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim < 1 or code_dim < 1:
            raise ValueError("input_dim and code_dim must be positive.")
        if depth < 1:
            raise ValueError("DSN branch depth must be >= 1.")
        if num_heads < 1 or code_dim % num_heads != 0:
            raise ValueError(
                "DSN code_dim must be divisible by num_heads, got "
                f"code_dim={code_dim}, num_heads={num_heads}."
            )
        if not 0 <= dropout < 1:
            raise ValueError("DSN dropout must be in [0, 1).")

        self.input_dim = int(input_dim)
        self.code_dim = int(code_dim)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.down_project = nn.Linear(self.input_dim, self.code_dim)
        self.blocks = nn.ModuleList(
            [
                make_transformer_block(
                    self.code_dim,
                    num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    dropout=dropout,
                    attn_drop=dropout,
                    drop_path=dropout,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )
        self.code_norm = nn.LayerNorm(self.code_dim)
        self.up_project = nn.Linear(self.code_dim, self.input_dim)
        self.output_norm = nn.LayerNorm(self.input_dim)

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(
                "Expected DSN branch input [B, T, D] with D="
                f"{self.input_dim}, got {tuple(features.shape)}."
            )
        code = self.down_project(self.input_norm(features))
        for block in self.blocks:
            code = block(code)
        code = self.code_norm(code)
        output = self.output_norm(self.up_project(code))
        return output, code


class DSNLatentDecoder(nn.Module):
    """Reconstruct the full REVE latent from both independently learned codes."""

    def __init__(
        self,
        invariant_dim: int,
        subject_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        combined_dim = int(invariant_dim) + int(subject_dim)
        if min(combined_dim, output_dim, hidden_dim) < 1:
            raise ValueError("All DSN decoder dimensions must be positive.")
        self.combined_dim = combined_dim
        self.output_dim = int(output_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(
        self,
        invariant_code: torch.Tensor,
        subject_code: torch.Tensor,
    ) -> torch.Tensor:
        if invariant_code.shape[:-1] != subject_code.shape[:-1]:
            raise ValueError(
                "DSN codes must share batch/token axes, got "
                f"{tuple(invariant_code.shape)} and "
                f"{tuple(subject_code.shape)}."
            )
        combined = torch.cat((invariant_code, subject_code), dim=-1)
        if combined.shape[-1] != self.combined_dim:
            raise ValueError(
                "Unexpected concatenated DSN width: "
                f"{combined.shape[-1]} != {self.combined_dim}."
            )
        return self.network(combined)


class DSNLatentDecomposition(nn.Module):
    """Two learned branches plus a shallow joint latent decoder."""

    def __init__(
        self,
        reve_dim: int,
        invariant_dim: int = 512,
        subject_dim: int = 256,
        branch_depth: int = 2,
        branch_heads: int = 8,
        decoder_hidden_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.reve_dim = int(reve_dim)
        self.invariant_dim = int(invariant_dim)
        self.subject_dim = int(subject_dim)
        self.branch_depth = int(branch_depth)
        self.branch_heads = int(branch_heads)
        self.decoder_hidden_dim = int(decoder_hidden_dim)

        self.invariant_branch = DSNLatentBranch(
            input_dim=self.reve_dim,
            code_dim=self.invariant_dim,
            depth=self.branch_depth,
            num_heads=self.branch_heads,
            dropout=dropout,
        )
        self.subject_branch = DSNLatentBranch(
            input_dim=self.reve_dim,
            code_dim=self.subject_dim,
            depth=self.branch_depth,
            num_heads=self.branch_heads,
            dropout=dropout,
        )
        self.decoder = DSNLatentDecoder(
            invariant_dim=self.invariant_dim,
            subject_dim=self.subject_dim,
            output_dim=self.reve_dim,
            hidden_dim=self.decoder_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        full_latent: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        invariant, invariant_code = self.invariant_branch(full_latent)
        subject, subject_code = self.subject_branch(full_latent)
        reconstruction = self.decoder(invariant_code, subject_code)
        return (
            invariant,
            subject,
            reconstruction,
            invariant_code,
            subject_code,
        )
