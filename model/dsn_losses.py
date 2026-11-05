"""Regularizers for the learned shared/private REVE decomposition.

The losses in this module are intentionally independent from the training
script so they can be unit-tested and reused by evaluation/ablation tools.
All statistics are accumulated in float32 because covariance estimates are
particularly fragile under fp16/bf16 autocast.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_observations(features: torch.Tensor) -> torch.Tensor:
    """Flatten every axis except the feature axis into observations."""
    if features.ndim < 2:
        raise ValueError(
            "Expected features with shape [..., D], got " f"{tuple(features.shape)}."
        )
    if features.shape[-1] < 1:
        raise ValueError("The feature dimension must be non-empty.")
    observations = features.float().reshape(-1, features.shape[-1])
    if observations.shape[0] < 2:
        raise ValueError("At least two observations are required for DSN statistics.")
    return observations


def cross_covariance_loss(
    invariant_code: torch.Tensor,
    subject_code: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Penalize normalized batch cross-covariance between two latent codes.

    Unlike a per-sample cosine orthogonality constraint, this is a soft
    distribution-level constraint. The two codes may have different widths.
    Token axes are treated as additional observations, which is appropriate
    here because REVE preserves aligned EEG-channel tokens in both branches.
    """
    if eps <= 0:
        raise ValueError("eps must be > 0.")

    invariant = _as_observations(invariant_code)
    subject = _as_observations(subject_code)
    if invariant.shape[0] != subject.shape[0]:
        raise ValueError(
            "The two DSN codes must have the same number of observations, "
            f"got {invariant.shape[0]} and {subject.shape[0]}."
        )

    invariant = invariant - invariant.mean(dim=0, keepdim=True)
    subject = subject - subject.mean(dim=0, keepdim=True)
    invariant = invariant / torch.sqrt(
        invariant.square().mean(dim=0, keepdim=True) + eps
    )
    subject = subject / torch.sqrt(subject.square().mean(dim=0, keepdim=True) + eps)

    denominator = max(1, invariant.shape[0] - 1)
    cross_covariance = invariant.transpose(0, 1) @ subject
    cross_covariance = cross_covariance / float(denominator)

    # A plain mean over D_i * D_s would make this objective vanish as the
    # bottlenecks get wider (512 x 256 with the default configuration). Use a
    # VICReg-style squared Frobenius norm normalized by the smaller code width:
    # an identity-like aligned subspace has an order-one penalty, while the
    # weight remains comparable across rectangular bottleneck ablations.
    normalization = float(
        min(
            cross_covariance.shape[0],
            cross_covariance.shape[1],
        )
    )
    return cross_covariance.square().sum() / normalization


def variance_floor_loss(
    invariant_code: torch.Tensor,
    subject_code: torch.Tensor,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Prevent either branch from satisfying decorrelation by collapsing."""
    if target_std <= 0:
        raise ValueError("target_std must be > 0.")
    if eps <= 0:
        raise ValueError("eps must be > 0.")

    penalties = []
    for code in (invariant_code, subject_code):
        observations = _as_observations(code)
        std = torch.sqrt(observations.var(dim=0, unbiased=False) + eps)
        penalties.append(F.relu(float(target_std) - std).mean())
    return torch.stack(penalties).mean()


def latent_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth-L1 reconstruction of the stop-gradient full REVE latent."""
    if prediction.shape != target.shape:
        raise ValueError(
            "Latent reconstruction shapes differ: "
            f"prediction={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}."
        )
    if beta <= 0:
        raise ValueError("beta must be > 0.")
    return F.smooth_l1_loss(
        prediction.float(),
        target.detach().float(),
        beta=float(beta),
    )
