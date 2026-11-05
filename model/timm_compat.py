"""Compatibility helpers for transformer blocks across timm releases."""

from __future__ import annotations

import inspect
from typing import Any

try:
    from timm.models.vision_transformer import Block as TimmTransformerBlock
except ImportError as exc:
    raise ImportError("timm is required for the ZEBRA transformer blocks.") from exc


_BLOCK_PARAMETERS = inspect.signature(TimmTransformerBlock.__init__).parameters


def make_transformer_block(
    dim: int,
    num_heads: int,
    *,
    dropout: float = 0.0,
    **kwargs: Any,
) -> TimmTransformerBlock:
    """Build a timm Block with the dropout keyword supported at runtime.

    timm 0.6 calls this argument ``drop``; newer releases renamed it to
    ``proj_drop``. The distinction does not change the learned parameter names
    used by the checkpoints.
    """
    if "proj_drop" in _BLOCK_PARAMETERS:
        kwargs["proj_drop"] = dropout
    elif "drop" in _BLOCK_PARAMETERS:
        kwargs["drop"] = dropout
    else:
        raise TypeError(
            "Unsupported timm Block API: neither 'drop' nor 'proj_drop' is "
            "present in Block.__init__."
        )
    return TimmTransformerBlock(dim, num_heads, **kwargs)
