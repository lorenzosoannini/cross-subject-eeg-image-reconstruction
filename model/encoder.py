"""REVE, LoRA and EEG-to-visual token components."""

from __future__ import annotations
import math
from typing import Any, Optional
import numpy as np
import torch
from torch import nn
from .timm_compat import make_transformer_block


class GradReverse(torch.autograd.Function):
    """
    Gradient-reversal operator used by the subject discriminator.

    `train_zebra.py` updates the class-level `lambd` value during training.
    The forward pass snapshots that value in the autograd context so the
    matching backward pass always uses the coefficient that was active when
    the current graph was built.
    """

    lambd: float = 1.0

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        lambd = float(GradReverse.lambd)
        if not np.isfinite(lambd) or lambd < 0.0:
            raise ValueError(
                "GradReverse.lambd must be a finite non-negative value, "
                f"got {lambd}."
            )
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return -ctx.lambd * grad_output


class LoRALinear(nn.Linear):
    """Drop-in ``nn.Linear`` with a gated low-rank residual branch."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        enabled: bool,
    ):
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}.")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be > 0, got {alpha}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {dropout}.")

        super().__init__(
            base_layer.in_features,
            base_layer.out_features,
            bias=base_layer.bias is not None,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )
        # Reuse the original Parameter objects, avoiding a second copy of the
        # large REVE projection matrices and retaining their state-dict names.
        self.weight = base_layer.weight
        self.bias = base_layer.bias
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                self.in_features,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.out_features,
                self.rank,
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.register_buffer(
            "lora_enabled",
            torch.tensor(bool(enabled), dtype=torch.bool),
            persistent=False,
        )
        self._lora_enabled = bool(enabled)

    def set_lora_enabled(self, enabled: bool) -> None:
        self._lora_enabled = bool(enabled)
        self.lora_enabled.fill_(self._lora_enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = nn.functional.linear(x, self.weight, self.bias)
        if not self._lora_enabled:
            return result
        lora_hidden = nn.functional.linear(
            self.lora_dropout(x),
            self.lora_A,
        )
        lora_result = nn.functional.linear(lora_hidden, self.lora_B)
        return result + self.scaling * lora_result


def inject_reve_lora(
    reve: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    enabled: bool,
) -> list[str]:
    """Inject LoRA only into the pretrained REVE attention backbone."""
    transformer = getattr(reve, "transformer", None)
    layers = getattr(transformer, "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError(
            "REVE has no non-empty `transformer.layers`; LoRA injection "
            "cannot continue."
        )

    targets = []
    for layer_index, layer in enumerate(layers):
        if len(layer) < 1:
            raise RuntimeError(
                f"REVE transformer layer {layer_index} has no attention "
                "module at index 0."
            )
        attention = layer[0]
        for projection_name in ("to_qkv", "to_out"):
            projection = getattr(attention, projection_name, None)
            if not isinstance(projection, nn.Linear):
                raise RuntimeError(
                    "Expected REVE attention projection "
                    f"transformer.layers.{layer_index}.0."
                    f"{projection_name} to be nn.Linear, got "
                    f"{type(projection).__name__}."
                )
            targets.append(
                (
                    "transformer.layers." f"{layer_index}.0.{projection_name}",
                    projection,
                )
            )

    injected_names = []
    for module_name, module in targets:
        parent_path, child_name = module_name.rsplit(".", 1)
        parent = reve.get_submodule(parent_path)
        setattr(
            parent,
            child_name,
            LoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                enabled=enabled,
            ),
        )
        injected_names.append(module_name)
    return injected_names


class SubjDiscriminator(nn.Module):
    def __init__(self, feature_dim: int, token_num: int, domain_classes: int):
        super().__init__()
        self.class_classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(feature_dim * token_num, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, domain_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.class_classifier(GradReverse.apply(x))


class SubjClassifier(nn.Module):
    def __init__(self, feature_dim: int, token_num: int, domain_classes: int):
        super().__init__()
        self.class_classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(feature_dim * token_num, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, domain_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.class_classifier(x)


class SemanticBottleneck(nn.Module):
    """ZEBRA projector operating on 256 visual-latent tokens."""

    def __init__(
        self,
        in_dim: int = 1216,
        out_dim: int = 1664,
        seq_dim: int = 256,
        hidden_dim: int = 2048,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(seq_dim),
            nn.Linear(seq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, seq_dim),
        )
        self.project = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 256, 1216]
        x = x.transpose(1, 2)  # [B, 1216, 256]
        x = self.mlp(x)
        x = x.transpose(1, 2)  # [B, 256, 1216]
        return self.project(x)  # [B, 256, 1664]


class ChannelWiseAttention(nn.Module):
    """Copied in spirit from the original ZEBRA implementation."""

    def __init__(self, in_dim: int, num_heads: int = 1):
        super().__init__()
        if in_dim % num_heads != 0:
            raise ValueError(
                f"in_dim={in_dim} must be divisible by num_heads={num_heads}."
            )
        self.query_proj = nn.Linear(in_dim, in_dim)
        self.key_proj = nn.Linear(in_dim, in_dim)
        self.value_proj = nn.Linear(in_dim, in_dim)
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.out_proj = nn.Linear(in_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        bsz, n_tokens, channels = x.shape
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        scores = (q.transpose(-2, -1) @ k) / (self.head_dim**0.5)
        attn = torch.softmax(scores, dim=-1)
        attn_out = attn @ v.transpose(-2, -1)
        attn_out = attn_out.transpose(-2, -1).reshape(
            bsz,
            n_tokens,
            channels,
        )
        return self.out_proj(attn_out)


class SparseAwareVisionProjector(nn.Module):
    """Project visual-latent tokens into OpenCLIP vision-token space."""

    def __init__(
        self,
        in_dim: int = 1216,
        out_dim: int = 1664,
        patch_num: int = 256,
    ):
        super().__init__()
        self.patch_num = patch_num
        self.bottleneck = SemanticBottleneck(
            in_dim=in_dim,
            out_dim=out_dim,
            seq_dim=patch_num,
        )
        self.broadcaster = ChannelWiseAttention(in_dim=out_dim, num_heads=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 256, 1216]
        x = self.bottleneck(x)
        return self.broadcaster(x)


class CrossAttentionResamplerLayer(nn.Module):
    """Perceiver-style EEG-to-visual resampling layer."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.query_norm = nn.LayerNorm(dim)
        self.source_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        hidden_dim = int(dim * mlp_ratio)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        normalized_source = self.source_norm(source)
        cross_out, _ = self.cross_attn(
            self.query_norm(queries),
            normalized_source,
            normalized_source,
            need_weights=False,
        )
        queries = queries + cross_out

        normalized_queries = self.self_norm(queries)
        self_out, _ = self.self_attn(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            need_weights=False,
        )
        queries = queries + self_out
        return queries + self.ffn(self.ffn_norm(queries))


class VisualTokenResampler(nn.Module):
    """Convert natural EEG tokens into 256 visual-aligned latent tokens."""

    def __init__(
        self,
        dim: int = 1216,
        out_tokens: int = 256,
        num_heads: int = 19,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if out_tokens < 1:
            raise ValueError("out_tokens must be >= 1.")
        if depth < 1:
            raise ValueError("depth must be >= 1.")
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.dim = dim
        self.out_tokens = out_tokens
        self.query_tokens = nn.Parameter(torch.empty(1, out_tokens, dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.layers = nn.ModuleList(
            [
                CrossAttentionResamplerLayer(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        if source.ndim != 3:
            raise RuntimeError(
                f"Expected source tokens [B, N, D], got {tuple(source.shape)}."
            )
        if source.shape[-1] != self.dim:
            raise RuntimeError(
                f"Expected source dim {self.dim}, got {source.shape[-1]}."
            )

        queries = self.query_tokens.expand(source.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, source)
        return self.final_norm(queries)


def _find_first_tensor(obj: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(obj):
        return obj

    if isinstance(obj, dict):
        preferred_keys = [
            "last_hidden_state",
            "hidden_states",
            "embeddings",
            "features",
            "x",
            "logits",
        ]
        for key in preferred_keys:
            value = obj.get(key)
            if torch.is_tensor(value):
                return value
            if isinstance(value, (tuple, list)):
                for item in value:
                    if torch.is_tensor(item):
                        return item
        for value in obj.values():
            tensor = _find_first_tensor(value)
            if tensor is not None:
                return tensor

    if isinstance(obj, (tuple, list)):
        for value in obj:
            tensor = _find_first_tensor(value)
            if tensor is not None:
                return tensor

    if hasattr(obj, "last_hidden_state") and torch.is_tensor(obj.last_hidden_state):
        return obj.last_hidden_state

    if hasattr(obj, "logits") and torch.is_tensor(obj.logits):
        return obj.logits

    return None


class REVEEncoder(nn.Module):
    """Wrap REVE with the encoder interface expected by the ZEBRA backbone."""

    def __init__(
        self,
        reve_model_name: str = "brain-bzh/reve-large",
        reve_positions_path: Optional[str] = None,
        n_channels: Optional[int] = None,
        reve_dim: int = 1216,
        freeze_reve: bool = True,
        reve_adaptation: Optional[str] = None,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        lora_enabled: bool = True,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        if not reve_positions_path:
            raise ValueError(
                "reve_positions_path must be provided explicitly for the "
                "THINGS-EEG2 channel montage."
            )

        self.reve_model_name = reve_model_name
        self.reve_positions_path = str(reve_positions_path)
        self.reve_dim = int(reve_dim)
        if reve_adaptation is None:
            reve_adaptation = "frozen" if freeze_reve else "full"
        if reve_adaptation not in {"frozen", "lora", "full"}:
            raise ValueError(
                "reve_adaptation must be one of frozen/lora/full, got "
                f"{reve_adaptation!r}."
            )
        self.reve_adaptation = str(reve_adaptation)
        self.freeze_reve = self.reve_adaptation == "frozen"
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.lora_target_modules: tuple[str, ...] = ()
        self.trust_remote_code = bool(trust_remote_code)

        from transformers import AutoModel

        self.reve = AutoModel.from_pretrained(
            reve_model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.reve_adaptation == "lora":
            self.lora_target_modules = tuple(
                inject_reve_lora(
                    self.reve,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                    enabled=lora_enabled,
                )
            )
        self.configure_trainability()

        positions = self._load_positions(self.reve_positions_path)
        self.n_channels = int(positions.shape[0])
        if n_channels is not None and int(n_channels) != self.n_channels:
            raise RuntimeError(
                "The configured EEG channel count does not match the REVE "
                f"positions file: n_channels={int(n_channels)}, "
                f"positions={self.n_channels} from {self.reve_positions_path}."
            )
        self.register_buffer("positions", positions, persistent=True)

    def configure_trainability(self) -> None:
        """Restore the requested REVE policy after broad model toggles."""
        if self.reve_adaptation == "full":
            for param in self.reve.parameters():
                param.requires_grad_(True)
            return

        for param in self.reve.parameters():
            param.requires_grad_(False)
        if self.reve_adaptation == "lora":
            for module in self.reve.modules():
                if isinstance(module, LoRALinear):
                    module.lora_A.requires_grad_(True)
                    module.lora_B.requires_grad_(True)

    def set_lora_enabled(self, enabled: bool) -> None:
        if self.reve_adaptation != "lora":
            return
        for module in self.reve.modules():
            if isinstance(module, LoRALinear):
                module.set_lora_enabled(enabled)
        if enabled and self.training:
            self.reve.train(True)
        else:
            self.reve.eval()

    @property
    def lora_enabled(self) -> bool:
        if self.reve_adaptation != "lora":
            return False
        states = {
            module._lora_enabled
            for module in self.reve.modules()
            if isinstance(module, LoRALinear)
        }
        if len(states) != 1:
            raise RuntimeError(
                "REVE LoRA modules do not share a consistent enabled state."
            )
        return states.pop()

    @staticmethod
    def _load_positions(path: str) -> torch.Tensor:
        resolved_path = path
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.join(REPO_ROOT, resolved_path)

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                "REVE positions file not found: "
                f"{resolved_path}. Expected a local [C, 3] .npy file."
            )

        arr = np.load(resolved_path).astype("float32")
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 1:
            raise RuntimeError(
                "Expected REVE positions shape [C, 3], got "
                f"{arr.shape} from {resolved_path}."
            )
        if not np.isfinite(arr).all():
            raise RuntimeError(f"REVE positions contain NaN or Inf: {resolved_path}")
        if np.unique(arr, axis=0).shape[0] != arr.shape[0]:
            raise RuntimeError(
                f"REVE positions contain duplicate rows: {resolved_path}"
            )
        return torch.from_numpy(arr)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_reve or (
            self.reve_adaptation == "lora" and not self.lora_enabled
        ):
            self.reve.eval()
        return self

    def _call_reve(self, eeg: torch.Tensor, positions: torch.Tensor) -> Any:
        call_attempts = [
            lambda: self.reve(eeg, positions),
            lambda: self.reve(eeg=eeg, positions=positions),
            lambda: self.reve(x=eeg, positions=positions),
            lambda: self.reve(input_values=eeg, positions=positions),
            lambda: self.reve(eeg=eeg, ch_pos=positions),
            lambda: self.reve(x=eeg, ch_pos=positions),
        ]

        type_errors = []
        for attempt in call_attempts:
            try:
                return attempt()
            except TypeError as exc:
                type_errors.append(str(exc))

        raise TypeError(
            "All REVE call conventions failed. Last errors: "
            + " || ".join(type_errors[-3:])
        )

    def _extract_tokens(self, reve_out: Any) -> torch.Tensor:
        tokens = _find_first_tensor(reve_out)
        if tokens is None:
            raise RuntimeError(
                "Could not find tensor output inside REVE output of type "
                f"{type(reve_out)}"
            )

        if tokens.ndim == 4:
            tokens = tokens.flatten(1, 2)
        elif tokens.ndim != 3:
            raise RuntimeError(
                "Expected REVE output with 3 or 4 dims, got shape "
                f"{tuple(tokens.shape)}"
            )

        if tokens.shape[-1] != self.reve_dim:
            raise RuntimeError(
                f"Unexpected REVE feature dim: got {tokens.shape[-1]}, "
                f"expected {self.reve_dim}."
            )
        return tokens

    def forward_encoder(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3:
            raise RuntimeError(f"Expected EEG shape [B, C, T], got {tuple(eeg.shape)}")
        if eeg.shape[1] != self.n_channels:
            raise RuntimeError(
                f"Expected {self.n_channels} EEG channels from the positions "
                f"file, got {eeg.shape[1]}."
            )

        positions = self.positions.to(device=eeg.device, dtype=eeg.dtype)
        positions = positions.unsqueeze(0).expand(eeg.shape[0], -1, -1)

        lora_warmup = self.reve_adaptation == "lora" and not self.lora_enabled
        if self.freeze_reve or lora_warmup:
            with torch.no_grad():
                reve_out = self._call_reve(eeg, positions)
        else:
            reve_out = self._call_reve(eeg, positions)

        tokens = self._extract_tokens(reve_out)
        if lora_warmup:
            # Keep every adapter connected to the loss with a zero gradient,
            # so DDP can build stable buckets while REVE itself runs without
            # storing activations during the head-only warm-up.
            zero_adapter_dependency = sum(
                parameter.reshape(-1)[0] * 0.0
                for module in self.reve.modules()
                if isinstance(module, LoRALinear)
                for parameter in (module.lora_A, module.lora_B)
            )
            tokens = tokens + zero_adapter_dependency
        if tokens.shape[1] != self.n_channels:
            raise RuntimeError(
                "This backbone expects one REVE temporal patch per channel for "
                "the configured 200-sample input. Got "
                f"{tokens.shape[1]} tokens for {self.n_channels} channels."
            )
        return tokens

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.forward_encoder(eeg)


class EEGReconDecoder(nn.Module):
    """Decode each natural channel token directly into its EEG time series."""

    def __init__(self, dim: int = 1216, recon_len: int = 200):
        super().__init__()
        self.dim = int(dim)
        self.recon_len = int(recon_len)
        self.to_signal = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, recon_len),
        )

    def forward(self, brain_embed: torch.Tensor) -> torch.Tensor:
        return self.to_signal(brain_embed)
