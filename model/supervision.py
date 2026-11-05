from __future__ import annotations
from pathlib import Path
from typing import Sequence
import torch
import torch.nn.functional as F
import open_clip
from diffusers import AutoencoderKL


def build_frozen_autoencoder(
    checkpoint_path: str,
    device: torch.device,
) -> AutoencoderKL:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing blurry-reconstruction autoencoder checkpoint: {path}"
        )
    autoencoder = AutoencoderKL(
        down_block_types=["DownEncoderBlock2D"] * 4,
        up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    state = torch.load(path, map_location="cpu")
    autoencoder.load_state_dict(state, strict=True)
    autoencoder.requires_grad_(False)
    autoencoder.eval()
    return autoencoder.to(device)


@torch.no_grad()
def build_text_targets(
    concept_names: Sequence[str],
    cache_dir: str,
    device: torch.device,
    batch_size: int,
    prompt_template: str,
) -> torch.Tensor:
    if "{}" not in prompt_template:
        raise ValueError("--text_prompt_template must contain '{}'.")
    embedder, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14",
        pretrained="laion2b_s39b_b160k",
        cache_dir=cache_dir,
        device=torch.device("cpu"),
    )
    del embedder.visual
    embedder = embedder.to(device).eval()
    embedder.requires_grad_(False)
    # SimpleTokenizer ships its BPE vocabulary with open_clip and does not
    # need the model-download cache. Some open_clip releases forward unknown
    # tokenizer kwargs to SimpleTokenizer, where cache_dir raises TypeError.
    tokenizer = open_clip.get_tokenizer("ViT-bigG-14")
    targets: list[torch.Tensor] = []
    for start in range(0, len(concept_names), batch_size):
        prompts = [
            prompt_template.format(name)
            for name in concept_names[start : start + batch_size]
        ]
        tokens = tokenizer(prompts).to(device)
        pooled = embedder.encode_text(tokens, normalize=False)
        targets.append(pooled.detach().float().cpu())
    del embedder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(targets, dim=0).to(device)


def multi_positive_token_clip_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visual_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    prediction = F.normalize(prediction.float().flatten(1), dim=-1)
    target = F.normalize(target.float().flatten(1), dim=-1)
    logits = prediction @ target.T / temperature
    positives = visual_ids[:, None] == visual_ids[None, :]
    minimum = torch.finfo(logits.dtype).min

    def directional(values: torch.Tensor) -> torch.Tensor:
        positive = torch.logsumexp(values.masked_fill(~positives, minimum), dim=1)
        normalizer = torch.logsumexp(values, dim=1)
        counts = positives.sum(dim=1).clamp_min(1).float()
        return (normalizer - positive + counts.log()).mean()

    return 0.5 * (directional(logits) + directional(logits.T))
