#!/usr/bin/env python3
"""Caption-guided SDXL enhancement for reconstruction tensors."""

from __future__ import annotations

import argparse
import contextlib
import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "generative_models"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors import safe_open
from tqdm.auto import tqdm

import utils
from sgm.util import append_dims

torch.backends.cuda.matmul.allow_tf32 = True


def resolve_precision(requested: str) -> str:
    if requested != "auto":
        return requested
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def autocast_context(precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = (
        torch.bfloat16
        if precision == "bf16"
        else torch.float16 if precision == "fp16" else torch.float32
    )
    return torch.autocast(device_type="cuda", dtype=dtype)


def merge_enhancement_shards(
    shard_dir: Path, num_processes: int, total_images: int
) -> torch.Tensor:
    merged: list[torch.Tensor | None] = [None] * total_images
    for rank in range(num_processes):
        path = shard_dir / f"rank_{rank:03d}_of_{num_processes:03d}.pt"
        payload = torch.load(path, map_location="cpu")
        if len(payload["indices"]) != len(payload["enhanced"]):
            raise RuntimeError(f"Misaligned enhancement shard: {path}")
        for index, image in zip(payload["indices"], payload["enhanced"]):
            index = int(index)
            if not 0 <= index < total_images:
                raise RuntimeError(f"Enhanced index out of range: {index}")
            if merged[index] is not None:
                raise RuntimeError(f"Duplicate enhanced index {index}.")
            merged[index] = image
    if any(image is None for image in merged):
        raise RuntimeError("Distributed enhancement merge is incomplete.")
    return torch.stack([image for image in merged if image is not None]).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--recons_file", default="all_recons.pt")
    parser.add_argument("--gts_file", default="all_gts.pt")
    parser.add_argument("--captions_file", default="pred_test_caption.pt")
    parser.add_argument("--output_file", default="all_enhanced_recons.pt")
    parser.add_argument("--weights_dir", default="pretrained_weights")
    parser.add_argument("--base_checkpoint", default="zavychromaxl_v30.safetensors")
    parser.add_argument("--sampler_steps", type=int, default=25)
    parser.add_argument("--img2img_timepoint", type=int, default=13)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
    )
    parser.add_argument("--generation_retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_previews",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if min(args.sampler_steps, args.img2img_timepoint, args.output_size) < 1:
        parser.error(
            "--sampler_steps, --img2img_timepoint and --output_size must be >= 1"
        )
    if args.img2img_timepoint > args.sampler_steps:
        parser.error("--img2img_timepoint cannot exceed --sampler_steps")
    if args.generation_retries < 1:
        parser.error("--generation_retries must be >= 1")
    return args


def load_safetensors_incrementally(
    model: torch.nn.Module, path: Path
) -> tuple[list[str], list[str]]:
    """Copy checkpoint tensors one at a time without a full state-dict copy."""
    destinations = model.state_dict(keep_vars=True)
    with safe_open(str(path), framework="pt", device="cpu") as source:
        source_keys = set(source.keys())
        destination_keys = set(destinations)
        with torch.no_grad():
            for key in sorted(source_keys & destination_keys):
                value = source.get_tensor(key)
                destination = destinations[key]
                if destination.shape != value.shape:
                    raise RuntimeError(
                        f"Checkpoint tensor {key!r} has shape "
                        f"{tuple(value.shape)}, expected "
                        f"{tuple(destination.shape)}."
                    )
                destination.copy_(value)
                del value
    missing = sorted(destination_keys - source_keys)
    unexpected = sorted(source_keys - destination_keys)
    return missing, unexpected


@torch.no_grad()
def encode_text_conditioning_low_host_memory(
    texts: list[str],
    *,
    device: torch.device,
    cache_dir: Path,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode both SDXL text streams before allocating the diffusion model."""
    from transformers import (
        AutoTokenizer,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
    )

    common_load_kwargs = {
        "low_cpu_mem_usage": True,
        # Preserve the original text-encoder weights. Autocast below matches
        # the former per-image execution precision without a host-side copy.
        "torch_dtype": torch.float32,
        "device_map": {"": str(device)},
        "cache_dir": str(cache_dir),
    }

    clip_l_id = "openai/clip-vit-large-patch14"
    clip_l_tokenizer = CLIPTokenizer.from_pretrained(
        clip_l_id, cache_dir=str(cache_dir)
    )
    clip_l = (
        CLIPTextModel.from_pretrained(clip_l_id, **common_load_kwargs)
        .eval()
        .requires_grad_(False)
    )
    clip_l_chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), 8):
        clip_l_inputs = clip_l_tokenizer(
            texts[start : start + 8],
            truncation=True,
            max_length=77,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = clip_l_inputs["input_ids"].to(device)
        with autocast_context(precision):
            clip_l_output = clip_l(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        # SDXL's FrozenCLIPEmbedder(layer="hidden", layer_idx=11).
        clip_l_chunks.append(clip_l_output.hidden_states[11].detach().cpu())
        del clip_l_output, clip_l_inputs, input_ids
    clip_l_tokens = torch.cat(clip_l_chunks)
    del clip_l_chunks, clip_l, clip_l_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    bigg_id = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    bigg_tokenizer = AutoTokenizer.from_pretrained(bigg_id, cache_dir=str(cache_dir))
    bigg = (
        CLIPTextModelWithProjection.from_pretrained(bigg_id, **common_load_kwargs)
        .eval()
        .requires_grad_(False)
    )
    crossattn_chunks: list[torch.Tensor] = []
    pooled_chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), 8):
        bigg_inputs = bigg_tokenizer(
            texts[start : start + 8],
            truncation=True,
            max_length=77,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = bigg_inputs["input_ids"].to(device)
        with autocast_context(precision):
            bigg_output = bigg(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        # Match FrozenOpenCLIPEmbedder2(layer="penultimate"): stop before
        # the last transformer block, then apply the final text LayerNorm.
        bigg_tokens = bigg.text_model.final_layer_norm(bigg_output.hidden_states[-2])
        clip_l_batch = clip_l_tokens[start : start + len(input_ids)].to(device)
        crossattn_chunks.append(
            torch.cat((clip_l_batch, bigg_tokens), dim=-1).detach().cpu()
        )
        pooled_chunks.append(bigg_output.text_embeds.detach().cpu())
        del bigg_output, bigg_inputs, input_ids, bigg_tokens, clip_l_batch
    crossattn = torch.cat(crossattn_chunks)
    pooled = torch.cat(pooled_chunks)
    del crossattn_chunks, pooled_chunks, bigg, bigg_tokenizer, clip_l_tokens
    gc.collect()
    torch.cuda.empty_cache()
    return crossattn, pooled


def prepare_models(
    args: argparse.Namespace,
    device: torch.device,
    captions: list[str],
):
    from omegaconf import OmegaConf
    from sgm.models.diffusion import DiffusionEngine

    unclip_config = OmegaConf.load(
        PROJECT_ROOT / "generative_models" / "configs" / "unclip6.yaml"
    )
    unclip_params = OmegaConf.to_container(unclip_config, resolve=True)["model"][
        "params"
    ]
    sampler_config = unclip_params["sampler_config"]
    sampler_config["params"]["num_steps"] = args.sampler_steps

    base_config = OmegaConf.load(
        PROJECT_ROOT / "generative_models" / "configs" / "inference" / "sd_xl_base.yaml"
    )
    refiner_params = OmegaConf.to_container(base_config, resolve=True)["model"][
        "params"
    ]
    full_conditioner_config = refiner_params["conditioner_config"]
    precision = resolve_precision(args.precision)
    negative_prompt = (
        "painting, extra fingers, mutated hands, poorly drawn hands, poorly "
        "drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, "
        "extra limbs, cloned face, glitchy, distorted face, anime"
    )
    text_values = list(captions) + [negative_prompt]
    text_crossattn, text_pooled = encode_text_conditioning_low_host_memory(
        text_values,
        device=device,
        cache_dir=Path(args.weights_dir).expanduser().resolve(),
        precision=precision,
    )

    # Text is already encoded above. Keep only the three lightweight size/
    # crop embedders in the engine conditioner so no text tower coexists with
    # the SDXL network in host or device memory.
    conditioner_config = {
        **full_conditioner_config,
        "params": {
            **full_conditioner_config["params"],
            "emb_models": full_conditioner_config["params"]["emb_models"][2:],
            "staged_device": str(device),
        },
    }
    checkpoint_path = (
        Path(args.weights_dir).expanduser().resolve() / args.base_checkpoint
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing SDXL enhancement checkpoint: {checkpoint_path}"
        )
    base_engine = (
        DiffusionEngine(
            network_config=refiner_params["network_config"],
            denoiser_config=refiner_params["denoiser_config"],
            first_stage_config=refiner_params["first_stage_config"],
            conditioner_config=conditioner_config,
            sampler_config=sampler_config,
            scale_factor=refiner_params["scale_factor"],
            disable_first_stage_autocast=refiner_params["disable_first_stage_autocast"],
            construction_device=device,
        )
        .eval()
        .requires_grad_(False)
        .to(device)
    )
    missing, unexpected = load_safetensors_incrementally(base_engine, checkpoint_path)
    print(
        f"Restored from {checkpoint_path} with {len(missing)} missing and "
        f"{len(unexpected)} unexpected keys"
    )
    if missing:
        print(f"Missing Keys: {missing}")
    if unexpected:
        print(f"Unexpected Keys: {unexpected}")

    common = {
        "original_size_as_tuple": torch.ones(1, 2, device=device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2, device=device),
        "target_size_as_tuple": torch.ones(1, 2, device=device) * 1024,
    }
    geometry = base_engine.conditioner(common)["vector"].detach().cpu()
    caption_crossattn = text_crossattn[:-1]
    caption_vectors = torch.cat(
        (text_pooled[:-1], geometry.expand(len(captions), -1)), dim=-1
    )
    crossattn_uc = text_crossattn[-1:].to(device)
    vector_uc = torch.cat((text_pooled[-1:], geometry), dim=-1).to(device)
    return (
        base_engine,
        caption_crossattn,
        caption_vectors,
        crossattn_uc,
        vector_uc,
    )


def normalize_captions(value, expected: int) -> list[str]:
    if isinstance(value, np.ndarray):
        captions = value.tolist()
    elif isinstance(value, (list, tuple)):
        captions = list(value)
    else:
        raise ValueError(f"Unsupported caption payload type: {type(value).__name__}")
    captions = [str(caption) for caption in captions]
    if len(captions) != expected:
        raise ValueError(f"Caption/image count mismatch: {len(captions)} vs {expected}")
    return captions


def save_preview(sample: torch.Tensor, gt: torch.Tensor, path: Path) -> None:
    sample_np = sample.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    gt = F.interpolate(
        gt[None],
        size=sample.shape[-2:],
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]
    gt_np = gt.clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    preview = cv2.cvtColor(
        np.concatenate((sample_np, gt_np), axis=0), cv2.COLOR_RGB2BGR
    )
    cv2.imwrite(str(path), preview)


@torch.no_grad()
def enhance(
    args: argparse.Namespace,
    device: torch.device,
    indices: list[int],
) -> torch.Tensor:
    input_dir = Path(args.input_dir).expanduser().resolve()
    all_gts = torch.load(input_dir / args.gts_file, map_location="cpu").float()
    all_recons = torch.load(input_dir / args.recons_file, map_location="cpu").float()
    if all_gts.shape != all_recons.shape or all_recons.ndim != 4:
        raise ValueError(
            f"Invalid aligned image tensors: {all_recons.shape}, {all_gts.shape}"
        )
    if not torch.isfinite(all_recons).all():
        bad = (
            torch.nonzero(~torch.isfinite(all_recons).flatten(1).all(dim=1))
            .flatten()
            .tolist()
        )
        raise ValueError(
            "Base reconstructions are non-finite at indices "
            f"{bad[:30]}; rerun stage 3 before enhancement."
        )
    captions = normalize_captions(
        torch.load(input_dir / args.captions_file, map_location="cpu"),
        len(all_recons),
    )
    (
        engine,
        caption_crossattn,
        caption_vectors,
        crossattn_uc,
        vector_uc,
    ) = prepare_models(args, device, captions)
    engine.sampler.guider.scale = args.cfg_scale
    preview_dir = input_dir / "enhanced_previews"
    if args.save_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)

    def denoiser(x, sigma, conditioning):
        return engine.denoiser(engine.model, x, sigma, conditioning)

    precision = resolve_precision(args.precision)

    def generate_one(index: int) -> torch.Tensor:
        with autocast_context(precision), engine.ema_scope():
            engine.sampler.num_steps = args.sampler_steps
            image = F.interpolate(
                all_recons[[index]],
                size=(768, 768),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).to(device)
            latent = engine.encode_first_stage(image * 2 - 1)
            conditioning = {
                "crossattn": caption_crossattn[[index]].to(device),
                "vector": caption_vectors[[index]].to(device),
            }
            unconditional = {
                "crossattn": crossattn_uc,
                "vector": vector_uc,
            }
            noise = torch.randn_like(latent)
            sigmas = engine.sampler.discretization(args.sampler_steps).to(device)
            sigmas = sigmas[-args.img2img_timepoint :]
            # Keep the selected tail of the original schedule. Calling
            # prepare_sampling_loop here would recompute a different schedule
            # from its shorter length and rescale the latent inconsistently.
            current = latent + noise * append_dims(sigmas[0], latent.ndim)
            for timestep in range(len(sigmas) - 1):
                current = engine.sampler.sampler_step(
                    sigmas[timestep].expand(latent.shape[0]),
                    sigmas[timestep + 1].expand(latent.shape[0]),
                    denoiser,
                    current,
                    cond=conditioning,
                    uc=unconditional,
                    gamma=0,
                )
            return (
                torch.clamp(
                    (engine.decode_first_stage(current) + 1.0) / 2.0,
                    min=0.0,
                    max=1.0,
                )[0]
                .detach()
                .float()
                .cpu()
            )

    enhanced: list[torch.Tensor] = []
    for index in tqdm(indices, desc=f"Enhance rank {device.index}"):
        sample = None
        for attempt in range(1, args.generation_retries + 1):
            utils.seed_everything(args.seed + index * 1009 + attempt - 1)
            candidate = generate_one(index)
            if torch.isfinite(candidate).all():
                sample = candidate
                break
            print(
                f"[WARN] Enhancement index {index} was non-finite, "
                f"attempt {attempt}/{args.generation_retries}."
            )
        if sample is None:
            raise FloatingPointError(
                f"Enhancement index {index} remained non-finite after "
                f"{args.generation_retries} attempts at precision={precision}."
            )
        sample = F.interpolate(
            sample[None],
            size=(args.output_size, args.output_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        enhanced.append(sample)
        if args.save_previews:
            save_preview(sample, all_gts[index], preview_dir / f"frame_{index:04d}.jpg")
    return torch.stack(enhanced).float()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("SDXL enhancement requires a CUDA GPU.")
    utils.seed_everything(args.seed + accelerator.process_index)
    input_dir = Path(args.input_dir).expanduser().resolve()
    recons = torch.load(input_dir / args.recons_file, map_location="cpu")
    source_count = len(recons)
    local_indices = list(
        range(
            accelerator.process_index,
            len(recons),
            accelerator.num_processes,
        )
    )
    del recons
    if not local_indices:
        raise RuntimeError(
            "More distributed processes were requested than reconstructions."
        )
    result = enhance(args, device, local_indices)
    shard_dir = input_dir / ".stage6_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"indices": local_indices, "enhanced": result},
        shard_dir
        / (
            f"rank_{accelerator.process_index:03d}_of_"
            f"{accelerator.num_processes:03d}.pt"
        ),
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final = merge_enhancement_shards(
            shard_dir, accelerator.num_processes, source_count
        )
        output_path = input_dir / args.output_file
        torch.save(final, output_path)
        print(
            f"Saved enhanced reconstructions {tuple(final.shape)} " f"to {output_path}"
        )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
