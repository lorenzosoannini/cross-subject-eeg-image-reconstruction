from __future__ import annotations
import contextlib
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare_unclip(
    weights_dir: Path,
    device: torch.device,
    sampler_steps: int,
    *,
    include_openclip: bool = True,
    model_dtype: torch.dtype | None = None,
    low_host_memory: bool = False,
) -> DiffusionEngine:
    from omegaconf import OmegaConf
    from sgm.models.diffusion import DiffusionEngine

    config = OmegaConf.load(
        PROJECT_ROOT / "generative_models" / "configs" / "unclip6.yaml"
    )
    config = OmegaConf.to_container(config, resolve=True)
    params = config["model"]["params"]
    first_stage_config = params["first_stage_config"]
    first_stage_config["target"] = "sgm.models.autoencoder.AutoencoderKL"
    sampler_config = params["sampler_config"]
    sampler_config["params"]["num_steps"] = sampler_steps
    if not include_openclip:
        # Reconstruction already receives cross-attention image tokens via
        # utils.unclip_recon().  Keeping the frozen ViT-bigG embedder here is
        # therefore unnecessary once those tokens have been computed, and
        # costs roughly another 10 GB of host memory.
        emb_models = params["conditioner_config"]["params"]["emb_models"]
        openclip_target = "sgm.modules.encoders.modules.FrozenOpenCLIPImageEmbedder"
        params["conditioner_config"]["params"]["emb_models"] = [
            config for config in emb_models if config.get("target") != openclip_target
        ]
    target_dtype = model_dtype or torch.float32
    if low_host_memory:
        print(
            "[unCLIP] low-host-memory construction on "
            f"{device} with dtype={target_dtype}"
        )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(target_dtype)
    try:
        device_context = (
            torch.device(device) if low_host_memory else contextlib.nullcontext()
        )
        with device_context:
            engine = DiffusionEngine(
                network_config=params["network_config"],
                denoiser_config=params["denoiser_config"],
                first_stage_config=first_stage_config,
                conditioner_config=params["conditioner_config"],
                sampler_config=sampler_config,
                scale_factor=params["scale_factor"],
                disable_first_stage_autocast=params["disable_first_stage_autocast"],
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    checkpoint_path = weights_dir / "unclip6_epoch0_step110000.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing SDXL unCLIP checkpoint: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if include_openclip:
        engine.load_state_dict(payload["state_dict"], strict=True)
    else:
        incompatible = engine.load_state_dict(payload["state_dict"], strict=False)
        invalid_missing = [
            key for key in incompatible.missing_keys if key != "denoiser.sigmas"
        ]
        invalid_unexpected = [
            key
            for key in incompatible.unexpected_keys
            if not key.startswith("conditioner.embedders.0.")
        ]
        if invalid_missing or invalid_unexpected:
            raise RuntimeError(
                "Unexpected unCLIP state mismatch without OpenCLIP: "
                f"missing={invalid_missing[:10]}, "
                f"unexpected={invalid_unexpected[:10]}"
            )
    del payload
    return engine.eval().requires_grad_(False).to(device=device, dtype=target_dtype)


def prepare_vector_suffix(
    engine: DiffusionEngine, device: torch.device
) -> torch.Tensor:
    batch = {
        "jpg": torch.randn(1, 3, 1, 1, device=device),
        "original_size_as_tuple": torch.ones(1, 2, device=device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2, device=device),
    }
    return engine.conditioner(batch)["vector"].to(device)
