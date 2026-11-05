#!/usr/bin/env python3
"""Caption reconstructed THINGS-EEG2 images with BLIP-2."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor

MODEL_ID = "Salesforce/blip2-opt-2.7b"
MODEL_REVISION = "ec3c346f4aff99452053bae7abd1753401e6f28a"


def merge_caption_shards(
    shard_dir: Path, num_processes: int, total_images: int
) -> list[str]:
    merged: list[str | None] = [None] * total_images
    for rank in range(num_processes):
        path = shard_dir / f"rank_{rank:03d}_of_{num_processes:03d}.pt"
        payload = torch.load(path, map_location="cpu")
        if len(payload["indices"]) != len(payload["captions"]):
            raise RuntimeError(f"Misaligned BLIP-2 shard: {path}")
        for index, caption in zip(payload["indices"], payload["captions"]):
            index = int(index)
            if not 0 <= index < total_images:
                raise RuntimeError(f"Caption index out of range: {index}")
            if merged[index] is not None:
                raise RuntimeError(f"Duplicate caption index {index}.")
            merged[index] = str(caption)
    if any(caption is None for caption in merged):
        raise RuntimeError("Distributed BLIP-2 caption merge is incomplete.")
    return [str(caption) for caption in merged]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", default="pred_test_caption.pt")
    parser.add_argument("--cache_dir", default="pretrained_weights")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("--batch_size and --max_new_tokens must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    input_dir = Path(args.input_dir).expanduser().resolve()
    images_path = input_dir / "all_recons.pt"
    output_path = input_dir / args.output_file
    if not images_path.is_file():
        raise FileNotFoundError(f"Missing reconstruction tensor: {images_path}")
    images = torch.load(images_path, map_location="cpu")
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W] images, got {type(images)}")

    device = accelerator.device
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    processor = Blip2Processor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval().requires_grad_(False)
    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        raise ValueError(
            "The BLIP-2 checkpoint has no image token index; use the "
            "configured revision compatible with this Transformers release."
        )
    # Some Transformers releases do not propagate the OPT start token into
    # the nested BLIP-2 text config.
    if model.config.text_config.bos_token_id is None:
        bos_token_id = processor.tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = model.generation_config.bos_token_id
        if bos_token_id is None:
            raise ValueError("BLIP-2 has no text generation start token.")
        model.config.text_config.bos_token_id = int(bos_token_id)

    local_indices = list(
        range(accelerator.process_index, len(images), accelerator.num_processes)
    )
    if not local_indices:
        raise RuntimeError(
            "More distributed processes were requested than images to caption."
        )
    captions: list[str] = []
    for start in tqdm(
        range(0, len(local_indices), args.batch_size),
        desc=f"Caption BLIP-2 rank {accelerator.process_index}",
    ):
        batch_indices = local_indices[start : start + args.batch_size]
        batch = [to_pil_image(images[index].clamp(0, 1)) for index in batch_indices]
        inputs = processor(
            images=batch,
            text=[""] * len(batch),
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs.get("input_ids")
        if input_ids is None or input_ids.shape[1] == 0:
            raise ValueError("BLIP-2 processor produced no text input IDs.")
        image_token_counts = (input_ids == int(image_token_id)).sum(dim=1)
        if not torch.all(image_token_counts == model.config.num_query_tokens):
            raise ValueError(
                "BLIP-2 processor/model image-token configuration differs: "
                f"counts={image_token_counts.tolist()}, expected="
                f"{model.config.num_query_tokens}."
            )
        inputs = {
            key: (
                value.to(device, dtype=dtype)
                if value.is_floating_point()
                else value.to(device)
            )
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        captions.extend(
            text.strip()
            for text in processor.batch_decode(generated_ids, skip_special_tokens=True)
        )
    if len(captions) != len(local_indices):
        raise RuntimeError(
            "Local caption/image count mismatch: "
            f"{len(captions)} vs {len(local_indices)}"
        )
    shard_dir = input_dir / ".stage5_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"indices": local_indices, "captions": captions},
        shard_dir
        / (
            f"rank_{accelerator.process_index:03d}_of_"
            f"{accelerator.num_processes:03d}.pt"
        ),
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_captions = merge_caption_shards(
            shard_dir, accelerator.num_processes, len(images)
        )
        torch.save(final_captions, output_path)
        (input_dir / "pred_test_caption.txt").write_text(
            "\n".join(final_captions) + "\n", encoding="utf-8"
        )
        print(f"Saved {len(final_captions)} captions to {output_path}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
