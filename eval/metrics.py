from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from eval.eval_metrics import (
    cal_alexnet,
    cal_clip,
    cal_efficientnet,
    cal_inceptionv3,
    cal_pixcorr,
    cal_ssim,
    cal_swav,
)


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def load_image_tensor(
    path: Path,
    label: str,
    *,
    indices: torch.Tensor | None = None,
    expected_source_length: int | None = None,
) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label} tensor: {path}")
    try:
        value = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        try:
            value = torch.load(path, map_location="cpu", mmap=True)
        except TypeError:
            value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise ValueError(
            f"{label} must be a [N,C,H,W] tensor; got "
            f"{type(value).__name__} {getattr(value, 'shape', None)}"
        )
    if value.shape[1] != 3:
        raise ValueError(f"{label} must have three RGB channels: {value.shape}")
    if expected_source_length is not None and len(value) != expected_source_length:
        raise ValueError(
            f"{label}/sample-map length mismatch: "
            f"{len(value)} != {expected_source_length}."
        )
    if indices is not None:
        value = value.index_select(0, indices)
    if not torch.isfinite(value).all():
        finite = torch.isfinite(value)
        bad_images = torch.nonzero(~finite.flatten(1).all(dim=1)).flatten().tolist()
        nan_count = int(torch.isnan(value).sum().item())
        positive_inf = int(torch.isposinf(value).sum().item())
        negative_inf = int(torch.isneginf(value).sum().item())
        raise ValueError(
            f"{label} contains non-finite values in {len(bad_images)}/"
            f"{len(value)} images; indices={bad_images[:30]}, "
            f"NaN={nan_count}, +Inf={positive_inf}, -Inf={negative_inf}. "
            "Do not replace them before evaluation: regenerate the affected "
            "reconstructions with the numerically guarded stage 3/6."
        )
    minimum, maximum = float(value.min()), float(value.max())
    if minimum < -1e-4 or maximum > 1.0001:
        raise ValueError(f"{label} must be in [0,1], found [{minimum}, {maximum}].")
    return value.float()


def evaluate(
    all_recons: torch.Tensor,
    all_gts: torch.Tensor,
    feature_batch_size: int = 32,
) -> dict[str, float]:
    if all_recons.shape != all_gts.shape:
        raise ValueError(
            "Reconstruction/ground-truth tensors must have identical shape: "
            f"{all_recons.shape} vs {all_gts.shape}."
        )
    results: dict[str, float] = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("-------- Low-level --------")
    results["pixcorr"] = _float(
        cal_pixcorr(all_recons, all_gts, batch_size=feature_batch_size)
    )
    print(f"{'PixCorr':<15}: {results['pixcorr']:.4f}")
    results["ssim"] = _float(
        cal_ssim(all_recons, all_gts, batch_size=feature_batch_size)
    )
    print(f"{'SSIM':<15}: {results['ssim']:.4f}")
    alexnet_2, alexnet_5 = cal_alexnet(
        all_recons,
        all_gts,
        device=device,
        batch_size=feature_batch_size,
    )
    results["alexnet_2"] = _float(alexnet_2)
    results["alexnet_5"] = _float(alexnet_5)
    print(f"{'Alex(2)':<15}: {results['alexnet_2'] * 100:.2f}")
    print(f"{'Alex(5)':<15}: {results['alexnet_5'] * 100:.2f}")

    print("-------- High-level --------")
    results["inception_v3"] = _float(
        cal_inceptionv3(
            all_recons,
            all_gts,
            device=device,
            batch_size=feature_batch_size,
        )
    )
    print(f"{'InceptionV3':<15}: {results['inception_v3'] * 100:.2f}")
    results["clip"] = _float(
        cal_clip(
            all_recons,
            all_gts,
            device=device,
            batch_size=feature_batch_size,
        )
    )
    print(f"{'CLIP':<15}: {results['clip'] * 100:.2f}")
    results["efficientnet"] = _float(
        cal_efficientnet(
            all_recons,
            all_gts,
            device=device,
            batch_size=feature_batch_size,
        )
    )
    print(f"{'EfficientNet':<15}: {results['efficientnet']:.3f}")
    results["swav"] = _float(
        cal_swav(
            all_recons,
            all_gts,
            device=device,
            batch_size=feature_batch_size,
        )
    )
    print(f"{'SwAV':<15}: {results['swav']:.3f}")
    return results


def main():
    p = argparse.ArgumentParser(
        description="Evaluate every paired image in a completed export"
    )
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--mode", choices=["base", "enhance"], default="base")
    a = p.parse_args()
    with (a.input_dir / "samples.csv").open(newline="") as handle:
        samples = list(csv.DictReader(handle))
    metadata = json.loads((a.input_dir / "metadata.json").read_text())
    if [int(s["visual_index"]) for s in samples] != metadata["visual_indices"]:
        raise ValueError("Pairing metadata mismatch")
    recons = load_image_tensor(
        a.input_dir
        / ("all_recons.pt" if a.mode == "base" else "all_enhanced_recons.pt"),
        "reconstruction",
        expected_source_length=len(samples),
    )
    targets = load_image_tensor(
        a.input_dir / "all_gts.pt", "ground truth", expected_source_length=len(samples)
    )
    result = evaluate(recons, targets)
    result["n_images"] = len(samples)
    (a.input_dir / f"metrics_{a.mode}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
