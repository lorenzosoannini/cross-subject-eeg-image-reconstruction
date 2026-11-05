"""Mean F_s across repeated EEG responses, then prior and unCLIP decoding."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "generative_models")]
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader.images import load_image
from data_loader.pipeline import dataset, concept_rows
from data_loader.protocol import digest
from model.network import SemanticGenerator
from model.aggregation import aggregate_conditions
from train import read_config, check_checkpoint, provenance, amp
from eval.engine import prepare_unclip, prepare_vector_suffix
import utils


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.json")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--subject", type=int, choices=[4, 10], default=10)
    a = p.parse_args()
    config = read_config(a.config)
    if not torch.cuda.is_available():
        raise RuntimeError("unCLIP generation requires CUDA")
    device = torch.device("cuda")
    utils.seed_everything(config["seed"])
    if a.output_dir.exists():
        raise FileExistsError("Use a fresh generation directory")
    a.output_dir.mkdir(parents=True)
    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    check_checkpoint(payload, config, "downstream")
    if payload["epoch"] != config["epochs"]:
        raise ValueError("Use the final downstream checkpoint")
    model = SemanticGenerator(
        Path(config["index_dir"]) / "reve_positions/things_eeg2_63_positions.npy"
    )
    model.load_downstream_state(payload["model_state_dict"])
    del payload
    model.to(device).eval()
    eeg_data = dataset(config, "evaluation", subject=a.subject)
    loader = DataLoader(
        eeg_data, batch_size=16, shuffle=False, num_workers=config["num_workers"]
    )
    (conditions, counts, events) = aggregate_conditions(model, loader, device, config)
    ordered = sorted(conditions)
    tokens = {}
    for start in range(0, len(ordered), 4):
        keys = ordered[start : start + 4]
        utils.seed_everything(config["seed"] + keys[0] * 1009)
        fs = torch.stack([conditions[k] for k in keys]).to(device)
        with amp(config, device):
            pred = model.diffusion_prior.p_sample_loop_ddim(
                fs.shape, {"text_embed": fs}, timesteps=20, cond_scale=1.0
            )
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Non-finite prior tokens")
        for i, key in enumerate(keys):
            tokens[key] = pred[i].float().cpu()
    del model, conditions
    torch.cuda.empty_cache()
    engine = prepare_unclip(
        Path(config["weights_dir"]),
        device,
        38,
        include_openclip=False,
        model_dtype=torch.bfloat16 if config["precision"] == "bf16" else torch.float32,
        low_host_memory=True,
    )
    suffix = prepare_vector_suffix(engine, device)
    recons = []
    for key in ordered:
        utils.seed_everything(config["seed"] + key * 1009)
        sample = (
            utils.unclip_recon(
                tokens[key][None].to(device),
                engine,
                suffix,
                device=device,
                autocast_dtype=(
                    torch.bfloat16 if config["precision"] == "bf16" else None
                ),
            )[0]
            .float()
            .cpu()
        )
        sample = F.interpolate(
            sample[None],
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        if not torch.isfinite(sample).all():
            raise FloatingPointError(f"Non-finite reconstruction {key}")
        recons.append(sample)
    torch.save(torch.stack(recons), a.output_dir / "all_recons.pt")
    (rows, _, _) = concept_rows(config)
    by_id = {int(r["visual_index"]): r for r in rows}
    targets = []
    records = []
    for key in ordered:
        row = by_id[key]
        target = load_image(Path(config["data_root"]) / row["image_path"])
        target = F.interpolate(
            target[None],
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0].clamp(0, 1)
        targets.append(target)
        records.append(
            dict(
                visual_index=key,
                visual_uid=row["visual_uid"],
                subject=a.subject,
                repetition_count=counts[key],
                image_path=row["image_path"],
            )
        )
    torch.save(torch.stack(targets), a.output_dir / "all_gts.pt")
    with (a.output_dir / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "provenance": provenance(config),
        "checkpoint_sha256": digest(a.checkpoint),
        "subject": a.subject,
        "split": "evaluation",
        "repetition_strategy": "embedding_mean",
        "condition": "F_s from E_i",
        "visual_indices": ordered,
        "repetition_counts": counts,
        "retained_trial_rows": events,
        "prior_sampling_steps": 20,
        "prior_condition_scale": 1.0,
        "unclip_steps": 38,
        "seed": config["seed"],
        "precision": config["precision"],
        "encoder_batch_size": 16,
        "prior_batch_size": 4,
    }
    (a.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
