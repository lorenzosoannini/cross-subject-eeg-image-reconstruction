"""Source-only representation pretraining and F_s-conditioned downstream fitting."""

from __future__ import annotations
import argparse
import contextlib
import json
import importlib.metadata
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "generative_models"))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader.pipeline import dataset, concept_rows
from data_loader.protocol import statistics_manifest, digest
from data_loader.things_eeg2_reve_dataset import masked_reconstruction_mse
from model.network import EEGModel, ReconstructionModel
from model.dsn_losses import (
    latent_reconstruction_loss,
    cross_covariance_loss,
    variance_floor_loss,
)
from model.encoder import LoRALinear
from model.supervision import (
    build_frozen_autoencoder,
    build_text_targets,
    multi_positive_token_clip_loss,
)
import utils

PROTOCOL = "eeg-fs-source-only-8-1-1-v1"


def read_config(filename):
    config = json.loads(Path(filename).read_text())
    if config["precision"] not in ("fp32", "bf16"):
        raise ValueError("Supported precision: fp32 or bf16")
    if config["pretraining_epoch"] != 50:
        raise ValueError("Reference pretraining selection is epoch 50")
    return config


def amp(config, device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and config["precision"] == "bf16"
        else contextlib.nullcontext()
    )


def provenance(config):
    return {
        "protocol": PROTOCOL,
        "index_hashes": statistics_manifest(config["index_dir"]),
        "statistics_hash": digest(
            Path(config["statistics_dir"]) / "source_statistics.npz"
        ),
        "statistics_metadata_hash": digest(
            Path(config["statistics_dir"]) / "source_statistics.json"
        ),
    }


def check_checkpoint(payload, config, stage):
    if payload.get("stage") != stage or payload.get("provenance") != provenance(config):
        raise ValueError(
            "Checkpoint stage/protocol/index/statistics mismatch; historical checkpoints need retraining"
        )
    if payload["config"]["categories"] != config["categories"]:
        raise ValueError("Checkpoint category mismatch")


def build_representation(config):
    return EEGModel(
        Path(config["index_dir"]) / "reve_positions/things_eeg2_63_positions.npy"
    )


def alignment(prediction, target, ids, permutation=None, beta=None):
    if permutation is None:
        return multi_positive_token_clip_loss(prediction, target, ids, 0.006)
    p = F.normalize(prediction.float().flatten(1), dim=-1)
    t = F.normalize(target.float().flatten(1), dim=-1)
    positive = (ids[:, None] == ids[None, :]).float()
    positive /= positive.sum(dim=1, keepdim=True)
    weights = beta[:, None] * positive + (1 - beta[:, None]) * positive[permutation]
    logits = p @ t.T / 0.006
    reverse = weights.T / weights.T.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return -0.5 * (
        (weights * logits.log_softmax(dim=1)).sum(dim=1).mean()
        + (reverse * logits.T.log_softmax(dim=1)).sum(dim=1).mean()
    )


def objective(model, batch, config, stage, device, progress, training, extras):
    eeg = batch["eeg"].to(device)
    target = batch["clip_image"].to(device).float()
    ids = batch["visual_index"].to(device)
    if stage == "pretrain":
        subject = batch["subject_id"].to(device) - 1
        mask = batch["valid_mask"].to(device)
        perm = beta = None
        if training:
            perm = torch.randperm(len(eeg), device=device)
            beta = torch.distributions.Beta(0.15, 0.15).sample((len(eeg),)).to(device)
            beta = torch.where(
                torch.rand(len(eeg), device=device) < 0.5, beta, torch.ones_like(beta)
            )
            eeg = beta[:, None, None] * eeg + (1 - beta[:, None, None]) * eeg[perm]
            mask = mask | (mask[perm] & (beta[:, None] < 1))
        out = model(eeg)

        def subject_loss(logits):
            ce = F.cross_entropy(logits.float(), subject, reduction="none")
            return (
                ce.mean()
                if perm is None
                else (
                    beta * ce
                    + (1 - beta)
                    * F.cross_entropy(logits.float(), subject[perm], reduction="none")
                ).mean()
            )

        losses = {
            "eeg": masked_reconstruction_mse(
                out["eeg_reconstruction"].float(), eeg.float(), mask
            ),
            "full_alignment": alignment(out["F"], target, ids, perm, beta),
            "semantic_alignment": alignment(out["F_s"], target, ids, perm, beta),
            "subject_classifier": subject_loss(out["subject_logits"]),
            "subject_discriminator": subject_loss(out["subject_adversarial_logits"]),
            "latent": latent_reconstruction_loss(
                out["latent_reconstruction"], out["E"]
            ),
            "covariance": cross_covariance_loss(out["z_i"], out["z_s"]),
            "variance": variance_floor_loss(out["z_i"], out["z_s"]),
        }
        weights = dict(config["pretraining_weights"])
        for name in ("latent", "covariance", "variance"):
            weights[name] *= min(
                1.0, progress / max(1, config["regularizer_warmup_epochs"])
            )
        weights["semantic_alignment"] *= min(
            1.0, progress / max(1, config["semantic_warmup_epochs"])
        )
    else:
        autoencoder, text_targets = extras
        concepts = batch["concept_label"].to(device)
        with torch.no_grad():
            image_latent = (
                autoencoder.encode(2 * batch["image"].to(device) - 1).latent_dist.mode()
                * 0.18215
            )
        out = model(eeg, target)
        onehot = F.one_hot(concepts, num_classes=len(text_targets)).float()
        losses = {
            "prior": out["prior_loss"],
            "full_alignment": alignment(out["F"], target, ids),
            "semantic_alignment": alignment(out["F_s"], target, ids),
            "image_classifier_all": F.binary_cross_entropy_with_logits(
                out["image_logits_all"].float(), onehot
            ),
            "image_classifier": F.binary_cross_entropy_with_logits(
                out["image_logits"].float(), onehot
            ),
            "image_discriminator": F.binary_cross_entropy_with_logits(
                out["image_adversarial_logits"].float(), onehot
            ),
            "text": multi_positive_token_clip_loss(
                out["text_prediction"], text_targets[concepts], concepts, 0.006
            ),
            "blurry": F.l1_loss(out["blurry_latent"].float(), image_latent.float()),
        }
        weights = config["downstream_weights"]
    weighted = {name: loss * weights[name] for name, loss in losses.items()}
    loss = sum(weighted.values())
    metrics = {
        **{name: float(value.detach()) for name, value in losses.items()},
        **{
            f"weighted/{name}": float(value.detach())
            for name, value in weighted.items()
        },
        "loss": float(loss.detach()),
    }
    return loss, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["pretrain", "downstream"])
    p.add_argument("--config", default="config.json")
    p.add_argument("--pretrain-checkpoint")
    p.add_argument("--wandb-project")
    a = p.parse_args()
    config = read_config(a.config)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU")
    device = torch.device("cuda")
    utils.seed_everything(config["seed"])
    output = Path(config["output_dir"]) / a.stage
    output.mkdir(parents=True, exist_ok=True)
    if (output / "history.jsonl").exists():
        raise FileExistsError("Choose a fresh output_dir for a new run")
    representation = build_representation(config)
    _, uids, names = concept_rows(config)
    extras = None
    if a.stage == "downstream":
        if not a.pretrain_checkpoint:
            p.error("--pretrain-checkpoint is required for downstream")
        payload = torch.load(
            a.pretrain_checkpoint, map_location="cpu", weights_only=False
        )
        check_checkpoint(payload, config, "pretrain")
        if payload["epoch"] != config["pretraining_epoch"]:
            raise ValueError("Use the selected epoch-50 pretraining checkpoint")
        representation.load_state_dict(payload["model_state_dict"], strict=True)
        model = ReconstructionModel(
            representation,
            len(uids),
            Path(config["weights_dir"]) / "coco_tokens_avg_proj.pth",
        )
        autoencoder = build_frozen_autoencoder(
            str(Path(config["weights_dir"]) / "sd_image_var_autoenc.pth"), device
        )
        targets = build_text_targets(
            names, config["weights_dir"], device, 32, "a photo of {}"
        )
        extras = (autoencoder, targets)
    else:
        model = representation
    model.to(device)
    batchsize = config[
        "pretraining_batch_size" if a.stage == "pretrain" else "downstream_batch_size"
    ]
    train_data = dataset(config, "train", training_images=a.stage == "downstream")
    val_data = dataset(config, "val", training_images=a.stage == "downstream")
    train_loader = DataLoader(
        train_data,
        batch_size=batchsize,
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=batchsize, shuffle=False, num_workers=config["num_workers"]
    )
    if len(train_loader) == 0:
        raise ValueError("Training loader has no full batch")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    warmup = max(1, config["warmup_epochs"] * len(train_loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.1 + 0.9 * min(1.0, step / warmup)
    )
    run = None
    if a.wandb_project:
        import wandb

        run = wandb.init(project=a.wandb_project, config=config)
    frozen_provenance = provenance(config)
    versions = {}
    for distribution in (
        "torch",
        "torchvision",
        "numpy",
        "transformers",
        "timm",
        "diffusers",
        "dalle2-pytorch",
    ):
        versions[distribution] = importlib.metadata.version(distribution)
    manifest = {
        "config": config,
        "software_versions": versions,
        "provenance": frozen_provenance,
        "stage": a.stage,
        "train_trials": len(train_data),
        "validation_trials": len(val_data),
        "concept_uids": uids,
        "pretrain_checkpoint": a.pretrain_checkpoint,
        "downstream_representation": "frozen EEG encoder and decomposition; trainable resampler/projectors",
        "implementation": "portfolio refactor; historical metrics not independently reproduced",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    global_step = 0
    for epoch in range(config["epochs"]):
        if a.stage == "pretrain":
            for module in model.eeg_encoder.reve.modules():
                if isinstance(module, LoRALinear):
                    module.set_lora_enabled(epoch >= config["reve_frozen_epochs"])
        summary = {"epoch": epoch + 1}
        for phase, loader in [("train", train_loader), ("validation", val_loader)]:
            model.train(phase == "train")
            totals = {}
            count = 0
            for i, batch in enumerate(loader):
                progress = epoch + (i + 1) / len(loader)
                with torch.set_grad_enabled(phase == "train"), amp(config, device):
                    loss, metrics = objective(
                        model,
                        batch,
                        config,
                        a.stage,
                        device,
                        progress,
                        phase == "train",
                        extras,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {phase} loss")
                if phase == "train":
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["gradient_clip"]
                    )
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                n = len(batch["eeg"])
                count += n
                for name, value in metrics.items():
                    totals[name] = totals.get(name, 0.0) + n * value
            if not count:
                raise ValueError("Empty loader")
            summary.update(
                {f"{phase}/{name}": value / count for name, value in totals.items()}
            )
        summary.update(step=global_step, learning_rate=optimizer.param_groups[0]["lr"])
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
        if run:
            run.log(summary, step=global_step)
        payload = {
            "stage": a.stage,
            "epoch": epoch + 1,
            "config": config,
            "provenance": frozen_provenance,
            "concept_uids": uids,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        destination = output / "last.pth"
        temporary = output / "last.partial.pth"
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        if a.stage == "pretrain" and epoch + 1 == config["pretraining_epoch"]:
            torch.save(payload, output / "epoch_050.pth")
    if run:
        run.finish()


if __name__ == "__main__":
    main()
