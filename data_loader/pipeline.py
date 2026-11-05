"""Explicit source-only training datasets and EEG-only evaluation inputs."""

from .images import load_image
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .protocol import SOURCES, HELDOUT, validate_partitions
from .things_eeg2_reve_dataset import ThingsEEG2ReveDataset


def concept_rows(config):
    with (Path(config["index_dir"]) / "visuals.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = {
        r["concept_uid"]: r["concept_name"]
        for r in rows
        if r["partition"] == "training"
    }
    uids = sorted(names)
    if config["categories"] == 100:
        uids = sorted(
            np.random.default_rng(42).choice(uids, 100, replace=False).tolist()
        )
    elif config["categories"] != 1654:
        raise ValueError("Use 100 development or 1654 final categories")
    return (
        rows,
        uids,
        [names[uid].split(" ", 1)[-1].replace("_", " ") for uid in uids],
    )


def dataset(config, split, subject=None, training_images=False):
    validate_partitions(config["index_dir"])
    (rows, uids, names) = concept_rows(config)
    if split in ("train", "val"):
        if subject is not None:
            raise ValueError("Training/development always uses source subjects only")
        subjects = SOURCES
    elif split == "evaluation":
        if subject not in (HELDOUT, 4):
            raise ValueError("Evaluation subjects: 10 heldout or 4 source reference")
        subjects = [subject]
    else:
        raise ValueError("Unknown study split")
    base = ThingsEEG2ReveDataset(
        root=config["data_root"],
        clip_cache=config["clip_cache"],
        split=split,
        index_dir=config["index_dir"],
        stats_root=config["statistics_dir"],
        subjects=subjects,
        allowed_concept_uids=uids,
        return_eeg_target=True,
        return_clip_image=split != "evaluation",
    )
    return (
        TrainingImages(base, rows, uids, config["data_root"])
        if training_images
        else base
    )


class TrainingImages(Dataset):

    def __init__(self, base, rows, uids, root):
        self.base = base
        self.labels = {u: i for (i, u) in enumerate(uids)}
        self.paths = {
            int(r["visual_index"]): Path(root) / r["image_path"] for r in rows
        }

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        sample = self.base[i]
        sample["image"] = load_image(self.paths[sample["visual_index"]])
        sample["concept_label"] = self.labels[sample["concept_uid"]]
        return sample
