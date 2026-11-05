"""Frozen image partitions and source-only EEG normalization."""

from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

SOURCES = tuple(range(1, 10))
HELDOUT = 10


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def partition_images(rows, seed=42):
    groups = {}
    for row in rows:
        if row["partition"] == "training":
            groups.setdefault(row["concept_uid"], []).append(int(row["visual_index"]))
    splits = {name: [] for name in ("train", "val", "evaluation")}
    for concept, indices in sorted(groups.items()):
        if len(indices) != 10 or len(set(indices)) != 10:
            raise ValueError(f"Expected ten distinct images for {concept}")
        entropy = int(hashlib.sha256(concept.encode()).hexdigest()[:8], 16)
        order = np.random.default_rng(
            np.random.SeedSequence([seed, entropy])
        ).permutation(sorted(indices))
        splits["val"].append(int(order[0]))
        splits["evaluation"].append(int(order[1]))
        splits["train"].extend(map(int, order[2:]))
    return {name: sorted(values) for name, values in splits.items()}


def validate_partitions(index):
    index = Path(index)
    values = {
        s: set(np.load(index / "splits" / f"{s}_visual_indices.npy").tolist())
        for s in ("train", "val", "evaluation")
    }
    for a, b in [("train", "val"), ("train", "evaluation"), ("val", "evaluation")]:
        if values[a] & values[b]:
            raise ValueError(f"Image leakage: {a}/{b}")
    with (index / "visuals.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    concepts = {}
    for row in rows:
        if row["partition"] != "training":
            continue
        counts = concepts.setdefault(row["concept_uid"], [0, 0, 0])
        for i, s in enumerate(values):
            counts[i] += int(int(row["visual_index"]) in values[s])
    if not concepts or any(c != [8, 1, 1] for c in concepts.values()):
        raise ValueError("Every training concept must have exactly 8/1/1 photographs")
    known = {int(r["visual_index"]) for r in rows if r["partition"] == "training"}
    if set.union(*values.values()) != known:
        raise ValueError("Partitions must cover native-training photographs only")
    return values


class Moments:
    """Stable population moments, weighting every valid time sample equally."""

    def __init__(self, channels):
        self.count = 0
        self.mean = np.zeros(channels, dtype=np.float64)
        self.m2 = np.zeros(channels, dtype=np.float64)

    def update(self, epoch):
        epoch = np.asarray(epoch, dtype=np.float64)
        if (
            epoch.ndim != 2
            or epoch.shape[0] != len(self.mean)
            or not np.isfinite(epoch).all()
        ):
            raise ValueError("Invalid real EEG window")
        n = epoch.shape[1]
        if n == 0:
            raise ValueError("Empty EEG window")
        mean = epoch.mean(axis=1)
        delta = mean - self.mean
        total = self.count + n
        self.m2 += ((epoch - mean[:, None]) ** 2).sum(
            axis=1
        ) + delta**2 * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def result(self, epsilon=1e-6):
        if self.count == 0 or epsilon <= 0:
            raise ValueError("No samples or invalid epsilon")
        return self.mean.copy(), np.maximum(np.sqrt(self.m2 / self.count), epsilon)


def statistics_manifest(index):
    index = Path(index)
    return {
        name: digest(index / name)
        for name in [
            "trials.npy",
            "recordings.json",
            "visuals.csv",
            "splits/train_visual_indices.npy",
            "splits/val_visual_indices.npy",
            "splits/evaluation_visual_indices.npy",
        ]
    }


def fit_statistics(root, index, output, concept_uids=None):
    root, index, output = map(Path, (root, index, output))
    validate_partitions(index)
    train = np.load(index / "splits/train_visual_indices.npy")
    with (index / "visuals.csv").open(newline="") as handle:
        image_rows = list(csv.DictReader(handle))
    available = sorted(
        {r["concept_uid"] for r in image_rows if r["partition"] == "training"}
    )
    selected = available if concept_uids is None else sorted(concept_uids)
    if not set(selected) <= set(available):
        raise ValueError("Unknown normalization concepts")
    eligible = [
        int(r["visual_index"])
        for r in image_rows
        if r["concept_uid"] in selected and r["partition"] == "training"
    ]
    train = train[np.isin(train, eligible)]
    trials = np.load(index / "trials.npy", mmap_mode="r")
    recordings = json.loads((index / "recordings.json").read_text())
    channels = tuple(recordings[0]["channel_names"])
    moments = Moments(len(channels))
    kept = 0
    for recording in recordings:
        if recording["subject"] not in SOURCES or recording["partition"] != "training":
            continue
        if tuple(recording["channel_names"]) != channels:
            raise ValueError("Channel order mismatch")
        rows = trials[
            (trials["recording_id"] == recording["recording_id"])
            & np.isin(trials["visual_index"], train)
            & (trials["clean_valid_length"] >= 38)
        ]
        continuous = np.load(root / recording["eeg_path"], mmap_mode="r")
        for row in rows:
            start = int(row["onset_sample"])
            length = int(row["clean_valid_length"])
            if length > 40:
                raise ValueError("Invalid early window")
            moments.update(continuous[:, start : start + length])
            kept += 1
    mean, std = moments.result()
    metadata = {
        "concept_uids": selected,
        "source_subjects": list(SOURCES),
        "heldout_subject": HELDOUT,
        "window_mode": "clean_early",
        "minimum_length": 38,
        "epsilon": 1e-6,
        "ddof": 0,
        "weighting": "valid_time_samples",
        "sample_count": moments.count,
        "trial_count": kept,
        "index_hashes": statistics_manifest(index),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "source_statistics.npz", mean=mean, std=std, channel_names=channels
    )
    metadata["statistics_sha256"] = digest(output / "source_statistics.npz")
    (output / "source_statistics.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def load_statistics(folder, channels, index, concept_uids=None):
    folder = Path(folder)
    meta = json.loads((folder / "source_statistics.json").read_text())
    if concept_uids is not None and sorted(concept_uids) != meta["concept_uids"]:
        raise ValueError("Statistics category scope mismatch")
    if meta["source_subjects"] != list(SOURCES) or meta["heldout_subject"] != HELDOUT:
        raise ValueError("Statistics include an incompatible subject assignment")
    if meta["index_hashes"] != statistics_manifest(index):
        raise ValueError("Statistics/index provenance mismatch")
    if meta["statistics_sha256"] != digest(folder / "source_statistics.npz"):
        raise ValueError("Statistics hash mismatch")
    with np.load(folder / "source_statistics.npz") as archive:
        if tuple(archive["channel_names"]) != tuple(channels):
            raise ValueError("Statistics channel order mismatch")
        mean, std = archive["mean"].astype(np.float32), archive["std"].astype(
            np.float32
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 1e-6):
        raise ValueError("Invalid statistics")
    return mean, std
