"""THINGS-EEG2 windows with frozen source-training normalization, clipping and right padding. Only clean_early input is supported."""

from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Sequence
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from data_loader.things_eeg2_cache_identity import (
    validate_legacy_cache_rows,
    visual_identity_sha256,
)

WINDOW_MODES = ("clean_early",)
SPLITS = ("train", "val", "evaluation")
EXPECTED_EEG_CHANNELS = 63
EXPECTED_REVE_LENGTH = 200
EXPECTED_CLIP_SHAPE = (256, 1664)
EXPECTED_CLIP_STORAGE_DTYPE = np.dtype(np.float16)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | Path, *, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _normalize_int_selection(
    values: Sequence[int] | None, *, minimum: int, maximum: int, name: str
) -> tuple[int, ...] | None:
    if values is None:
        return None
    normalized = tuple(sorted({int(value) for value in values}))
    if not normalized:
        raise ValueError(f"{name} non può essere vuoto.")
    invalid = [value for value in normalized if value < minimum or value > maximum]
    if invalid:
        raise ValueError(f"{name}: valori fuori [{minimum}, {maximum}]: {invalid}")
    return normalized


@dataclass(frozen=True)
class RecordingInfo:
    recording_id: int
    subject: int
    session: int
    partition: str
    eeg_path: Path
    events_path: Path
    metadata_path: Path
    eeg_shape: tuple[int, int]
    channel_names: tuple[str, ...]
    trial_start: int
    trial_stop: int


@dataclass(frozen=True)
class DatasetSummary:
    split: str
    window_mode: str
    n_trials: int
    n_subjects: int
    n_sessions: int
    n_visuals: int
    n_concepts: int
    n_subject_visual_groups: int
    min_clean_valid_length: int
    canonical_test20_only: bool
    clip_shape: tuple[int, int, int]
    clip_dtype: str


class ThingsEEG2ReveDataset(Dataset[Dict[str, Any]]):
    """
    Dataset on-the-fly per il manifest prodotto da
    build_things_eeg2_index.py.

    Parameters
    ----------
    root:
        Radice THINGS-EEG2.
    clip_cache:
        Directory contenente image_tokens.npy oppure percorso diretto al file.
    split:
        train, val o test.
    window_mode:
        clean_early only.
    subjects, sessions:
        Filtri opzionali 1-based.
    allowed_concept_uids:
        Universo di classi determinato prima dell'uso degli split. La stessa
        sequenza deve essere passata a train, validation e test held-out.
    min_clean_valid_length:
        Filtro qualità applicato sempre al campo clean_valid_length.
    canonical_test20_only:
        Per split=test conserva soltanto le ripetizioni marcate nel manifest.
    return_eeg_target:
        Restituisce una copia esplicita di eeg per la reconstruction loss.
    return_clip_image:
        Carica e restituisce i token OpenCLIP. Può essere disabilitato per
        diagnostiche che usano soltanto l'EEG, evitando sia la cache sia una
        copia [256, 1664] per ogni trial.
    strict:
        Esegue controlli di hash, shape, dtype e ordine canali.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        clip_cache: str | Path,
        split: Literal["train", "val", "evaluation"],
        window_mode: Literal["clean_early"] = "clean_early",
        index_dir: str | Path | None = None,
        stats_root: str | Path | None = None,
        positions_path: str | Path | None = None,
        subjects: Sequence[int] | None = None,
        sessions: Sequence[int] | None = None,
        allowed_concept_uids: Sequence[str] | None = None,
        min_clean_valid_length: int = 38,
        canonical_test20_only: bool = True,
        clip_std: float = 15.0,
        output_length: int = EXPECTED_REVE_LENGTH,
        return_eeg_target: bool = True,
        return_clip_image: bool = True,
        strict: bool = True,
    ) -> None:
        super().__init__()
        if split not in SPLITS:
            raise ValueError(f"split deve essere uno di {SPLITS}: {split}")
        if window_mode not in WINDOW_MODES:
            raise ValueError(
                f"window_mode deve essere uno di {WINDOW_MODES}: {window_mode}"
            )
        if output_length != EXPECTED_REVE_LENGTH:
            raise ValueError("REVE richiede in questa pipeline output_length=200.")
        if min_clean_valid_length < 1 or min_clean_valid_length > 40:
            raise ValueError("min_clean_valid_length deve essere in [1, 40].")
        if clip_std <= 0:
            raise ValueError("clip_std deve essere > 0.")
        self.root = Path(root).expanduser().resolve()
        self.index_dir = (
            Path(index_dir).expanduser().resolve()
            if index_dir is not None
            else self.root / "portfolio_index"
        )
        self.stats_root = (
            Path(stats_root).expanduser().resolve()
            if stats_root is not None
            else self.root / "source_statistics"
        )
        self.positions_path = (
            Path(positions_path).expanduser().resolve()
            if positions_path is not None
            else self.index_dir / "reve_positions" / "things_eeg2_63_positions.npy"
        )
        self.split = split
        self.window_mode = window_mode
        self.subjects = _normalize_int_selection(
            subjects, minimum=1, maximum=255, name="subjects"
        )
        self.sessions = _normalize_int_selection(
            sessions, minimum=1, maximum=255, name="sessions"
        )
        self.allowed_concept_uids = (
            tuple((str(value) for value in allowed_concept_uids))
            if allowed_concept_uids is not None
            else None
        )
        if self.allowed_concept_uids is not None and len(
            set(self.allowed_concept_uids)
        ) != len(self.allowed_concept_uids):
            raise ValueError("allowed_concept_uids contiene duplicati.")
        self.min_clean_valid_length = int(min_clean_valid_length)
        self.canonical_test20_only = bool(canonical_test20_only)
        self.clip_std = float(clip_std)
        self.output_length = int(output_length)
        self.return_eeg_target = bool(return_eeg_target)
        self.return_clip_image = bool(return_clip_image)
        self.strict = bool(strict)
        self._validate_required_index_files()
        self._recordings = self._load_recordings()
        self._channel_names = self._validate_recording_channel_order()
        self._positions = self._load_positions()
        if self.return_clip_image:
            (self._clip_tokens_path, self._clip_mapping_path) = (
                self._resolve_clip_paths(clip_cache)
            )
            (self._clip_shape, self._clip_dtype) = self._validate_clip_cache()
        else:
            supplied_clip_path = Path(clip_cache).expanduser().resolve()
            if supplied_clip_path.suffix == ".npy":
                self._clip_tokens_path = supplied_clip_path
                self._clip_mapping_path = (
                    supplied_clip_path.parent / "visual_index_to_row.npy"
                )
            else:
                self._clip_tokens_path = supplied_clip_path / "image_tokens.npy"
                self._clip_mapping_path = supplied_clip_path / "visual_index_to_row.npy"
            index_metadata = json.loads(
                (self.index_dir / "index_metadata.json").read_text(encoding="utf-8")
            )
            visual_count = int(index_metadata["visual_count"])
            self._clip_shape = (visual_count, *EXPECTED_CLIP_SHAPE)
            self._clip_dtype = EXPECTED_CLIP_STORAGE_DTYPE
        trials_path = self.index_dir / "trials.npy"
        all_trials = np.load(trials_path, mmap_mode="r")
        split_visuals = np.load(
            self.index_dir / "splits" / f"{self.split}_visual_indices.npy"
        ).astype(np.int64, copy=False)
        (split_visuals, self._split_concept_uids) = (
            self._filter_split_visuals_by_concept(split_visuals)
        )
        mask = np.isin(all_trials["visual_index"], split_visuals, assume_unique=False)
        expected_partition_code = 0
        mask &= all_trials["partition_code"] == expected_partition_code
        mask &= all_trials["clean_valid_length"] >= self.min_clean_valid_length
        if self.subjects is not None:
            mask &= np.isin(
                all_trials["subject"], np.asarray(self.subjects, dtype=np.uint8)
            )
        if self.sessions is not None:
            mask &= np.isin(
                all_trials["session"], np.asarray(self.sessions, dtype=np.uint8)
            )
        self._trial_rows = np.flatnonzero(mask).astype(np.int64, copy=False)
        self._trials = np.array(all_trials[self._trial_rows], copy=True)
        del all_trials
        if len(self._trials) == 0:
            raise RuntimeError(
                "Nessun trial dopo i filtri. Controlla split, soggetti, sessioni e min_clean_valid_length."
            )
        self._validate_selected_trials(split_visuals)
        self._eeg_memmaps: dict[int, np.memmap] = {}
        self._clip_memmap: np.memmap | None = None
        self._clip_mapping: np.ndarray | None = None
        self._stats_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._subject_visual_groups: list[np.ndarray] | None = None
        self._subject_visual_group_keys: list[tuple[int, int]] | None = None
        from data_loader.protocol import load_statistics

        (self._source_mean, self._source_std) = load_statistics(
            self.stats_root,
            self._channel_names,
            self.index_dir,
            self.allowed_concept_uids,
        )

    def _validate_required_index_files(self) -> None:
        required = [
            self.index_dir / "trials.npy",
            self.index_dir / "recordings.json",
            self.index_dir / "visuals.csv",
            self.index_dir / "index_metadata.json",
            self.index_dir / "splits" / f"{self.split}_visual_indices.npy",
            self.positions_path,
        ]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "File indice mancanti:\n  "
                + "\n  ".join((str(path) for path in missing))
            )
        metadata = json.loads(
            (self.index_dir / "index_metadata.json").read_text(encoding="utf-8")
        )
        if int(metadata.get("recording_count", -1)) != 80 and self.strict:
            raise ValueError(
                f"index_metadata recording_count inatteso: {metadata.get('recording_count')}"
            )
        if int(metadata.get("visual_count", -1)) != 16740 and self.strict:
            raise ValueError(
                f"index_metadata visual_count inatteso: {metadata.get('visual_count')}"
            )
        if not math.isclose(
            float(metadata.get("target_sfreq", -1)), 200.0, rel_tol=0.0, abs_tol=1e-09
        ):
            raise ValueError("L'indice non è a 200 Hz.")

    def _load_recordings(self) -> tuple[RecordingInfo, ...]:
        payload = json.loads(
            (self.index_dir / "recordings.json").read_text(encoding="utf-8")
        )
        payload = sorted(payload, key=lambda item: int(item["recording_id"]))
        ids = [int(item["recording_id"]) for item in payload]
        if ids != list(range(len(payload))):
            raise ValueError("recording_id non contigui.")
        recordings: list[RecordingInfo] = []
        for item in payload:
            eeg_path = _resolve_path(item["eeg_path"], root=self.root)
            events_path = _resolve_path(item["events_path"], root=self.root)
            metadata_path = _resolve_path(item["metadata_path"], root=self.root)
            missing = [
                path
                for path in (eeg_path, events_path, metadata_path)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Recording {item['recording_id']} incompleta: {missing}"
                )
            shape = tuple((int(v) for v in item["eeg_shape"]))
            if len(shape) != 2 or shape[0] != EXPECTED_EEG_CHANNELS:
                raise ValueError(f"Shape recording inattesa: {shape}")
            recordings.append(
                RecordingInfo(
                    recording_id=int(item["recording_id"]),
                    subject=int(item["subject"]),
                    session=int(item["session"]),
                    partition=str(item["partition"]),
                    eeg_path=eeg_path,
                    events_path=events_path,
                    metadata_path=metadata_path,
                    eeg_shape=(shape[0], shape[1]),
                    channel_names=tuple((str(name) for name in item["channel_names"])),
                    trial_start=int(item["trial_start"]),
                    trial_stop=int(item["trial_stop"]),
                )
            )
        return tuple(recordings)

    def _validate_recording_channel_order(self) -> tuple[str, ...]:
        reference = self._recordings[0].channel_names
        if len(reference) != EXPECTED_EEG_CHANNELS:
            raise ValueError(
                f"Attesi {EXPECTED_EEG_CHANNELS} canali, trovati {len(reference)}."
            )
        if len(set(reference)) != len(reference):
            raise ValueError("Nomi canale duplicati.")
        for recording in self._recordings[1:]:
            if recording.channel_names != reference:
                raise ValueError(
                    f"Ordine canali differente nella registrazione {recording.recording_id}."
                )
        return reference

    def _load_positions(self) -> Tensor:
        positions = np.load(self.positions_path)
        positions = np.asarray(positions, dtype=np.float32)
        if positions.shape != (EXPECTED_EEG_CHANNELS, 3):
            raise ValueError(f"REVE positions shape inattesa: {positions.shape}")
        if not np.isfinite(positions).all():
            raise ValueError("REVE positions contiene NaN/Inf.")
        if np.unique(positions, axis=0).shape[0] != EXPECTED_EEG_CHANNELS:
            raise ValueError("REVE positions duplicate.")
        positions_json = self.positions_path.with_suffix(".json")
        if positions_json.is_file():
            metadata = json.loads(positions_json.read_text(encoding="utf-8"))
            names = tuple((str(name) for name in metadata["channel_names"]))
            if names != self._channel_names:
                raise ValueError(
                    "Ordine canali delle REVE positions diverso dalle registrazioni."
                )
            if self.strict:
                digest = _sha256_file(self.positions_path)
                expected = metadata.get("positions_sha256")
                if expected is not None and digest != expected:
                    raise ValueError("Hash REVE positions non valido.")
        return torch.from_numpy(np.array(positions, copy=True))

    def _filter_split_visuals_by_concept(
        self, split_visuals: np.ndarray
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        concept_by_visual: dict[int, str] = {}
        with (self.index_dir / "visuals.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                concept_by_visual[int(row["visual_index"])] = str(row["concept_uid"])
        self._concept_by_visual_index = concept_by_visual
        self._concept_to_index = {
            concept_uid: index
            for (index, concept_uid) in enumerate(
                sorted(set(concept_by_visual.values()))
            )
        }
        missing_visuals = [
            int(index) for index in split_visuals if int(index) not in concept_by_visual
        ]
        if missing_visuals:
            raise ValueError(
                f"Visual dello split mancanti da visuals.csv: {missing_visuals[:10]}"
            )
        if self.allowed_concept_uids is None:
            filtered = split_visuals
        else:
            allowed = set(self.allowed_concept_uids)
            filtered = np.asarray(
                [
                    int(index)
                    for index in split_visuals
                    if concept_by_visual[int(index)] in allowed
                ],
                dtype=np.int64,
            )
            present = {concept_by_visual[int(index)] for index in filtered}
            missing_concepts = allowed - present
            if self.split in {"train", "val"} and missing_concepts:
                raise ValueError(
                    f"Class subset non rappresentato interamente nello split {self.split}: {sorted(missing_concepts)[:10]}."
                )
        concepts = tuple(sorted({concept_by_visual[int(index)] for index in filtered}))
        if len(filtered) == 0:
            raise RuntimeError(
                f"Nessun visual nello split {self.split} dopo il filtro delle classi."
            )
        return (filtered, concepts)

    def _resolve_clip_paths(self, clip_cache: str | Path) -> tuple[Path, Path]:
        supplied = Path(clip_cache).expanduser().resolve()
        if supplied.is_dir():
            tokens = supplied / "image_tokens.npy"
            mapping = supplied / "visual_index_to_row.npy"
        else:
            tokens = supplied
            mapping = supplied.parent / "visual_index_to_row.npy"
        missing = [path for path in (tokens, mapping) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Cache OpenCLIP incompleta:\n  "
                + "\n  ".join((str(path) for path in missing))
            )
        return (tokens, mapping)

    def _validate_clip_cache(self) -> tuple[tuple[int, int, int], np.dtype]:
        tokens = np.load(self._clip_tokens_path, mmap_mode="r")
        shape = tuple((int(v) for v in tokens.shape))
        dtype = tokens.dtype
        mapping = np.load(self._clip_mapping_path, mmap_mode="r")
        if mapping.ndim != 1:
            raise ValueError("visual_index_to_row deve essere 1D.")
        if len(mapping) != shape[0]:
            raise ValueError("Mapping OpenCLIP e token non allineati.")
        if np.any(mapping < 0) or np.any(mapping >= shape[0]):
            raise ValueError("Mapping OpenCLIP fuori range.")
        if len(np.unique(mapping)) != len(mapping):
            raise ValueError("Mapping OpenCLIP non biunivoco.")
        if shape[1:] != EXPECTED_CLIP_SHAPE:
            raise ValueError(
                f"Token OpenCLIP shape inattesa: {shape}; attesa [N,{EXPECTED_CLIP_SHAPE[0]},{EXPECTED_CLIP_SHAPE[1]}]."
            )
        if dtype != EXPECTED_CLIP_STORAGE_DTYPE:
            raise ValueError(f"Token OpenCLIP dtype {dtype}; atteso float16.")
        metadata_path = self._clip_tokens_path.parent / "metadata.json"
        audit_path = self._clip_tokens_path.parent / "audit_report.json"
        if self.strict:
            if not metadata_path.is_file() or not audit_path.is_file():
                raise FileNotFoundError(
                    "metadata.json/audit_report.json mancanti dalla cache."
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if metadata.get("status") != "complete":
                raise ValueError("Cache OpenCLIP non complete.")
            if int(metadata.get("completed_rows", -1)) != shape[0]:
                raise ValueError("completed_rows OpenCLIP incoerente.")
            if audit.get("status") != "complete":
                raise ValueError("Audit OpenCLIP non complete.")
            if int(audit.get("nonfinite_values", -1)) != 0:
                raise ValueError("Audit OpenCLIP contiene non-finite.")
            if int(audit.get("all_zero_rows", -1)) != 0:
                raise ValueError("Audit OpenCLIP contiene righe nulle.")
            visuals_path = self.index_dir / "visuals.csv"
            signature = metadata.get("signature", {})
            expected_identity = signature.get("visual_identity_sha256")
            if expected_identity is not None:
                current_identity = visual_identity_sha256(visuals_path)
                if current_identity != expected_identity:
                    raise ValueError(
                        "La cache OpenCLIP appartiene a identità visuali diverse."
                    )
            else:
                expected_hash = signature.get("visuals_sha256")
                current_hash = _sha256_file(visuals_path)
                if expected_hash is None or current_hash != expected_hash:
                    cache_rows_path = self._clip_tokens_path.parent / "cache_rows.csv"
                    if not cache_rows_path.is_file():
                        raise ValueError(
                            "Hash visuals.csv diverso e cache_rows.csv mancante: impossibile validare la cache legacy."
                        )
                    try:
                        validate_legacy_cache_rows(
                            visuals_path=visuals_path, cache_rows_path=cache_rows_path
                        )
                    except ValueError as exc:
                        raise ValueError(
                            "La cache OpenCLIP appartiene a immagini diverse dal visuals.csv corrente."
                        ) from exc
        return (shape, dtype)

    def _validate_selected_trials(self, split_visuals: np.ndarray) -> None:
        if np.any(self._trials["recording_id"] >= len(self._recordings)):
            raise ValueError("Trial con recording_id fuori range.")
        if np.any(self._trials["visual_index"] >= self._clip_shape[0]):
            raise ValueError("Trial con visual_index fuori cache.")
        if np.any(self._trials["clean_valid_length"] > 40):
            raise ValueError("clean_valid_length > 40.")
        if np.any(self._trials["clean_valid_length"] < 1):
            raise ValueError("clean_valid_length < 1.")
        selected_visuals = np.unique(self._trials["visual_index"].astype(np.int64))
        if not np.all(np.isin(selected_visuals, split_visuals)):
            raise ValueError("Visual leakage nel filtro split.")
        for record in self._trials[:: max(1, len(self._trials) // 1024)]:
            recording = self._recordings[int(record["recording_id"])]
            if int(record["subject"]) != recording.subject:
                raise ValueError("Subject trial/recording incoerente.")
            if int(record["session"]) != recording.session:
                raise ValueError("Session trial/recording incoerente.")
            start = int(record["onset_sample"])
            if not 0 <= start < recording.eeg_shape[1]:
                raise ValueError("Onset fuori recording.")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_eeg_memmaps"] = {}
        state["_clip_memmap"] = None
        state["_clip_mapping"] = None
        state["_stats_cache"] = {}
        state["_subject_visual_groups"] = None
        state["_subject_visual_group_keys"] = None
        return state

    def close(self) -> None:
        self._eeg_memmaps.clear()
        self._clip_memmap = None
        self._clip_mapping = None
        self._stats_cache.clear()

    def _get_eeg_memmap(self, recording_id: int) -> np.memmap:
        array = self._eeg_memmaps.get(recording_id)
        if array is None:
            recording = self._recordings[recording_id]
            array = np.load(recording.eeg_path, mmap_mode="r")
            if array.shape != recording.eeg_shape:
                raise ValueError(
                    f"Shape EEG cambiata per {recording.eeg_path}: {array.shape} != {recording.eeg_shape}"
                )
            if array.dtype != np.float32:
                raise ValueError("Il continuo EEG deve essere float32.")
            self._eeg_memmaps[recording_id] = array
        return array

    def _get_clip_memmap(self) -> np.memmap:
        if self._clip_memmap is None:
            self._clip_memmap = np.load(self._clip_tokens_path, mmap_mode="r")
        return self._clip_memmap

    def _get_clip_mapping(self) -> np.ndarray:
        if self._clip_mapping is None:
            self._clip_mapping = np.load(self._clip_mapping_path, mmap_mode="r")
        return self._clip_mapping

    def __len__(self) -> int:
        return len(self._trials)

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self._channel_names

    @property
    def clip_tokens_path(self) -> Path:
        return self._clip_tokens_path

    @property
    def summary(self) -> DatasetSummary:
        (groups, _) = self.subject_visual_groups()
        subjects = np.unique(self._trials["subject"])
        sessions = np.unique(self._trials["session"])
        visuals = np.unique(self._trials["visual_index"])
        concepts = {self._concept_by_visual_index[int(index)] for index in visuals}
        return DatasetSummary(
            split=self.split,
            window_mode=self.window_mode,
            n_trials=len(self),
            n_subjects=len(subjects),
            n_sessions=len(sessions),
            n_visuals=len(visuals),
            n_concepts=len(concepts),
            n_subject_visual_groups=len(groups),
            min_clean_valid_length=self.min_clean_valid_length,
            canonical_test20_only=self.canonical_test20_only,
            clip_shape=self._clip_shape,
            clip_dtype=str(self._clip_dtype),
        )

    def original_trial_row(self, index: int) -> int:
        return int(self._trial_rows[index])

    def trial_record(self, index: int) -> np.void:
        return self._trials[index]

    def recording_info(self, recording_id: int) -> RecordingInfo:
        return self._recordings[int(recording_id)]

    def subject_visual_key(self, index: int) -> tuple[int, int]:
        record = self._trials[index]
        return (int(record["subject"]), int(record["visual_index"]))

    def concept_uid_for_visual(self, visual_index: int) -> str:
        try:
            return self._concept_by_visual_index[int(visual_index)]
        except KeyError as exc:
            raise KeyError(f"visual_index sconosciuto: {visual_index}") from exc

    def concept_index_for_visual(self, visual_index: int) -> int:
        """Return the stable dataset-wide integer identity of a concept."""
        concept_uid = self.concept_uid_for_visual(visual_index)
        return int(self._concept_to_index[concept_uid])

    def subject_visual_groups(self) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
        if (
            self._subject_visual_groups is not None
            and self._subject_visual_group_keys is not None
        ):
            return (self._subject_visual_groups, self._subject_visual_group_keys)
        groups: dict[tuple[int, int], list[int]] = {}
        for local_index, record in enumerate(self._trials):
            key = (int(record["subject"]), int(record["visual_index"]))
            groups.setdefault(key, []).append(local_index)
        ordered_keys = sorted(groups)
        ordered_groups = [
            np.asarray(groups[key], dtype=np.int64) for key in ordered_keys
        ]
        self._subject_visual_group_keys = ordered_keys
        self._subject_visual_groups = ordered_groups
        return (ordered_groups, ordered_keys)

    def _window_length(self, record: np.void) -> int:
        field = "clean_valid_length"
        length = int(record[field])
        if length < 1 or length > self.output_length:
            raise ValueError(f"{field}={length} non valido alla riga trial.")
        return length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self._trials[index]
        recording_id = int(record["recording_id"])
        recording = self._recordings[recording_id]
        subject = int(record["subject"])
        session = int(record["session"])
        visual_index = int(record["visual_index"])
        onset = int(record["onset_sample"])
        valid_length = self._window_length(record)
        stop = onset + valid_length
        if stop > recording.eeg_shape[1]:
            raise IndexError(
                f"Finestra oltre recording: {onset}:{stop} > {recording.eeg_shape[1]}"
            )
        continuous = self._get_eeg_memmap(recording_id)
        epoch = np.array(continuous[:, onset:stop], dtype=np.float32, copy=True)
        if epoch.shape != (EXPECTED_EEG_CHANNELS, valid_length):
            raise RuntimeError(f"Shape epoca inattesa {epoch.shape}.")
        (mean, std) = self._get_stats(subject, session)
        np.subtract(epoch, mean[:, None], out=epoch)
        np.divide(epoch, std[:, None], out=epoch)
        np.clip(epoch, -self.clip_std, self.clip_std, out=epoch)
        if not np.isfinite(epoch).all():
            raise FloatingPointError(f"EEG non finito: trial locale {index}.")
        padded = np.zeros((EXPECTED_EEG_CHANNELS, self.output_length), dtype=np.float32)
        padded[:, :valid_length] = epoch
        valid_mask = np.zeros(self.output_length, dtype=np.bool_)
        valid_mask[:valid_length] = True
        eeg_tensor = torch.from_numpy(padded)
        output: Dict[str, Any] = {
            "eeg": eeg_tensor,
            "valid_mask": torch.from_numpy(valid_mask),
            "positions": self._positions,
            "subject_id": subject,
            "subject_index": subject - 1,
            "session_id": session,
            "session_index": session - 1,
            "visual_index": visual_index,
            "concept_uid": self.concept_uid_for_visual(visual_index),
            "concept_index": self._concept_to_index[
                self.concept_uid_for_visual(visual_index)
            ],
            "event_id": int(record["event_id"]),
            "recording_id": recording_id,
            "event_index": int(record["event_index"]),
            "onset_sample": onset,
            "next_event_sample": int(record["next_event_sample"]),
            "valid_length": valid_length,
            "clean_valid_length": int(record["clean_valid_length"]),
            "post400_valid_length": int(record["post400_valid_length"]),
            "canonical_test20": bool(record["canonical_test20"]),
            "trial_row": self.original_trial_row(index),
            "dataset_index": int(index),
        }
        if self.return_clip_image:
            mapping = self._get_clip_mapping()
            cache_row = int(mapping[visual_index])
            clip_memmap = self._get_clip_memmap()
            clip_tokens = np.array(clip_memmap[cache_row], dtype=np.float16, copy=True)
            if clip_tokens.shape != EXPECTED_CLIP_SHAPE:
                raise RuntimeError(
                    f"Token shape inattesa alla riga {cache_row}: {clip_tokens.shape}"
                )
            if not np.isfinite(clip_tokens).all():
                raise FloatingPointError(
                    f"Token OpenCLIP non finiti alla riga {cache_row}."
                )
            output["clip_image"] = torch.from_numpy(clip_tokens)
            output["clip_cache_row"] = cache_row
        if self.return_eeg_target:
            output["eeg_target"] = eeg_tensor.clone()
        return output

    def _get_stats(self, subject, session):
        return (self._source_mean, self._source_std)


def masked_reconstruction_mse(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    reduction: Literal["mean", "none", "sum"] = "mean",
) -> Tensor:
    """
    MSE soltanto sui campioni EEG reali, normalizzata per sample.

    prediction/target:
        [B, C, T] oppure [C, T]
    valid_mask:
        [B, T] oppure [T]

    La normalizzazione per sample evita che i trial da 38/39 campioni abbiano
    peso inferiore a quelli da 40 campioni.
    """
    squeeze_batch = False
    if prediction.ndim == 2:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
        valid_mask = valid_mask.unsqueeze(0)
        squeeze_batch = True
    if prediction.ndim != 3:
        raise ValueError("prediction deve avere shape [B,C,T].")
    if target.shape != prediction.shape:
        raise ValueError(
            f"target shape {target.shape} != prediction {prediction.shape}."
        )
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask deve avere shape [B,T].")
    if valid_mask.shape[0] != prediction.shape[0]:
        raise ValueError("Batch valid_mask incoerente.")
    if valid_mask.shape[1] != prediction.shape[2]:
        raise ValueError("Tempo valid_mask incoerente.")
    if reduction not in {"mean", "none", "sum"}:
        raise ValueError("reduction deve essere mean, none o sum.")
    error = (prediction - target).square()
    mask = valid_mask[:, None, :].to(device=error.device, dtype=error.dtype)
    numerator = (error * mask).sum(dim=(1, 2))
    denominator = (mask.sum(dim=(1, 2)) * error.shape[1]).clamp_min(1)
    per_sample = numerator / denominator
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    if squeeze_batch:
        return per_sample.squeeze(0)
    return per_sample
